import dataclasses
import functools
import time

import torch
from torch.optim import AdamW
from datasets import load_dataset, load_metric
from peft import LoraConfig, VeraConfig, TaskType, get_peft_model
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from transformers.utils.generic import find_labels
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from .model.config import DynaLoraConfig
from .model.model import DinaLoraModel, DynaLoraModel
from .utils.helpers import (
    compute_metrics,
    preprocess_function_builder,
    remove_unused_columns,
)
from .utils.classes import ModelArguments
from .utils.wrapper import PeftModelWrapper
from transformers import HfArgumentParser

GLUE_TASKS = (
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
)


def get_model(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tokenizer


def get_config(
    lora,
    r,
    alpha,
    dropout,
    schedule_type=None,
    allocator_type=None,
    aggregate_type=None,
):
    match lora:
        case "vera":
            peft_config = VeraConfig(
                task_type=TaskType.SEQ_CLS,
                r=r,
                vera_dropout=dropout,
            )
        case "dynalora":
            peft_config = DynaLoraConfig(
                task_type=TaskType.SEQ_CLS,  # TODO: Add mapping for other task types
                inference_mode=False,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                schedule_type=schedule_type,
                allocator_type=allocator_type,
                aggregate_type=aggregate_type,
            )
        case "dinalora":
            peft_config = DynaLoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                schedule_type=schedule_type,
                allocator_type=allocator_type,
                aggregate_type=aggregate_type,
            )

        case _:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,  # TODO: Add mapping for other task types
                inference_mode=False,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
            )
    return peft_config


def load_dataset_metrics(task):
    # Load dataset and metric for the given task
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric("glue", actual_task)
    return dataset, metric


def main():
    args, hf_args = HfArgumentParser(
        (ModelArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    # we'll do this manually, based on the target model's forward signature
    hf_args.remove_unused_columns = False
    print(hf_args, args)

    task = args.task

    # Load dataset and metric for the given task
    dataset, metric = load_dataset_metrics(task)

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    model, tokenizer = get_model(args.model_name, num_labels)

    # find the label names in the model's signature
    # NOTE: we need this because the DynaLoraModel does not have a meaningful signature
    hf_args.label_names = find_labels(model.__class__)

    preprocess_function = preprocess_function_builder(task, tokenizer)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    if "train" in encoded_dataset.column_names:
        for key in encoded_dataset.column_names:
            encoded_dataset[key] = remove_unused_columns(
                encoded_dataset[key], model, hf_args.label_names or []
            )

    lora_config = get_config(
        args.lora,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        schedule_type=args.schedule_type,
        allocator_type=args.allocator_type,
        aggregate_type=args.aggregate_type,
    )
    match args.lora:
        case "dynalora":
            model = PeftModelWrapper(
                peft_model=DynaLoraModel(model, lora_config, "dynalora"),
                peft_config=lora_config,
                adapter_name="dynalora",
            )
            model.set_output_dir(hf_args.output_dir)
        case "dinalora":
            model = PeftModelWrapper(
                peft_model=DinaLoraModel(model, lora_config, "dinalora"),
                peft_config=lora_config,
                adapter_name="dinalora",
            )
        case _:
            model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    metric_name = (
        "pearson"
        if task == "stsb"
        else "matthews_correlation"
        if task == "cola"
        else "accuracy"
    )
    model_name = args.model_name.split("/")[-1]

    hf_args = dataclasses.replace(
        hf_args,
        output_dir=f"{hf_args.output_dir}/{model_name}-{args.lora}-finetuned-{task}",
        evaluation_strategy="epoch",
        metric_for_best_model=metric_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
    )

    # create an optimizer. One for each module
    optimizer = AdamW([
        {"params": adapter.parameters(), "lr": args.learning_rate}
        for adapter in model.get_base_model().adapter_modules
    ])

    validation_key = (
        "validation_mismatched"
        if task == "mnli-mm"
        else "validation_matched"
        if task == "mnli"
        else "validation"
    )

    trainer = Trainer(
        model,
        hf_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,  # Pass tokenizer again so that it pads correctly
        compute_metrics=functools.partial(compute_metrics, task=task, metric=metric),
        optimizers=(optimizer, None),
    )

    tick = time.perf_counter()

    class ProfCallback(TrainerCallback):
        def __init__(self, prof):
            self.prof = prof

        def on_step_end(self, args, state, control, **kwargs):
            self.prof.step()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            skip_first=3, wait=1, warmup=1, active=4, repeat=6
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(hf_args.output_dir),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        trainer.add_callback(ProfCallback(prof=prof))
        trainer.train()
    trainer.save_model(hf_args.output_dir)
    print(f"Training took {time.perf_counter() - tick:.2f}s")


if __name__ == "__main__":
    main()
