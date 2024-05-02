import dataclasses
import functools
import time

import torch
from datasets import load_dataset, load_metric
from peft import LoraConfig, VeraConfig, TaskType, get_peft_model
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from .model.config import DynaLoraConfig
from .model.model import DynaLoraModel
from .utils.helpers import compute_metrics, preprocess_function_builder
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

def get_config(lora, r, alpha, dropout, schedule_type=None, allocator_type=None, aggregate_type=None):
    peft_config = DynaLoraConfig(
        task_type=TaskType.SEQ_CLS,  # TODO: Add mapping for other task types
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        schedule_type=schedule_type,
        allocator_type=allocator_type,
        aggregate_type=aggregate_type
    ) if lora == "lora" else VeraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        vera_dropout=dropout,
        # TODO: support for VeRA
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
    print(hf_args, args)

    task = args.task

    # Load dataset and metric for the given task
    dataset, metric = load_dataset_metrics(task)

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    model, tokenizer = get_model(args.model_name, num_labels)

    if args.dynalora:
        lora_config = get_config(
            args.lora,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
            schedule_type=args.schedule_type,
            allocator_type=args.allocator_type,
            aggregate_type=args.aggregate_type,
        )
        model = PeftModelWrapper(peft_model=DynaLoraModel(model, lora_config, 'dynalora'),
                                 peft_config=lora_config, 
                                 adapter_name='dynalora')
    else:
        peft_config = get_config(args.lora, args.lora_r, args.lora_alpha, args.lora_dropout)
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    preprocess_function = preprocess_function_builder(task, tokenizer)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

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

    # args = TrainingArguments(
    #     f"{args.output_dir}/{model_name}-{args.lora}-finetuned-{task}",
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=args.learning_rate,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     num_train_epochs=args.epochs,
    #     seed=args.seed,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     metric_for_best_model=metric_name,
    # )

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
            skip_first=3, wait=1, warmup=1, active=2, repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("hf-training-trainer"),
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
