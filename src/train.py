import dataclasses
import functools
import time
import os

import torch
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
from transformers import RobertaForSequenceClassification

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

def maybe_select_all_linear(model, target_modules):
    isroberta = isinstance(model, RobertaForSequenceClassification)
    if target_modules == 'all-linear':
        if not isroberta:
            raise ValueError(
                "all-linear is only supported for roberta-base for now.")
        else:
            return ["key", "query", "value",
                    "attention.output.dense",
                    "intermediate.dense", "output.dense"]
    else:
        return target_modules

def get_modules_to_save(task_type):
    # this is needed because otherwise peft won't save the classifier weights to the checkpoints
    # this mechanism is implemented in PeftModelForSequenceClassification class which is a subclass of PeftModel,
    # but the PeftModel.__init__() cannot be called from the PeftModelWrapper because there are some stuffs
    # which are not compatible with DynaLoraModel (Miki feel free to detail which are those).
    # Instead the stuffs needed from PeftModel are implemented in PeftModelWrapper.
    if task_type == TaskType.SEQ_CLS:
        return ["classifier", "score"]
    else:
        raise ValueError(f"Task type {task_type} is not supported yet.")

def get_config(
    lora,
    r,
    alpha,
    dropout,
    model,
    target_modules=None,
    schedule_type=None,
    allocator_type=None,
    aggregate_type=None,
    task_type=TaskType.SEQ_CLS,
):
    # find target modules
    target_modules = maybe_select_all_linear(model, target_modules)
    # find modules to save
    modules_to_save = get_modules_to_save(task_type)
    match lora:
        case "vera":
            peft_config = VeraConfig(
                task_type=task_type,
                r=r,
                vera_dropout=dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save
            )
        case "dynalora":
            peft_config = DynaLoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                schedule_type=schedule_type,
                allocator_type=allocator_type,
                aggregate_type=aggregate_type,
                modules_to_save=modules_to_save
            )
        case "dinalora":
            peft_config = DynaLoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                schedule_type=schedule_type,
                allocator_type=allocator_type,
                aggregate_type=aggregate_type,
                modules_to_save=modules_to_save
            )

        case _:
            peft_config = LoraConfig(
                task_type=task_type,  # TODO: Add mapping for other task types
                inference_mode=False,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save
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
        model=model,
        target_modules=args.target_modules,
        schedule_type=args.schedule_type,
        allocator_type=args.allocator_type,
        aggregate_type=args.aggregate_type,
    )
    match args.lora:
        case "dynalora":
            if lora_config.task_type==TaskType.SEQ_CLS:
                model = PeftModelWrapper(
                    peft_model=DynaLoraModel(model, lora_config),
                    peft_config=lora_config
                )
            else:
                print("Task type not supported for DynaLora. Only Sequence classification is supported yet.")
                exit(1)
            model.set_output_dir(hf_args.output_dir)
        case "dinalora":
            if lora_config.task_type==TaskType.SEQ_CLS:
                # this is needed because otherwise peft won't save the classifier weights to the checkpoints
                # this mechanism is implemented in PeftModelForSequenceClassification class which is a subclass of PeftModel,
                # but the PeftModel.__init__() cannot be called from the PeftModelWrapper because there are some stuffs 
                # which are not compatible with DynaLoraModel (Miki feel free to detail which are those).
                # Instead the stuffs needed from PeftModel are implemented in PeftModelWrapper. 
                lora_config.modules_to_save = ["classifier", "score"]
                model = PeftModelWrapper(
                    peft_model=DinaLoraModel(model, lora_config),
                    peft_config=lora_config
                )
            else:
                print("Task type not supported for DinaLora. Only Sequence classification is supported yet.")
                exit(1)
        case _:
            model = get_peft_model(model, lora_config)
    print(model)
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

    # initialize modules (if needed)
    getattr(model, "init_modules", lambda: None)()
    print('initialized modules')
    model.print_trainable_parameters()

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
        trainer.train(resume_from_checkpoint=hf_args.resume_from_checkpoint)
    trainer.save_model(os.path.join(hf_args.output_dir, 'final')) # otherwise it gets messy
    print(f"Training took {time.perf_counter() - tick:.2f}s")


if __name__ == "__main__":
    main()
