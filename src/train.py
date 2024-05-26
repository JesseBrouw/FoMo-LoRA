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
from .model.config import DynaLoraConfig, DynaVeraConfig
from .model.model import DinaLoraModel, DynaLoraModel, DynaVeraModel
from .model.optimizer import (
    create_layerwise_optimizer_and_scheduler,
    SUPPORTED_OPTIMIZERS
    )

from .utils.helpers import (
    compute_metrics,
    preprocess_function_builder,
    remove_unused_columns,
)
from .utils.classes import ModelArguments
from .utils.wrapper import PeftModelWrapper
from transformers import HfArgumentParser
from transformers import RobertaForSequenceClassification

import logging
logging.basicConfig(level=logging.INFO)


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
    vera_d_initial=0.1,
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
                d_initial=vera_d_initial,
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
        case "dynavera":
            peft_config = DynaVeraConfig(
                task_type=task_type,
                inference_mode=False,
                r=r,
                vera_dropout=dropout,
                d_initial=vera_d_initial,
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
        vera_d_initial=args.vera_d_initial,
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
                model = PeftModelWrapper(
                    peft_model=DinaLoraModel(model, lora_config),
                    peft_config=lora_config
                )
            else:
                print("Task type not supported for DinaLora. Only Sequence classification is supported yet.")
                exit(1)
        case "dynavera":
            if lora_config.task_type==TaskType.SEQ_CLS:
                model = PeftModelWrapper(
                    peft_model=DynaVeraModel(model, lora_config),
                    peft_config=lora_config
                )
            else:
                print("Task type not supported for DynaVera. Only Sequence classification is supported yet.")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # build the optimizer and scheduler
    optimizer_arg = (None, None) # default to Trainer's constructors
    if args.use_layerwise_optim and hf_args.optim.lower() in SUPPORTED_OPTIMIZERS:
        optimizer_arg = create_layerwise_optimizer_and_scheduler(
            model,
            hf_args,
            num_training_steps=len(encoded_dataset["train"]) * hf_args.num_train_epochs,
            num_warmup_steps=(hf_args.warmup_steps if hf_args.warmup_steps > 0 else \
                              hf_args.num_train_epochs * len(encoded_dataset["train"]) * hf_args.warmup_ratio),
        )
        
    # initialize the modules
    getattr(model, "init_modules", lambda: None)()

    trainer = Trainer(
        model,
        hf_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,  # Pass tokenizer again so that it pads correctly
        compute_metrics=functools.partial(compute_metrics, task=task, metric=metric),
        optimizers=optimizer_arg,
    )

    tick = time.time()

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
#    trainer.train()
    print(f"Training took {time.time() - tick:.1f}s")
    trainer.save_model(os.path.join(hf_args.output_dir, 'final_model'))
    trainer.evaluate(encoded_dataset["test"])


if __name__ == "__main__":
    main()
