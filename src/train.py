# %% [markdown]
# # OwOLoRA
#
# ## Problem
#
# LoRA finetuning has emerged as a popular technique for training large language models. Instead of training a model from scratch, we can take a pre-trained model and fine-tune it on a smaller dataset.

import dataclasses
import functools

import torch
from datasets import load_dataset, load_metric
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from .utils.helpers import compute_metrics, preprocess_function_builder
from .utils.classes import ModelArguments
from transformers import HfArgumentParser

GLUE_TASKS = (    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli")

def get_model(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tokenizer


def load_dataset_metrics(task):
    # Load dataset and metric for the given task
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric("glue", actual_task)
    return dataset, metric


def main():
    args, hf_args = HfArgumentParser((ModelArguments, TrainingArguments)).parse_args_into_dataclasses()
    
    task = args.task

    # Load dataset and metric for the given task
    dataset, metric = load_dataset_metrics(task)

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    model, tokenizer = get_model(args.model_name, num_labels)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # TODO: Add mapping for other task types
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

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
    
    hf_args = dataclasses.replace(hf_args, output_dir=f"{hf_args.output_dir}/{model_name}-{args.lora}-finetuned-{task}", evaluation_strategy="epoch", metric_for_best_model=metric_name, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, save_strategy="epoch")
    
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

    trainer.train()
    trainer.save_model(f"{args.output_dir}/{model_name}-finetuned-{task}")

if __name__ == "__main__":
    main()