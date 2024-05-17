import numpy as np
import inspect


def preprocess_function_builder(task, tokenizer):
    """Dynamically build a preprocess function based on the task"""
    # To preprocess our dataset, we will thus need the names of the columns containing the sentence(s). The following dictionary keeps track of the correspondence task to column names:
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key], truncation=True
        )

    return preprocess_function

def compute_metrics(eval_pred, task, metric):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def get_model_signature(target_model, label_names = []):
    """Get the signature of the target model"""
    signature = inspect.signature(target_model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += list(
        set(["label", "label_ids"] + label_names))
    return signature_columns

def remove_unused_columns(dataset, target_model, label_names = []):
    """Remove columns that are not used for the task

    NOTE: This is a simplified version of transformers.trainer.Trainer._remove_unused_columns
    """
    columns_needed = get_model_signature(target_model, label_names)
    columns_not_needed = set(dataset.column_names) - set(columns_needed)
    return dataset.remove_columns(columns_not_needed)
