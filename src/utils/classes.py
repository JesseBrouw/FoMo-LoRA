import dataclasses


@dataclasses.dataclass
class ModelArguments:
    """
    Represents the arguments for a model.

    Attributes:
        model_name (str): The name of the model. Default is "roberta-base".
        task (str): The task associated with the model. Default is "cola".
        r (int): The value of r. Default is 8.
        alpha (float): The value of alpha. Default is 0.5.
        dropout (float): The dropout rate. Default is 0.1.
        lora (str): The lora value. Default is "lora".
        batch_size (int): The batch size. Default is 64.
    """

    model_name: str = dataclasses.field(
        default="roberta-base", metadata={"help": "The name of the model."}
    )
    task: str = dataclasses.field(
        default="cola", metadata={"help": "The task associated with the model."}
    )
    lora_r: int = dataclasses.field(default=8, metadata={"help": "The value of r."})
    lora_alpha: float = dataclasses.field(
        default=0.5, metadata={"help": "The value of alpha."}
    )
    lora_dropout: float = dataclasses.field(
        default=0.1, metadata={"help": "The dropout rate."}
    )
    lora: str = dataclasses.field(default="lora", metadata={"help": "The lora value."})
    batch_size: int = dataclasses.field(
        default=64, metadata={"help": "The batch size."}
    )
    epochs: float = dataclasses.field(
        default=5.0, metadata={"help": "The number of epochs."}
    )
    device: str = dataclasses.field(
        default="cuda", metadata={"help": "The device to use."}
    )
    dynalora: bool = dataclasses.field(
        default=False, metadata={"help": "Whether to use DynaLoRA."},
        kw_only=True, repr=True
    )
    schedule_type: str = dataclasses.field(
        default="no_schedule", metadata={"help": "The schedule type."}
    )
    allocator_type: str = dataclasses.field(
        default="topk;1", metadata={"help": "The allocator type."}
    )
    aggregate_type: str = dataclasses.field(
        default="l2", metadata={"help": "The aggregate type."}
    )