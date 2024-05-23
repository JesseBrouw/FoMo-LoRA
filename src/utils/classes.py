from typing import Optional, Union
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
    target_modules: Optional[Union[list[str], str]] = dataclasses.field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    batch_size: int = dataclasses.field(
        default=64, metadata={"help": "The batch size."}
    )
    epochs: float = dataclasses.field(
        default=5.0, metadata={"help": "The number of epochs."}
    )
    device: str = dataclasses.field(
        default="cuda", metadata={"help": "The device to use."}
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
    use_layerwise_optim: bool = dataclasses.field(
        default=False,
        metadata={'help': ("Whether to use layer-wise optimization. Default is False.")},
        repr=True)
    vera_d_initial: float = dataclasses.field(
        default=0.1,
        metadata={"help": "The value of d for VERA."}
    ) # default and we don't need to touch it. Adding for completeness
    check_activations: bool = dataclasses.field(
        default=False,
        metadata={"help": "Whether to check activations."}
    )
