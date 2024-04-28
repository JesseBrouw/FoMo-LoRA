from peft.config import PeftConfig
from dataclasses import dataclass, field
from .schedule import BaseSchedule, OnceSchedule, PeriodicSchedule
import warnings
from typing import Literal, Optional, Union

@dataclass
def DynaLoraConfig(PeftConfig):
    """
        Configuration for the dynamic LoRA model.
        
        Parameters:
        - schedule: BaseSchedule
            The schedule type that will determine how ofter the adapters are updated.
            by default, only once in the beginning.
        - aggregate_type: str
            The type of aggregation to use for the cumulative activations. 
            Currently, only 'l2' is supported.
        - k: int 
            The number of active modules in the model.
            If k is set, k_percent is ignored.
        - k_percent: float
            The percentage of active modules in the model.
            If k is set, k_percent is ignored.
    """
    schedule: BaseSchedule = field(default_factory=BaseSchedule)
    aggregate_type: Literal['l2', 'l1'] = field(default='l2')
    k: int = field(default=None)
    k_percent: float = field(default=None)

    def __post_init__(self):
        if self.k is None and self.k_percent is None:
            raise ValueError("Either k or k_percent must be set.")
        if self.k is not None and self.k_percent is not None:
            warnings.warn("Both k and k_percent are set. k will be used.")
        if self.k < 0:
            raise ValueError("k must be a non-negative integer.")
        if self.k_percent < 0 or self.k_percent > 1:
            raise ValueError("k_percent must be in the range [0, 1].")
        # Thats enough for now
