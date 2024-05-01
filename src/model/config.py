from peft.config import PeftConfig
from dataclasses import dataclass, field
from .schedule import BaseSchedule, OnceSchedule, PeriodicSchedule
from .allocator import BaseAllocator, TopKAllocator, ThresholdAllocator, MultinomialAllocator
import warnings
import torch
from typing import Literal, Optional, Union

@dataclass
class DynaLoraConfig(PeftConfig):
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
    schedule_type: str = field(
        default='no_schedule',
        metadata={'description': 'The schedule type that will determine how often the adapters are updated. By default, only once in the beginning.', 'choices': ['no_schedule', 'once;<after_step>', 'periodic;<interval>']})
    allocator_type: str = field(default='topk;1', metadata={'description': 'The allocator type that will determine how to select the active modules.', 'choices': ['topk;<k>', 'threshold;<T>', 'multinomial;<k>', 'scaled_multinomial;<k>']})
    aggregate_type: Literal['l2', 'l1'] = field(default='l2')

    def __post_init__(self):
        self.schedule = self.__parse_schedule__(self.schedule_type)
        self.allocator = self.__parse_allocator__(self.allocator_type)
        self.aggregator = self.__parse_aggregator__(self.aggregate_type)

    def __parse_schedule__(self, schedule_type: str) -> BaseSchedule:
        schedule, *args = schedule_type.split(';')
        if schedule == 'no_schedule':
            return BaseSchedule()
        elif len(args) == 0:
            warnings.warn('No arguments provided for schedule. Using no_schedule as default.')
            return BaseSchedule()
        elif schedule == 'once':
            return OnceSchedule(after_step=int(args[0]))
        elif schedule == 'periodic':
            return PeriodicSchedule(period=int(args[0]))
        else:
            warnings.warn('Invalid schedule type. Using no_schedule as default.')
            return BaseSchedule()

    def __parse_allocator__(self, allocator_type: str) -> BaseAllocator:
        allocator, *args = allocator_type.split(';')
        if len(args) == 0:
            warnings.warn('No arguments provided for allocator. Using topk;1 as default.')
            return TopKAllocator(k=1)
        elif allocator == 'topk':
            return TopKAllocator(k=int(args[0]))
        elif allocator == 'threshold':
            return ThresholdAllocator(T=float(args[0]))
        elif allocator == 'multinomial':
            return MultinomialAllocator(k=int(args[0]))
        elif allocator == 'scaled_multinomial':
            raise NotImplementedError('Scaled multinomial allocator is not implemented yet.')
        else:
            warnings.warn('Invalid allocator type. Using topk;1 as default.')
            return TopKAllocator(k=1)

    def __parse_aggregator__(self, aggregate_type: str):
        if aggregate_type == 'l2':
            return lambda x: torch.norm(x, p=2)
        elif aggregate_type == 'l1':
            return lambda x: torch.norm(x, p=1)
        else:
            warnings.warn('Invalid aggregate type. Using l2 as default.')
            return lambda x: torch.norm(x, p=2)