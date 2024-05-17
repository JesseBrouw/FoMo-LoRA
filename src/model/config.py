from peft import LoraConfig
from dataclasses import dataclass, field
from .schedule import BaseSchedule, OnceSchedule, PeriodicSchedule
from .allocator import (BaseAllocator,
                        TopKAllocator,
                        ThresholdAllocator,
                        MultinomialAllocator,
                        ScaledMultinomialAllocator)
import warnings
import torch
from typing import Literal, Optional, Union

@dataclass
class DynaLoraConfig(LoraConfig):
    """
        Configuration for the dynamic LoRA model.

        Attributes:
        - schedule_type (str): The schedule type that will determine how often the adapters are updated.
        By default, only once in the beginning. Possible choices are 'no_schedule', 'once;<after_step>', 'periodic;<interval>'.
        - allocator_type (str): The allocator type that will determine how to select the active modules.
        Possible choices are 'topk;<k>', 'threshold;<T>', 'multinomial;<k>', 'scaled_multinomial;<k>;<gamma>'.
        - aggregate_type (Literal['l2', 'l1']): The type of aggregation to use for the cumulative activations.
        Currently, only 'l2' and 'l1' are supported.
    """
    schedule_type: str = field(
        default='no_schedule',
        metadata={'help': ("The schedule type that will determine how often the adapters are updated. By default, only once in the beginning. Choices: ['no_schedule', 'once;<after_step>', 'periodic;<interval>']")})
    allocator_type: str = field(default='topk;1', metadata={'help': ("The allocator type that will determine how to select the active modules. Choices: ['topk;<k>', 'threshold;<T>', 'multinomial;<k>', 'scaled_multinomial;<k>;<gamma>']")})
    aggregate_type: Literal['l2', 'l1'] = field(default='l2', 
                                                metadata={'help': ("The type of aggregation to use for the cumulative activations. Choices: ['l2', 'l1']")})

    def __post_init__(self):
        super().__post_init__()
        if any(map(lambda x: x is None,
                   [self.schedule_type, self.allocator_type, self.aggregate_type])):
            warnings.warn('DynaLoraConfig called but missing required arguments.' \
                  'Initializing as a LoraConfig.')
            return
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
            return ThresholdAllocator(threshold=float(args[0]))
        elif allocator == 'multinomial':
            return MultinomialAllocator(k=int(args[0]))
        elif allocator == 'scaled_multinomial':
            if len(args) < 2:
                raise ValueError('Scaled multinomial allocator requires two arguments.')
            return ScaledMultinomialAllocator(k=int(args[0]), gamma=float(args[1]))
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