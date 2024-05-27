"""
    When doing dynamic allocation of resources, it makes a lot of sense 
    to groups PEFT layers which adapt the same pre-trained layer.

    In LoRA, for example, we can activate/deactivate the A and B matrices
    of the same layer together. This is because they are always used together
    and also if we only ever update B, and A remains at initial value,
    the adapter will not actually learn anything (A=0 initially).

    The BaseModuleGroup and its subclasses achieve this by grouping layers
    based on their names. The subclasses define the applicable layers
    and the BaseModuleGroup provides the functionality to activate/deactivate
    the layers in the group.

    We can inject this seamlessly through a switch in the corresponding DynaLoraModel's init method.
    Only thing we need to ensure is that the allocator receives a relevant signal
    to decide which group to activate/deactivate. 

    We do this by adding property functions `cum_acts` and `counter` to the base class
    and defining `collate_cum_acts` and `collate_counter` in the allocator.
"""
from typing import List, Tuple, Dict, Any
import torch

from .layer import DinaLoraLayer, DynaLoraLinear, DynaVeraLinear

import logging
logger = logging.getLogger(__name__)

class BaseModuleGroup:
    applicable_layers: Tuple[str]
    def __init__(self, name: str) -> None:
        self.name = name
        self.modules: Dict[torch.nn.Module] = {}

    def add_module(self, base_name: str, module: torch.nn.Module) -> None:
        """
        Adds module to this group. 
        Parameters:
        - base_name: the string corresponding to one of the applicable layers.
                     setting the same base_name will result in an error
        - module: the module to add to the group
        """
        if base_name in self.modules:
            logger.error(f'Trying to add a module with the same base_name {base_name} to the group. Ignoring.')
            return
        self.modules[base_name] = module

    def sanity_check(self) -> None:
        """
        Checks if the group has all the required modules.
        """
        for layer in self.applicable_layers:
            if layer not in self.modules:
                logger.error(f'Missing module for layer {layer} in the group {self.name}.')
                return

    # interfaces to access the different functions and properties of the
    # modules in the group

    def activate(self) -> None:
        for module in self.modules.values():
            act_fn = getattr(module, 'activate', None)
            if act_fn is None:
                logger.error('Trying to activate a module that does not have an activate method.')
                continue
            act_fn()

    def deactivate(self) -> None:
        for module in self.modules.values():
            deact_fn = getattr(module, 'deactivate', None)
            if deact_fn is None:
                logger.error('Trying to activate a module that does not have a deactivate method.')
                continue
            deact_fn()

    def reset_cum_acts(self) -> None:
        for module in self.modules.values():
            reset_fn = getattr(module, 'reset_cum_acts', None)
            if reset_fn is None:
                logger.error('Trying to reset cum_acts on a module that does not have a reset_cum_acts method.')
                continue
            reset_fn()

    @property
    def cum_acts(self) -> List[int]:
        return self._collate_cum_acts()

    @property
    def counter(self) -> List[int]:
        return self._collate_counter()

    def _collate_cum_acts(self) -> List[int]:
        raise NotImplementedError('This method should be implemented by the derived class.')
    def _collate_counter(self) -> List[int]:
        raise NotImplementedError('This method should be implemented by the derived class.')

class DynaLoraModuleGroup(BaseModuleGroup):
    applicable_layers = ('lora_A', 'lora_B')

    def _collate_counter(self) -> List[int]:
        return self.modules['lora_A'][0].counter
    def _collate_cum_acts(self) -> List[int]:
        # todo: figure out something smarter maybe
        # could also be an average, doesn't matter as long as don't overflow
        return sum([module.cum_acts for module in self.modules.values()])

# inherit DynaLoraModuleGroup because the collate functions are identical anyway
class DynaVeraModuleGroup(DynaLoraModuleGroup):
    applicable_layers = ('lambda_A', 'lambda_B')

class DinaLoraModuleGroup(BaseModuleGroup):
    applicable_layers = ('lora_A', 'lora_B')

    def _collate_counter(self) -> List[int]:
        return self.modules['lora_A'].counter
    def _collate_cum_acts(self) -> List[int]:
        # imitate the chain rule for the cum_acts
        # the gradient of the A matrix is given by
        # dL/dA = dL/dh * dh/dA = dL/dh * x'.T
        # the gradient of B matrix is given by
        # dL/dB = dL/dx' * x.T = dL/dh * dh/dx' * x.T
        # h being the outupt and x' = Bx
        # combining the two, we get
        # dL/dBA = dL/dh * x.T, the gradient of the combined matrix
        # since we do not know dh/dx', we cannot calculate the combined gradient.
        # so its easier to just sum the cum_acts of the two matrices
        return sum([module.cum_acts for module in self.modules.values()])