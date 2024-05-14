from peft import LoraModel
from peft.config import PeftConfig
from typing import Any, Callable, Union, List, Tuple, Dict
from transformers import PreTrainedModel

from .layer import DynaLoraLayer, Linear as DynaLoraLinear, dispatch_dynamic
from .layer import DinaLoraLayer, DinaLinear as DinaLoraLinear, dispatch_dynamic_dina
from .config import DynaLoraConfig

import os
import json

# TODO: this should just be a mixin
# Question: in which order does python look for method definitions?
# Answer: it looks in the class itself first, then in the parent classes in the order they are defined.

class AbstractMixin:
    dispatchers: Tuple[Callable]
    applicable_modules: Tuple[Any]

class BaseMixin(AbstractMixin):
    """
        Base Mixin class to account for injecting LoraLayers 
        and keeping track of their cumulative activations. 

        It overrides the _create_new_module function of BasePeftModel
    """
    def __init__(self, adapter_name: str, peft_config: Union[Dict[str,PeftConfig], PeftConfig]) -> None:
        self.adapter_name = adapter_name
        if isinstance(peft_config, PeftConfig):
            self.peft_config = peft_config
        else:
            self.peft_config = peft_config[adapter_name]
        if not hasattr(self.peft_config, 'allocator') or not hasattr(self.peft_config, 'schedule'):
            raise ValueError(
                "The PeftConfig object must have a schedule and an allocator attribute."
            )
        self.schedule = self.peft_config.schedule
        self.allocator = self.peft_config.allocator
        # initialize the active modules
        self._init_modules()

    def set_output_dir(self, output_dir):
        # initialize logging file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_path = os.path.join(output_dir, "dynalora_logs.json")
        data = {"schedule": self.peft_config["dynalora"].schedule_type, "allocator": self.peft_config["dynalora"].allocator_type, "aggregate": self.peft_config["dynalora"].aggregate_type,
                "cum_acts": [], "masks": []}
        with open(self.output_path, "w") as f:
            json.dump(data, f)

    @classmethod
    def _create_new_module(cls, lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = cls.dispatchers

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def __call__(self, *args, **kwargs):
        # Realloc if needed, then step the scheduler.
        if self.schedule.reallocate:
            self.reassign_active_modules()
        self.schedule.step()

    def _init_modules(self):
        """
            Randomly select modules to activate
        """

        if not hasattr(self, "adapter_modules"):
            self.adapter_modules = self._find_adapter_modules()
        # use the *configured* choice function to obtain the k modules we want to activate
        mask = self.allocator([1 for mod in self.adapter_modules])
        for mod, msk in zip(self.adapter_modules, mask):
            if msk:
                mod.activate()
            else:
                mod.deactivate()

    def reassign_active_modules(self):
        """
        Reassigns the active modules to the model
        """
        if not isinstance(self, LoraModel) or not hasattr(self, "model"):
            raise ValueError(
                "This method is only supported for LoraModel instances, for now."
            )

        if not hasattr(self, "adapter_modules"):
            self.adapter_modules = self._find_adapter_modules()

        # use the *configured* choice function to obtain the k modules we want to activate
        cum_acts = [mod.cum_acts for mod in self.adapter_modules]
        mask = self.allocator(cum_acts)

        # activate the k modules, deactivate the rest.
        #   this is easiest via linear search and
        #   O(1) lookup
        for mod, msk in zip(self.adapter_modules, mask):
            if msk:
                mod.activate()
            else:
                mod.deactivate()
                
        # log to json
        with open(self.output_path, "r") as f:
            data = json.load(f)
            data["cum_acts"].append([cum_act.item() for cum_act in cum_acts])
            data["masks"].append(mask.tolist())
        with open(self.output_path, "w") as f:
            json.dump(data, f)
            

    def _find_adapter_modules(self):
        """
        Find all adapter modules in the model
        """
        adapter_modules = []
        for _, module in self.model.named_modules():
            if isinstance(module, self.applicable_modules):
                adapter_modules.append(module)
        return adapter_modules

# DynaLoRA
class DynaLoraMixin(BaseMixin):
    """
        Mixin class to account for injecting DinaLoraLayers 
        and keeping track of their cumulative activations., 
        
        it overrides the _create_new_module function of BasePeftModel
    """
    dispatchers = (dispatch_dynamic,)
    applicable_modules = (DynaLoraLayer,)

class DynaLoraModel(LoraModel, DynaLoraMixin):
    def __init__(self,
                 model: PreTrainedModel,
                 peft_config: PeftConfig,
                 adapter_name: str) -> None:
        LoraModel.__init__(self, model, peft_config, adapter_name)
        DynaLoraMixin.__init__(self, adapter_name, peft_config)

    def __call__(self, *args, **kwargs):
        # first, see if reallocation is due
        DynaLoraMixin.__call__(self, *args, **kwargs)
        # then, perform the forward pass
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        return DynaLoraMixin._create_new_module(lora_config, adapter_name, target, **kwargs)



## DinaLoRA
class DinaLoraMixin(BaseMixin):
    """
        Mixin class to account for injecting DinaLoraLayers 
        and keeping track of their cumulative activations., 
        
        it overrides the _create_new_module function of BasePeftModel
    """
    dispatchers = (dispatch_dynamic_dina,)
    applicable_modules = (DinaLoraLayer,)


class DinaLoraModel(LoraModel, DinaLoraMixin):
    def __init__(self,
                 model: PreTrainedModel,
                 peft_config: PeftConfig,
                 adapter_name: str) -> None:
        LoraModel.__init__(self, model, peft_config, adapter_name)
        DinaLoraMixin.__init__(self, adapter_name, peft_config)

    def __call__(self, *args, **kwargs):
        # first, see if reallocation is due
        DinaLoraMixin.__call__(self, *args, **kwargs)
        # then, perform the forward pass
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        return DinaLoraMixin._create_new_module(lora_config, adapter_name, target, **kwargs)
