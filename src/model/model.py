from peft import LoraModel
from peft.config import PeftConfig
from typing import Union, List, Tuple, Dict
from transformers import PreTrainedModel

from .layer import DynaLoraLayer, Linear as DynaLoraLinear, dispatch_dynamic
from .config import DynaLoraConfig

# TODO: this should just be a mixin
# Question: in which order does python look for method definitions?
# Answer: it looks in the class itself first, then in the parent classes in the order they are defined
class DynaLoraMixin:
    """
        Mixin class to account for injecting DynaLoraLayers 
        and keeping track of their cumulative activations., 
        
        it overrides the _create_new_module function of BasePeftModel
    """
    def __init__(self, adapter_type: str, peft_config: Union[Dict[str,PeftConfig], PeftConfig]) -> None:
        self.adapter_type = adapter_type
        if isinstance(peft_config, PeftConfig):
            self.peft_config = peft_config
        else:
            self.peft_config = peft_config[adapter_type]
        self.schedule = self.peft_config.schedule
        self.allocator = self.config.allocator

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = [dispatch_dynamic]

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

    def forward(self, *args, **kwargs):
        # Realloc if needed, then step the scheduler.
        if self.schedule.reallocate:
            self.reassign_active_modules()
        self.config.schedule.step()

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
        mask = self.allocator([mod.cum_acts for mod in self.adapter_modules])

        # TODO: activate the k modules, deactivate the rest.
        #       this is easiest via linear search and
        #       O(1) lookup
        for mod, msk in zip(self.adapter_modules, mask):
            if msk:
                mod.activate()
            else:
                mod.deactivate()

    def _find_adapter_modules(self):
        """
        Find all adapter modules in the model
        """
        adapter_modules = []
        for _, module in self.model.named_modules():
            if isinstance(module, DynaLoraLayer):
                adapter_modules.append(module)
        return adapter_modules

class DynaLoraModel(LoraModel, DynaLoraMixin):
    def __init__(self,
                 model: PreTrainedModel,
                 lora_config: PeftConfig,
                 dynalora_config: DynaLoraConfig,
                 adapter_name: str) -> None:

        LoraModel.__init__(self, model, lora_config, adapter_name)
        DynaLoraMixin.__init__(self, model, dynalora_config, adapter_name)

    def forward(self, *args, **kwargs):
        # first, see if reallocation is due
        DynaLoraMixin.forward(self, *args, **kwargs)
        # then, perform the forward pass
        LoraModel.forward(self, *args, **kwargs)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        return DynaLoraMixin._create_new_module(lora_config, adapter_name, target, **kwargs)
