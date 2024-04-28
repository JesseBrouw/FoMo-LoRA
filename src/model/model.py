from peft import LoraModel
from .layer import DynaLoraLayer, Linear as DynaLoraLinear, dispatch_dynamic
from peft.config import PeftConfig
from typing import Union, List, Tuple, Dict

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

        

        # TODO: use the *configured* choice function to obtain the k modules we want to activate
        
        # TODO: where are the adapters kept in the LoRA model?
        # TODO: activate the k modules, deactivate the rest. this is easiest via linear search and
        #       O(1) lookup
        
