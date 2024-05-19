from peft.peft_model import PeftModel
from torch.nn import Module
import os
from typing import Any
import torch

from ..model.model import BaseMixin as DynaBaseMixin

# FIXME: if we push to the PEFT library this wrapper will need to go away.
#        This is a temporary solution to allow for the dynamic LoRA model to be used
#        for example for printing the number of trainable params
class PeftModelWrapper(PeftModel, Module):
    """
        Wrapper class for the PeftModel.
        
        We use this class to wrap a DynaLoraModel instance
        and still be able to use all benefits of the PeftModel.
    """
    def __init__(self, peft_model, peft_config, adapter_name: str = 'default') -> None:
        Module.__init__(self)
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        self._is_prompt_learning = False
        self.peft_type = peft_config.peft_type
        self.active_adapter = adapter_name

        self.base_model = peft_model
        self.peft_config = {adapter_name: peft_config}

        self.set_additional_trainable_modules(self.peft_config[adapter_name], self.active_adapter)

    def get_base_model(self) -> Module:
        # override is needed otherwise this would skip our cool DynaLora stack
        return self.base_model

    def save_pretrained(self,
                        save_directory: str,
                        safe_serialization: bool = True,
                        selected_adapters: list[str] | None = None,
                        save_embedding_layers: str | bool = "auto",
                        is_main_process: bool = True,
                        **kwargs: Any) -> None:
        super().save_pretrained(save_directory, safe_serialization, selected_adapters, save_embedding_layers, is_main_process, **kwargs)
        dynalora_model = self.base_model
        if not isinstance(dynalora_model, DynaBaseMixin):
            return
        output_file = os.path.join(save_directory, f"{self.adapter_name}.pt")
        save_dict = {
            'mask': dynalora_model.mask,
            'schedule': getattr(dynalora_model.schedule, 'get_state', lambda: {})(),
            'allocator': getattr(dynalora_model.allocator, 'get_state', lambda: {})(),
            'adapter_modules': {
                adapter_name: adapter_module.get_state()
                for adapter_name, adapter_module in dynalora_model.named_adapter_modules.items()
            }
        }
        torch.save(save_dict, output_file)

    def load_adapter(self,
                     model_id: str,
                     adapter_name: str,
                     is_trainable: bool = False,
                     torch_device: str | None = None,
                     **kwargs: Any):
        super().load_adapter(model_id, adapter_name, is_trainable, torch_device, **kwargs)
        dynalora_model = self.base_model
        if not isinstance(dynalora_model, DynaBaseMixin):
            return
        path = os.path.join(model_id, f"{adapter_name}.pt")
        states = torch.load(path)
        dynalora_model.set_mask(states['mask'])
        getattr(dynalora_model.allocator,'set_state', lambda: None)(states['allocator'])
        getattr(dynalora_model.schedule,'set_state', lambda: None)(states['schedule'])
        for adapter_name, adapter in dynalora_model.named_adapter_modules.items():
            if adapter_name in states['adapter_modules']:
                adapter.set_state(states['adapter_modules'][adapter_name])
