from peft.peft_model import PeftModel
from torch.nn import Module

# FIXME: if we push to the PEFT library this wrapper will need to go away.
#        This is a temporary solution to allow for the dynamic LoRA model to be used
#        for example for printing the number of trainable params
class PeftModelWrapper(PeftModel, Module):
    """
        Wrapper class for the PeftModel.
        
        We use this class to wrap a DynaLoraModel instance
        and still be able to use all benefits of the PeftModel.
    """
    def __init__(self, peft_model, peft_config, adapter_name) -> None:
        Module.__init__(self)
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        self.modules_to_save = None
        self._is_prompt_learning = False
        self.peft_type = peft_config.peft_type
        self.active_adapter = adapter_name

        self.base_model = peft_model
        self.peft_config = {adapter_name: peft_config}
        # for compatibility.
        self.set_additional_trainable_modules(self.peft_config, self.active_adapter)

    def get_base_model(self) -> Module:
        # override is needed otherwise this would skip our cool DynaLora stack
        return self.base_model