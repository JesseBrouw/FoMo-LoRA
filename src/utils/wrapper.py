from peft.peft_model import PeftModel
from torch.nn import Module
import os
import torch

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

        self._is_prompt_learning = False
        self.peft_type = peft_config.peft_type
        self.active_adapter = adapter_name

        self.base_model = peft_model
        # very important to set modules_to_save attribute
        # it is set for peft_config which in real is lora_config in train.py for sequence classification (Classification head)
        # but self.set_additional_trainable_modules() won't set it unless it is set within the self.peft_config not in the self.peft_config[adapter_name]
        self.peft_config = peft_config #{adapter_name: peft_config, "modules_to_save":peft_config.modules_to_save}
        # for compatibility.
        self.set_additional_trainable_modules(self.peft_config, self.active_adapter)

    def get_base_model(self) -> Module:
        # override is needed otherwise this would skip our cool DynaLora stack
        return self.base_model
    """
    def save_pretrained(self, save_directory, state_dict=None, safe_serialization=False):
    """
    #Save the model and the adapter weights to a directory so that it can be reloaded using `from_pretrained`.
    """
        # call PeftModel's save_pretrained
        super().save_pretrained(save_directory=save_directory, state_dict=state_dict, safe_serialization=safe_serialization)
        
        #os.makedirs(save_directory, exist_ok=True)

        # Save the base model
        self.base_model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization)

        # Save the adapter-specific weights
        #adapter_weights_file = os.path.join(save_directory, f"{self.active_adapter}_adapter_model.bin")
        #torch.save(self.state_dict(), adapter_weights_file)

        # Save the configuration
        #self.peft_config[self.active_adapter].save_pretrained(save_directory)
    """

    """
    @classmethod
    def from_pretrained(cls, model_class, pretrained_model_name_or_path, adapter_name, **kwargs):
    """
    #Load the model and the adapter weights from a directory.
    """
        base_model = model_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Load the adapter configuration
        peft_config = PeftModel.load_adapter_config(pretrained_model_name_or_path, adapter_name)

        # Initialize the wrapper with the base model and the adapter configuration
        model = cls(peft_model=base_model, peft_config=peft_config, adapter_name=adapter_name)

        # Load the adapter-specific weights
        adapter_weights_file = os.path.join(pretrained_model_name_or_path, f"{adapter_name}_adapter_model.bin")
        state_dict = torch.load(adapter_weights_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        return model
    """