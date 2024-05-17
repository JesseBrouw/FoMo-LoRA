import functools
import torch
import warnings
from typing import Optional
from peft import LoraConfig
from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora.layer import (LoraLayer,
                                    Linear as LoraLinear,
                                    Conv2d as LoraConv2d,
                                    Embedding as LoraEmbedding)


class GradientHook:
    def __init__(self, module):
        self.grad_output = None
        
        # NOTE: This function is deprecated in favor of register_full_backward_hook
        self.hook = module.register_hook(lambda tensor: self.backward_hook(tensor))

    def backward_hook(self, grad_output):
        """Should store the gradient output (no idea yet if it should be mean or sth else)."""
        self.grad_output = grad_output[0].mean()  # Assuming single output

    def remove(self):
        self.hook.remove()

class DinaLoraLayer(LoraLayer):
    """Dynamic LoRA layer that relies on gradient magnitude to reallocate adapters."""
    def __init__(self, *args, **kwargs):

        peft_config = kwargs.get("peft_config", None)
        if peft_config is None:
            raise ValueError("peft_config is required.")
        # counter to keep track of the number of forward passes.
        # Used by the allocator and we can also log it.
        self._counter = torch.tensor(0, requires_grad=False)
        self._is_active = False
        self.aggregator = peft_config.aggregator
        self.reset_cum_acts()

    def activate(self):
        """
            Enable gradients for the layer. (TODO: THIS IS NOT THE REAL IDEA, I WANT TO EXAMINE THE GRADIENTS OF THE BASE LAYER NOT THE ADAPTER, BUT GOTTA FIGURE OUT HOW TO TRACK THEM JUST FOR THE FIRST FEW ITERATIONS, i.e. full FT in the start, then switch to DINA/DYNA)
        """
        self._is_active = True
        for name, param in self.named_parameters():
            if name.split('.')[0] in LoraLayer.adapter_layer_names:
                param.requires_grad = True
                param.grad_hook = GradientHook(param)

    def deactivate(self):
        """
            Disable gradients for the layer.
        """
        self._is_active = False
        for name, param in self.named_parameters():
            if name.split('.')[0] in LoraLayer.adapter_layer_names:
                param.requires_grad = False
                if hasattr(param, "grad_hook"):
                    param.grad_hook.remove()
    def reset_cum_acts(self):
        self._cum_acts = torch.tensor(0.0, requires_grad=False)
    @property
    def cum_acts(self):
        return self._cum_acts
    @property
    def counter(self):
        return self._counter 


class DinaLinear(LoraLinear, DinaLoraLayer):
    """
        Overrides lora.Linear with the cumulative activations tracking.
    """
    def __init__(self, *args, **kwargs):
        LoraLinear.__init__(self, *args, **kwargs)
        DinaLoraLayer.__init__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> torch.Tensor:
        if self._is_active:
            # increment the counter only if
            # the layer is active
            self._counter += 1
        # copy-paste from LoraLinear
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # NOTE: this is the only place we do something different
                    ab_path = lora_B(lora_A(dropout(x))) * scaling
                    result = result + ab_path

                    grad_hook = getattr(lora_A.weight, "grad_hook", None)
                    if grad_hook is not None and grad_hook.grad_output is not None:
                        self._cum_acts = self._cum_acts + self.aggregator(grad_hook.grad_output.detach())
                else:
                    x = dropout(x)
                    result = result + \
                        self._apply_dora(x, lora_A, lora_B,
                                         scaling, active_adapter)

            result = result.to(torch_result_dtype)

        return result


def dispatch_dynamic_dina(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # if isinstance(target_base_layer, torch.nn.Embedding):
    #     embedding_kwargs = kwargs.copy()
    #     embedding_kwargs.pop("fan_in_fan_out", None)
    #     embedding_kwargs.update(lora_config.loftq_config)
    #     new_module = Embedding(target, adapter_name, **embedding_kwargs)
    # elif isinstance(target_base_layer, torch.nn.Conv2d):
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Conv2d(target, adapter_name, **kwargs)
    # elif isinstance(target_base_layer, torch.nn.Linear):
    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = DinaLinear(target, adapter_name, peft_config=lora_config, **kwargs)
    # elif isinstance(target_base_layer, Conv1D):
    #     if not kwargs["fan_in_fan_out"]:
    #         warnings.warn(
    #             "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
    #         )
    #         kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Linear(target, adapter_name,
    #                         is_target_conv_1d_layer=True, **kwargs)

    return new_module

class DynaLoraLayer(LoraLayer):
    """
        Dynamic LoRA layer. 

        Does almost the same as LoraLayer, but keeps track of the cumulative forward activations
        of the layer. This can be used to dynamically reallocate the adapters.
    """
    def __init__(self, *args, **kwargs):
        # loraLayer is typically initialized through another inheritance path
        # super().__init__(*args, **kwargs)
        
        # this is needed to put cum_acts on the right device because the addition won't work in forward if they are on different devices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        peft_config = kwargs.get("peft_config", None)
        if peft_config is None:
            raise ValueError("peft_config is required.")

        # counter to keep track of the number of forward passes.
        # Used by the allocator and we can also log it.
        self._counter = torch.tensor(0, requires_grad=False)
        self._is_active = False
        self.aggregator = peft_config.aggregator
        self.reset_cum_acts()

    def reset_cum_acts(self):
        # initializing to small epsilon otherwise multinomial allocator will fail because values can't be all 0
        self._cum_acts = torch.tensor(1e-6, requires_grad=False).to(self.device)

    def activate(self):
        """
            Enable gradients for the layer.
        """
        self._is_active = True
        for name, param in self.named_parameters():
            if name.split('.')[0] in LoraLayer.adapter_layer_names:
                param.requires_grad = True

    def deactivate(self):
        """
            Disable gradients for the layer.
        """
        self._is_active = False
        for name, param in self.named_parameters():
            if name.split('.')[0] in LoraLayer.adapter_layer_names:
                param.requires_grad = False

    @property
    def cum_acts(self):
        return self._cum_acts
    @property
    def counter(self):
        return self._counter

class Linear(LoraLinear, DynaLoraLayer):
    """
        Overrides lora.Linear with the cumulative activations tracking.
    """
    def __init__(self, *args, **kwargs):
        LoraLinear.__init__(self, *args, **kwargs)
        DynaLoraLayer.__init__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> torch.Tensor:
        if self._is_active:
            # increment the counter only if
            # the layer is active
            self._counter += 1
        result = super().forward(x, *args, **kwargs)
        #print("result: ", result)  
        aggregated = self.aggregator(result.detach())
        #print("Aggregated: ", aggregated)
        self._cum_acts = self._cum_acts + aggregated
        return result

def dispatch_dynamic(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # if isinstance(target_base_layer, torch.nn.Embedding):
    #     embedding_kwargs = kwargs.copy()
    #     embedding_kwargs.pop("fan_in_fan_out", None)
    #     embedding_kwargs.update(lora_config.loftq_config)
    #     new_module = Embedding(target, adapter_name, **embedding_kwargs)
    # elif isinstance(target_base_layer, torch.nn.Conv2d):
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Conv2d(target, adapter_name, **kwargs)
    # elif isinstance(target_base_layer, torch.nn.Linear):
    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, peft_config=lora_config, **kwargs)
    # elif isinstance(target_base_layer, Conv1D):
    #     if not kwargs["fan_in_fan_out"]:
    #         warnings.warn(
    #             "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
    #         )
    #         kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
    #     kwargs.update(lora_config.loftq_config)
    #     new_module = Linear(target, adapter_name,
    #                         is_target_conv_1d_layer=True, **kwargs)

    return new_module


# TODO: all the other layer types

