import functools
import torch
import warnings
from typing import Optional
from peft import LoraConfig, VeraConfig
from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora.layer import (LoraLayer,
                                    Linear as LoraLinear,
                                    Conv2d as LoraConv2d,
                                    Embedding as LoraEmbedding)
from peft.tuners.vera.layer import (VeraLayer,
                                    Linear as VeraLinear)

class DinaLoraLayer(LoraLayer):
    """Dynamic LoRA layer that relies on gradient magnitude to reallocate adapters."""
    def __init__(self, *args, **kwargs):

        peft_config = kwargs.get("peft_config", None)
        if peft_config is None:
            raise ValueError("peft_config is required.")
        # counter to keep track of the number of forward passes.
        # Used by the allocator and we can also log it.
        self._counter = 0
        self._is_active = True
        self.aggregator = peft_config.aggregator
        self.reset_cum_acts()
        self._register_hooks()

    def backward_hook(self, param: torch.nn.Parameter):
        if param.grad is not None:
            assert self._is_active, "The layer is not active, but a parameter is being updated."
            assert param not in self.param_acts, "A parameter is being updated twice."
            assert param in self.hooked_params, "A parameter is being updated, but it is not hooked."
            self.param_acts[param] = self.aggregator(param.grad_output.detach())
            self.hooked_params.remove(param)
            if len(self.hooked_params) == 0:
                self.cum_acts += sum([act for act in self.param_acts.values()])
                self.hooked_params = set(self.param_acts.keys())
                self.param_acts.clear()

    def _register_hooks(self):
        self.param_acts = {}
        self.hooked_params = set()
        for name, param in self.named_parameters():
            if name.split('.')[0] in LoraLayer.adapter_layer_names:
                param.register_post_accumulate_grad_hook(self.backward_hook)
                self.hooked_params.add(param)

    def activate(self):
        """
            Enable gradients for the layer. (TODO: THIS IS NOT THE REAL IDEA, I WANT TO EXAMINE THE GRADIENTS OF THE BASE LAYER NOT THE ADAPTER, BUT GOTTA FIGURE OUT HOW TO TRACK THEM JUST FOR THE FIRST FEW ITERATIONS, i.e. full FT in the start, then switch to DINA/DYNA)
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

    def reset_cum_acts(self):
        self._cum_acts = torch.tensor(1e-6, requires_grad=False)
    @property
    def cum_acts(self):
        return self._cum_acts
    @property
    def counter(self):
        return self._counter 

    def get_state(self):
        return {"cum_acts": self._cum_acts, 'is_active': self._is_active}
    def set_state(self, state):
        self._cum_acts = state["cum_acts"]
        self._is_active = state["is_active"]

class DinaLoraLinear(LoraLinear, DinaLoraLayer):
    """
        Overrides lora.Linear with the cumulative activations tracking.
    """
    def __init__(self, *args, **kwargs):
        LoraLinear.__init__(self, *args, **kwargs)
        DinaLoraLayer.__init__(self, *args, **kwargs)

    # NOTE: we don't have to do anythin with the forward pass,
    # the hooks are already registered, we just need to wait for the gradients to come in.

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
        new_module = DinaLoraLinear(target, adapter_name, peft_config=lora_config, **kwargs)
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


class DynaLayerMixin:
    """
        Dynamic LoRA layer. 

        Does almost the same as LoraLayer, but keeps track of the cumulative forward activations
        of the layer. This can be used to dynamically reallocate the adapters.
    """
    def __init__(self, *args, **kwargs):
        # this is needed to put cum_acts on the right device because the addition won't work in forward if they are on different devices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        peft_config = kwargs.get("peft_config", None)
        if peft_config is None:
            raise ValueError("peft_config is required.")

        # counter to keep track of the number of forward passes.
        # Used by the allocator and we can also log it.
        self._counter = 0

        self._is_active = True
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

    def get_state(self):
        return {"cum_acts": self._cum_acts, 'is_active': self._is_active}
    def set_state(self, state):
        self._cum_acts = state["cum_acts"]
        self._is_active = state["is_active"]

class DynaLoraLinear(LoraLinear, DynaLayerMixin):
    """
        Overrides lora.Linear with the cumulative activations tracking.
    """
    def __init__(self, *args, **kwargs):
        LoraLinear.__init__(self, *args, **kwargs)
        DynaLayerMixin.__init__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> torch.Tensor:
        if self._is_active:
            # increment the counter only if the layer is active
            self._counter += 1
        result = super().forward(x, *args, **kwargs)
        #print("result: ", result)  
        aggregated = self.aggregator(result.detach())
        #print("Aggregated: ", aggregated)
        self._cum_acts = self._cum_acts + aggregated.to(self._cum_acts.device)
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
        new_module = DynaLoraLinear(target, adapter_name, peft_config=lora_config, **kwargs)
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

class DynaVeraLinear(VeraLinear, DynaLayerMixin):
    """
        Overrides vera.Linear with the cumulative activations tracking.
    """
    def __init__(self, *args, **kwargs):
        VeraLinear.__init__(self, *args, **kwargs)
        DynaLayerMixin.__init__(self, *args, **kwargs)

    def forward(self, x: torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> torch.Tensor:
        if self._is_active:
            # increment the counter only if the layer is active
            self._counter += 1
        result = super().forward(x, *args, **kwargs)
        aggregated = self.aggregator(result.detach())
        self._cum_acts = self._cum_acts + aggregated
        return result

def dispatch_dynamic_vera(
        target: torch.nn.Module,
        adapter_name: str,
        lora_config: VeraConfig, # TODO: rename to peft_config in all dispachers
        **kwargs,
    ) -> Optional[torch.nn.Module]:
    bias = kwargs.pop("bias", False)

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
    else:
        raise ValueError(
            f"Target module {target} is not supported. Currently, only the following modules are supported: "
            "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
        )
    vera_A = kwargs.pop("vera_A", None)
    vera_B = kwargs.pop("vera_B", None)
    if not vera_A or not vera_B:
        raise ValueError("vera_A and vera_B are required for VeraLinear.")
    new_module = DynaVeraLinear(
        target,
        vera_A,
        vera_B,
        adapter_name,
        bias=bias,
        d_initial=lora_config.d_initial,
        peft_config=lora_config,
        **kwargs,
    )

    return new_module
