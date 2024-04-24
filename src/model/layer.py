import torch
import warnings
from typing import Optional
from peft import LoraConfig
from peft import LoraLayer, PeftConfig, BaseTunerLayer
from peft.tuners.lora.layer import (LoraLayer,
                                    Linear as LoraLinear,
                                    Conv2d as LoraConv2d,
                                    Embedding as LoraEmbedding)


class DynaLoraLayer():
    """
        Dynamic LoRA layer. 
        
        Does almost the same as LoraLayer, but keeps track of the cumulative forward activations
        of the layer. This can be used to dynamically reallocate the adapters.
    """
    def __init__(self, aggregate_type: str = 'l2', **kwargs):
        if aggregate_type == 'l2':
            self.aggregator = lambda x: torch.norm(x, p=2)
        self.reset_cum_acts()

    def reset_cum_acts(self):
        self.cum_acts = torch.nn.Parameter(
            torch.tensor(0.0), requires_grad=False)

    def get_cum_acts(self):
        return self.cum_acts
    
class Linear(LoraLinear, DynaLoraLayer):
    """
        Overrides lora.Linear with the cumulative activations tracking.
    """
    def __init__(self, **kwargs):
        LoraLinear.__init__(self, **kwargs)
        DynaLoraLayer.__init__(self, **kwargs)

    def forward(self, x: torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> torch.Tensor:
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
                    self.cum_acts = self.cum_acts + self.aggregator(ab_path.detach())
                    result = result + ab_path
                else:
                    x = dropout(x)
                    result = result + \
                        self._apply_dora(x, lora_A, lora_B,
                                         scaling, active_adapter)

            result = result.to(torch_result_dtype)

        return result


def dispatch_default(
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
        new_module = Linear(target, adapter_name, **kwargs)
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

