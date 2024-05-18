from typing import Dict, Any, Union
import torch
import logging
from torch.optim import AdamW
from transformers.optimization import (
    LayerWiseDummyOptimizer,
    LayerWiseDummyScheduler,
    get_scheduler
)
import logging
logging.basicConfig(level=logging.DEBUG) 

SUPPORTED_OPTIMIZERS = {
    'adamw_torch': AdamW
}

def create_layerwise_optimizer_and_scheduler(
    model, config, num_training_steps, num_warmup_steps
):
    """
        Create layer-wise optimizer and scheduler based on the configuration.
        
        For each **trainalbe** layer, we create an optimizer and a scheduler,
        and register post_accumulate_gradient_hooks, which will handle the
        gradient updates on a per-layer basis.
        
        This method returns a dummy optimizer and a dummy scheduler, which
        should be passed to the transformers.Trainer's optimizers argument.
        
        NOTE: this method only handles the layer-wise case, in any other case
        please refer to transformers.Trainer's relevant arguments.
    """
    # get the optimizer class based on the configuration
    optim_cls = SUPPORTED_OPTIMIZERS[config.optim.lower()]
    optim_kwargs = {
        "lr": config.learning_rate,
        "eps": config.adam_epsilon,
        "betas": (config.adam_beta1, config.adam_beta2),
    }
    # iterate over the model's parameters and create an optimizer and a scheduler
    # for each trainable layer
    optimizer_dict: Dict[torch.Tensor, torch.optim.Optimizer] = {}
    scheduler_dict: Dict[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler] = {}
    if logging.getLogger().level == logging.DEBUG:
        param_name_map = {param: name for name, param in model.named_parameters()}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # create optimizer
        optimizer = optim_cls([dict(params=[param])], **optim_kwargs)
        optimizer_dict[param] = optimizer
        # create scheduler
        # NOTE: this is not entirely necessary, as the scheduler would be
        # initialised layer-wise if the optimizer is also layer-wise,
        # but it's more understandable what is going on 
        # if everything is done explicitly here
        scheduler = get_scheduler(
            config.lr_scheduler_type,
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=config.lr_scheduler_kwargs
        )
        scheduler_dict[param] = scheduler
        logging.debug(f"Created optimizer and scheduler for layer {name}")

    def optimizer_hook(param):
        if param.grad is not None:
            if logging.getLogger().level == logging.DEBUG:
                logging.debug(f"Updating layer {param_name_map[param]}")
            optimizer_dict[param].step()
            optimizer_dict[param].zero_grad()
            scheduler_dict[param].step()

    for param in model.parameters():
        if param.requires_grad:
            param.register_post_accumulate_grad_hook(optimizer_hook)

    # create a dummy optimizer and scheduler
    # we can choose to pass the dict of optimizers if the scheduler is not 
    # explicitly initialised, the transformers library can also initialise it for us.
    # the exact same behaviour is implemented in this function tho.
    # we do not use this option right now, but it's good to have it
    # optimizer = LayerWiseDummyOptimizer(optimizer_dict)
    optimizer = LayerWiseDummyOptimizer(optimizer_dict={})
    scheduler = LayerWiseDummyScheduler()

    return optimizer, scheduler