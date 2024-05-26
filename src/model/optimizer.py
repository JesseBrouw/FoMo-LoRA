from typing import Dict, Any, Union
import os
import torch
import logging
from torch.optim import AdamW
from transformers.optimization import (
    LayerWiseDummyOptimizer,
    LayerWiseDummyScheduler,
    get_scheduler
)
from transformers import TrainingArguments
import logging
logger = logging.getLogger('lw-optim')

# supported optimizers
SUPPORTED_OPTIMIZERS = {
        'adamw_torch': AdamW
}

class LoadableLayerWiseDummyOptimizer(LayerWiseDummyOptimizer):
    """
        Dummy optimizer that can load the state from a dict.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 config: TrainingArguments,
                 logdir: str = '.') -> None:
        super().__init__([])
        # set up logger
        logger.handlers.clear()
        logger.addHandler(logging.FileHandler(os.path.join(logdir, 'optim.log')))

        self.model = model
        self.config = config
        self.optimizer_dict = {}
        self._make_optimizers()

    # def optimizer_hook(self, param):
    #     if param.grad is not None:
    #         self.optimizer_dict[param].step()
    #         self.optimizer_dict[param].zero_grad()

    def step(self, closure=None) -> float | None:
        # update params
        for name, param in self.model.named_parameters():
            logger.info(f'optim for %s, has grad: %s', name, param.grad is not None)
            if not param.requires_grad:
                continue
            if not param in self.optimizer_dict:
                logger.error('param %s not in optimizer_dict', name)
                continue
            self.optimizer_dict[param].step()
            
    def zero_grad(self) -> None:
        # zero grads
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if not param in self.optimizer_dict:
                logger.error('param %s not in optimizer_dict', name)
                continue
            self.optimizer_dict[param].zero_grad()

    def _make_optimizers(self):
        """
            Create a layer-wise optimizer based on the configuration.

            For each **trainalbe** layer, we create an optimizer,
            and register post_accumulate_gradient_hooks, which will handle the
            gradient updates on a per-layer basis.
        """
        # get the optimizer class based on the configuration
        optim_cls = SUPPORTED_OPTIMIZERS[self.config.optim.lower()]
        logger.info(f"Using optimizer {optim_cls} for layer-wise optimization")
        logger.info(f"Optimizer kwargs: lr={self.config.learning_rate}, eps={self.config.adam_epsilon}, betas=({self.config.adam_beta1}, {self.config.adam_beta2})")
        optim_kwargs = {
            "lr": self.config.learning_rate,
            "eps": self.config.adam_epsilon,
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
        }
        # iterate over the model's parameters and create an optimizer and a scheduler
        # for each trainable layer
        self.optimizer_dict: Dict[int, torch.optim.Optimizer] = {}
        self.name_to_param: Dict[str, torch.Tensor] = {}
        self.param_to_name: Dict[torch.Tensor, str] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # create optimizer
            optimizer = optim_cls([dict(params=[param])], **optim_kwargs)
            self.optimizer_dict[param] = optimizer
            self.name_to_param[name] = param
            self.param_to_name[param] = name
            logger.info(f"Created optimizer for layer {name}")

        # for param in self.model.parameters():
        #     logger.info('adding hook: %s %s', self.param_to_name.get(param, "unk"), param.requires_grad)
        #     if param.requires_grad:
        #         param.register_post_accumulate_grad_hook(self.optimizer_hook)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for param, optimizer in self.optimizer_dict.items():
            optimizer.load_state_dict(state_dict[self.param_to_name[param]])

    def state_dict(self):
        return {
            self.param_to_name[param]: optimizer.state_dict()
            for param, optimizer in self.optimizer_dict.items()
        }

class LoadableLayerWiseDummyScheduler(LayerWiseDummyScheduler):
    """
        Dummy scheduler that can load the state from a dict.
    """
    def __init__(self,
                 optimizer: LoadableLayerWiseDummyOptimizer,
                 config: TrainingArguments,
                 num_warmup_steps: int,
                 num_training_steps: int) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.config = config
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.scheduler_dict = {}
        self._make_schedulers()

    def scheduler_hook(self, param):
        if param.grad is not None:
            self.scheduler_dict[param].step()

    def _make_schedulers(self):
        """
            Create a layer-wise scheduler based on the configuration.

            For each **trainalbe** layer, we create a scheduler,
            and register post_accumulate_gradient_hooks, which will handle the
            gradient updates on a per-layer basis.
        """
        # iterate over the model's parameters and create an optimizer and a scheduler
        # for each trainable layer
        self.scheduler_dict: Dict[torch.Tensor, torch.optim.lr_scheduler.LRScheduler] = {}

        for param, optimizer in self.optimizer.optimizer_dict.items():
            # create scheduler
            scheduler = get_scheduler(
                self.config.lr_scheduler_type,
                optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
                scheduler_specific_kwargs=self.config.lr_scheduler_kwargs
            )
            self.scheduler_dict[param] = scheduler

        for _, param in self.optimizer.name_to_param.items():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.scheduler_hook)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        param_to_name = self.optimizer.param_to_name
        for param, scheduler in self.scheduler_dict.items():
            scheduler.load_state_dict(state_dict[param_to_name[param]])

    def state_dict(self):
        param_to_name = self.optimizer.param_to_name
        return {
            param_to_name[param]: scheduler.state_dict()
            for param, scheduler in self.scheduler_dict.items()
        }

def create_layerwise_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: TrainingArguments,
    num_warmup_steps: int,
    num_training_steps: int,
    logdir: str = '.'
):
    """
        Create a layer-wise optimizer and scheduler based on the configuration.

        For each **trainalbe** layer, we create an optimizer and a scheduler
        for each trainable layer.
    """
    optimizer = LoadableLayerWiseDummyOptimizer(model, config, logdir=logdir)
    scheduler = LoadableLayerWiseDummyScheduler(optimizer, config, num_warmup_steps, num_training_steps)
    return optimizer, scheduler
