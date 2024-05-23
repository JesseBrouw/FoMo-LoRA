"""
    Define an observer that logs the forward and backward activations of the model.
    computes:
    - L1
    - L2,
    - Frobenius norm
    - Spectral norm
"""
import torch
import logging
logger = logging.getLogger('activation_observer')

class ActivationObserver:
    def __init__(self, model):
        self.model = model
        self.fwd_activations = {}
        self.bwd_activations = {}
        self.param2name = {}
        self._init_from_model()

    def _compute_norms(self, activations: torch.nn.Module):
        l1 = torch.norm(activations, p=1)
        l2 = torch.norm(activations, p=2)
        frobenius = torch.norm(activations, p='fro')
        return l1, l2, frobenius

    def fwd_hook(self, module: torch.nn.Module, input_, output):
        """over the module"""
        l1, l2, frobenius = self._compute_norms(output.detach())
        for param in module.parameters():
            self.fwd_activations[param].append({
                'l1': l1.item(),
                'l2': l2.item(),
                'frobenius': frobenius.item()
            })

    def bwd_hook(self, param):
        """over the individual parameters"""
        if param.grad is None:
            return
        l1, l2, frobenius = self._compute_norms(param.grad)
        self.bwd_activations[param].append({
            'l1': l1.item(),
            'l2': l2.item(),
            'frobenius': frobenius.item()
        })

    def _init_from_model(self):
        for name, module in self.model.named_modules():
            if not module.requires_grad_ or len(list(module.children())) > 0:
                continue
            has_param = False
            for nname, param in module.named_parameters():
                if not param.requires_grad or param in self.param2name:
                    continue
                has_param = True
                self.param2name[param] = name+'.'+nname
                self.fwd_activations[param] = []
                self.bwd_activations[param] = []
                param.register_post_accumulate_grad_hook(self.bwd_hook)
            if has_param:
                module.register_forward_hook(self.fwd_hook)

    def save_state(self, path):
        fwd_acts = {self.param2name[param]: acts
                    for param, acts in self.fwd_activations.items()}
        bwd_acts = {self.param2name[param]: acts
                    for param, acts in self.bwd_activations.items()}
        state_dict = {
            'fwd_activations': fwd_acts,
            'bwd_activations': bwd_acts,
        }
        torch.save(state_dict, path)