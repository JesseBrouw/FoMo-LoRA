from typing import List, Dict, Any
from abc import ABC, abstractmethod
import torch
import json

class BaseAllocator(ABC):
    """
        Base class for all allocator classes.

        Allocators have a call function, which given a list of numbers,
        returns a (0,1) mask of the same length, where 1s indicate the
        selected elements.
    """
    def __init__(self, k: int = 0) -> None:
        self.k = k
        self.named_adapter_modules = None # to be set by the model
        self.output_path = None # for logging
        self.mask = None # for loading/saving

    # placeholders for now.
    def get_state(self):
        return {'mask': self.mask}
    def set_state(self, state):
        self.mask = state['mask']
        self._apply_mask(self.mask)

    def set_adapter_modules(self, adapter_modules: Dict[str, Any]):
        self.named_adapter_modules = adapter_modules
    def set_output_path(self, output_path: str):
        self.output_path = output_path

    def __call__(self) -> List[float]:
        """
            Reallocate (activate/deactivate) the adapter modules based on their state.
        """
        self.mask = self._compute_mask()
        self._apply_mask(self.mask)

    def _apply_mask(self, mask: torch.Tensor) -> None:
        for mod, msk in zip(self.named_adapter_modules.values(), mask):
            if msk:
                mod.activate()
            else:
                mod.deactivate()

    def _compute_mask(self) -> torch.Tensor:
        """
            Based on the adapter modules, compute the mask.

            If self.ouptut_path is set, log the mask and the activations.
        """
        raise NotImplementedError("This method should be implemented by the derived class.")

    def _make_json_log(self, acts, mask) -> None:
        if self.output_path is None:
            return
        # log to json
        with open(self.output_path, "r") as f:
            data = json.load(f)
            data["cum_acts"].append(acts)
            data["masks"].append(mask)
        with open(self.output_path, "w") as f:
            json.dump(data, f)

class TopKAllocator(BaseAllocator):
    """
        Allocator that selects the k largest elements.
    """
    def __init__(self, k: int = 0) -> None:
        super().__init__(k)

    def _compute_mask(self) -> torch.Tensor:
        if not hasattr(self, "adapter_modules") or self.named_adapter_modules is None:
            raise ValueError("Adapter modules have not been set.")
        values = [mod.cum_acts for mod in self.named_adapter_modules.values()]
        values = torch.tensor(values)
        _, idx = torch.topk(values, self.k)
        mask = torch.zeros_like(values)
        mask[idx] = 1
        # log
        self._make_json_log(values.tolist(), mask.tolist())
        return mask

class ThresholdAllocator(BaseAllocator):
    """
        Allocator that selects the top elements, until the sum of their values
        exceeds a threshold.

        Note that the threshold should be a value between 0 and 1,
        and the values will be automatially normalized.
    """
    def __init__(self, threshold: float) -> None:
        super().__init__() # k here is not used
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be in the range [0,1].")
        self.threshold = threshold

    def _compute_mask(self) -> torch.Tensor:
        if not hasattr(self, "adapter_modules") or self.named_adapter_modules is None:
            raise ValueError("Adapter modules have not been set.")
        values = [mod.cum_acts for mod in self.named_adapter_modules.values()]
        values, indices = torch.sort(torch.tensor(values), descending=True)
        # make sure they are normalized
        values = values / values.sum()
        # compute cumulative sum
        csum = values.cumsum(dim=0)
        mask = torch.zeros_like(values)
        mask[indices[csum < self.threshold]] = 1
        # log
        self._make_json_log(values.tolist(), mask.tolist())
        return mask

class MultinomialAllocator(BaseAllocator):
    """
        Allocator that selects the elements according to
        the multinomial distribution whose parameters are given by the values.
    """
    def __init__(self, k: int = 0) -> None:
        super().__init__(k) # here k is the number of elements to sample
    
    def _compute_mask(self) -> torch.Tensor:
        if not hasattr(self, "adapter_modules") or self.named_adapter_modules is None:
            raise ValueError("Adapter modules have not been set.")
        values = [mod.cum_acts for mod in self.named_adapter_modules.values()]
        values = torch.tensor(values, dtype=torch.float)
        mask = torch.zeros_like(values)
        mask[torch.multinomial(values, self.k)] = 1
        # log
        self._make_json_log(values.tolist(), mask.tolist())
        return mask

class ScaledMultinomialAllocator(BaseAllocator):
    """
        Allocator that selects modules based on their
        scaled cumulative activations, using a multinomial distribution.

        For a given module i, the weighted cumulative activations are computed as:
        w_i = exp(mod_i.cum_acts)/sum(exp(mod_j.cum_acts) for j in adapter_modules) + gamma * 1/mod_i.counter
    """
    def __init__(self, k: int) -> None:
        super().__init__(k) # here k is the number of elements to sample

    def _compute_mask(self) -> torch.Tensor:
        if not hasattr(self, "named_adapter_modules") or self.named_adapter_modules is None:
            raise ValueError("Adapter modules have not been set.")
        acts = torch.tensor(
            [mod.cum_acts for mod in self.named_adapter_modules.values()],
            requires_grad=False)
        counter = torch.tensor(
            [mod.counter for mod in self.named_adapter_modules.values()],
            requires_grad=False)
        weights = acts / acts.sum() * 1/(counter+1e-2)

        mask = torch.zeros_like(weights)
        mask[torch.multinomial(weights, self.k, replacement=False)] = 1
        # log (sorry if I make stuff break but its good to see how the scaling works)
        self._make_json_log({"acts": acts.tolist(), "weights": weights.tolist()}, mask.tolist())
        return mask
