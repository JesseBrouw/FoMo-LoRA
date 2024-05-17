from typing import List
from abc import ABC, abstractmethod
import torch

class BaseAllocator(ABC):
    """
        Base class for all allocator classes.

        Allocators have a call function, which given a list of numbers,
        returns a (0,1) mask of the same length, where 1s indicate the
        selected elements.
    """
    def __init__(self, k: int = 0) -> None:
        self.k = k

    @abstractmethod
    def __call__(self, values: List[float]) -> List[float]:
        """
            Given a list of values, return a list of masks.
        """
        pass

class TopKAllocator(BaseAllocator):
    """
        Allocator that selects the k largest elements.
    """
    def __init__(self, k: int = 0) -> None:
        super().__init__(k)

    def __call__(self, values: List[float]) -> torch.Tensor:
        values = torch.tensor(values)
        _, idx = torch.topk(values, self.k)
        mask = torch.zeros_like(values)
        mask[idx] = 1
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

    def __call__(self, values: List[float]) -> torch.Tensor:
        values, indices = torch.sort(torch.tensor(values), descending=True)
        # make sure they are normalized
        values = values / values.sum()
        # compute cumulative sum
        csum = values.cumsum(dim=0)
        mask = torch.zeros_like(values)
        mask[indices[csum < self.threshold]] = 1

        return mask

class MultinomialAllocator(BaseAllocator):
    """
        Allocator that selects the elements according to
        the multinomial distribution whose parameters are given by the values.
    """
    def __init__(self, k: int = 0) -> None:
        super().__init__(k) # here k is the number of elements to sample
    
    def __call__(self, values: List[float]) -> torch.Tensor:
        values = torch.tensor(values, dtype=torch.float)
        mask = torch.zeros_like(values)
        mask[torch.multinomial(values, self.k)] = 1
        return mask