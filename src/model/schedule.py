# TODO: schedule classes to abstract the reallocation logic
from abc import ABC


class BaseSchedule(ABC):
    """
        Base class for reallocation schedule.
        
        subclasses should implement the _reallocate private function,
        and optionally also the step function.
    """
    def __init__(self, n: int = 0):
        self.n = n

    def step(self):
        """Increment the step counter."""
        self.n += 1

    def get_state(self):
        return {"n": self.n}
    def set_state(self, state):
        self.n = state["n"]

    def _reallocate(self) -> bool:
        """
            Boolean valued function to determine if the adapters should be reallocated.

            Any subclass should implement this method.
            
            Return:
            - bool: True if the adapters should be reallocated, False otherwise.
        """
        return False

    @property
    def reallocate(self) -> bool:
        """
            Boolean flag to determine if the adapters should be reallocated.
        """
        # always no
        return self._reallocate()

class OnceSchedule(BaseSchedule):
    """
        Schedule that reallocate the adapters only once after the specified step.
    """
    def __init__(self, after_step:int = 0, n: int = 0):
        super().__init__(n)
        self.after_step = after_step

    def _reallocate(self) -> bool:
        return self.n == self.after_step

    def get_state(self):
        state = super().get_state()
        state["after_step"] = self.after_step
        return state
    def set_state(self, state):
        super().set_state(state)
        self.after_step = state["after_step"]

class PeriodicSchedule(BaseSchedule):
    """
        Schedule that reallocate the adapters every n steps.
    """

    def __init__(self, period: int = 1, n: int = 0):
        super().__init__(n)
        self.period = period

    def _reallocate(self) -> bool:
        return self.n % self.period == 0
    
    def get_state(self):
        state = super().get_state()
        state["period"] = self.period
        return state
    def set_state(self, state):
        super().set_state(state)
        self.period = state["period"]