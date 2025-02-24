from abc import ABC, abstractmethod
from typing import Callable, List

from datasets import Dataset


class BaseDataset(ABC):
    @property
    @abstractmethod
    def train_dataset(self) -> Dataset:
        """The training dataset."""
        pass

    @property
    @abstractmethod
    def eval_dataset(self) -> Dataset:
        """The evaluation dataset."""
        pass

    @property
    @abstractmethod
    def reward_funcs(self) -> List[Callable]:
        """The reward functions."""
        pass
