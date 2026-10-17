"""Abstract base class for voice converters."""

from abc import ABC, abstractmethod
import numpy as np


class BaseConverter(ABC):
    """Interface every voice-conversion backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the converter."""
        ...

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Convert a chunk of mono float32 audio.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio samples in [-1, 1], at the converter's expected
            sample rate.

        Returns
        -------
        np.ndarray
            Converted audio, same sample rate, mono float32.
        """
        ...

    def cleanup(self) -> None:
        """Release any heavy resources (models, GPU memory)."""
        pass
