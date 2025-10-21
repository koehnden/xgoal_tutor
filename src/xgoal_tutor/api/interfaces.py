"""Interfaces for plugging xGoal models into the API service."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from xgoal_tutor.api.schemas import ShotInput


class XGoalModel(ABC):
    """Interface for the predictive xGoal model."""

    @abstractmethod
    def predict(self, shots: Sequence["ShotInput"]) -> Sequence[float]:
        """Return xG values for the provided shots."""

    def reason_codes(self, shots: Sequence["ShotInput"]) -> Sequence[Iterable[str]]:
        """Optional detailed codes describing the prediction.

        Implementations can override this method to provide model-specific
        information. The default implementation returns empty reason codes for
        every shot.
        """

        return tuple([] for _ in shots)


class LanguageModel(ABC):
    """Interface for the natural-language explanation model."""

    @abstractmethod
    def generate_explanation(
        self,
        shot: "ShotInput",
        xg: float,
        reason_codes: Iterable[str],
    ) -> str:
        """Produce a natural-language explanation for the shot."""

