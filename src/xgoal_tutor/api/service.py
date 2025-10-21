"""Core business logic for the xGoal inference API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from xgoal_tutor.api.interfaces import LanguageModel, XGoalModel
from xgoal_tutor.api.schemas import ShotInput, ShotPredictionResponse


class ModelNotConfiguredError(RuntimeError):
    """Raised when a request is made before models are configured."""


class MatchNotFoundError(KeyError):
    """Raised when cached match predictions are not available."""

    def __init__(self, match_id: str) -> None:  # pragma: no cover - trivial
        super().__init__(f"No cached predictions found for match '{match_id}'.")
        self.match_id = match_id


@dataclass(slots=True)
class ShotPrediction:
    """Internal representation of a prediction."""

    shot: ShotInput
    xg: float
    reason_codes: List[str]
    explanation: str

    def to_response(self) -> ShotPredictionResponse:
        return ShotPredictionResponse(
            x=self.shot.x,
            y=self.shot.y,
            flags=self.shot.flags,
            xg=self.xg,
            reason_codes=list(self.reason_codes),
            explanation=self.explanation,
        )


class XGoalService:
    """Facade for interacting with the xGoal and language models."""

    def __init__(
        self,
        xgoal_model: Optional[XGoalModel] = None,
        language_model: Optional[LanguageModel] = None,
    ) -> None:
        self._xgoal_model = xgoal_model
        self._language_model = language_model
        self._match_cache: Dict[str, List[ShotPrediction]] = {}

    def set_xgoal_model(self, model: XGoalModel) -> None:
        """Attach the predictive model at runtime."""

        self._xgoal_model = model

    def set_language_model(self, model: LanguageModel) -> None:
        """Attach the language model at runtime."""

        self._language_model = model

    def predict_shots(
        self,
        shots: Sequence[ShotInput],
        match_id: Optional[str] = None,
    ) -> List[ShotPrediction]:
        """Generate predictions for the provided shots."""

        if not shots:
            return []

        if self._xgoal_model is None:
            raise ModelNotConfiguredError("xGoal model is not configured.")

        xg_values = list(self._xgoal_model.predict(shots))
        if len(xg_values) != len(shots):
            raise ValueError("xGoal model returned an unexpected number of predictions.")

        reason_codes_sequences = list(self._xgoal_model.reason_codes(shots))
        if len(reason_codes_sequences) != len(shots):
            raise ValueError("xGoal model reason codes must align with the provided shots.")

        predictions: List[ShotPrediction] = []
        for shot, xg_value, codes in zip(shots, xg_values, reason_codes_sequences):
            codes_list = list(codes)
            explanation = ""
            if self._language_model is not None:
                explanation = self._language_model.generate_explanation(shot, float(xg_value), codes_list)
            predictions.append(
                ShotPrediction(
                    shot=shot,
                    xg=float(xg_value),
                    reason_codes=codes_list,
                    explanation=explanation,
                )
            )

        if match_id:
            self._match_cache[match_id] = predictions

        return predictions

    def get_cached_match(self, match_id: str) -> List[ShotPrediction]:
        """Retrieve cached predictions for a match."""

        if match_id not in self._match_cache:
            raise MatchNotFoundError(match_id)
        return self._match_cache[match_id]

    def clear_cache(self) -> None:
        """Remove all cached match predictions."""

        self._match_cache.clear()


__all__ = [
    "MatchNotFoundError",
    "ModelNotConfiguredError",
    "ShotPrediction",
    "XGoalService",
]
