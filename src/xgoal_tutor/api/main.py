"""FastAPI application exposing the xGoal inference endpoints."""

from __future__ import annotations

import csv
import io
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from pydantic import ValidationError

from xgoal_tutor.api.interfaces import LanguageModel, XGoalModel
from xgoal_tutor.api.schemas import (
    CSVImportError,
    PredictShotsRequest,
    PredictShotsResponse,
    ShotCSVRow,
    ShotInput,
)
from xgoal_tutor.api.service import (
    MatchNotFoundError,
    ModelNotConfiguredError,
    XGoalService,
)


def create_app(
    xgoal_model: Optional[XGoalModel] = None,
    language_model: Optional[LanguageModel] = None,
) -> FastAPI:
    """Instantiate the FastAPI application with optional model implementations."""

    app = FastAPI(title="xGoal Inference Service", version="0.1.0")
    app.state.xgoal_service = XGoalService(xgoal_model, language_model)

    router = APIRouter()

    def get_service(request: Request) -> XGoalService:
        return request.app.state.xgoal_service

    @router.post("/predict_shots", response_model=PredictShotsResponse)
    def predict_shots(
        payload: PredictShotsRequest,
        service: XGoalService = Depends(get_service),
    ) -> PredictShotsResponse:
        try:
            predictions = service.predict_shots(payload.shots, match_id=payload.match_id)
        except ModelNotConfiguredError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        response_predictions = [prediction.to_response() for prediction in predictions]
        return PredictShotsResponse(match_id=payload.match_id, predictions=response_predictions)

    @router.get("/match/{match_id}/shots", response_model=PredictShotsResponse)
    def get_match_predictions(
        match_id: str,
        service: XGoalService = Depends(get_service),
    ) -> PredictShotsResponse:
        try:
            predictions = service.get_cached_match(match_id)
        except MatchNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        response_predictions = [prediction.to_response() for prediction in predictions]
        return PredictShotsResponse(match_id=match_id, predictions=response_predictions)

    @router.post("/import_csv", response_model=PredictShotsResponse)
    async def import_csv(
        file: UploadFile = File(...),
        match_id: Optional[str] = Query(
            default=None,
            description="Optional match identifier used to cache the imported predictions.",
        ),
        service: XGoalService = Depends(get_service),
    ) -> PredictShotsResponse:
        try:
            content = await file.read()
            text_stream = io.StringIO(content.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to decode the uploaded file as UTF-8.",
            ) from exc

        reader = csv.DictReader(text_stream)
        if not reader.fieldnames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV file must include a header row.",
            )

        shots: List[ShotInput] = []
        for row_index, row in enumerate(reader, start=2):  # start=2 accounts for header row
            try:
                validated_row = ShotCSVRow.model_validate(row)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=CSVImportError(row=row_index, errors=exc.errors()).model_dump(),
                ) from exc
            shots.append(ShotInput.model_validate(validated_row.model_dump()))

        if not shots:
            return PredictShotsResponse(match_id=match_id, predictions=[])

        try:
            predictions = service.predict_shots(shots, match_id=match_id)
        except ModelNotConfiguredError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        response_predictions = [prediction.to_response() for prediction in predictions]
        return PredictShotsResponse(match_id=match_id, predictions=response_predictions)

    app.include_router(router)
    return app


app = create_app()

__all__ = ["app", "create_app"]
