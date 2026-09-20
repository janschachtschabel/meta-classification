"""Pydantic request models for the API (the fixed-shape responses are in ``responses``,
the rest are plain dicts).

One module per reason to change: ``training`` for the levers of a run, ``serving`` for
what a prediction or a correction may ask, ``datasets`` for the work on datasets and
bundles, ``common`` for the wording and the validator they share. Importers keep saying
``from ..schemas import TrainRequest``.
"""

from .datasets import AnalyzeRequest, EvaluateRequest, ExportRequest, ValidateRequest
from .serving import ExplainRequest, FeedbackRequest, MultiPredictRequest, PredictRequest
from .training import ModelInfo, TrainRequest

__all__ = [
    "AnalyzeRequest",
    "EvaluateRequest",
    "ExplainRequest",
    "ExportRequest",
    "FeedbackRequest",
    "ModelInfo",
    "MultiPredictRequest",
    "PredictRequest",
    "TrainRequest",
    "ValidateRequest",
]
