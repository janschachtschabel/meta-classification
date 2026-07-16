"""Exceptions whose messages are crafted for API clients.

Arbitrary exception text can leak paths or data fragments, so the training job
sanitizes unknown failures to a generic message. Errors of the types below are
the deliberate exception: their messages are written for the user (wrong column
name, too few rows, bad cv_folds, ...) and safe to surface on /train/status.
"""

from __future__ import annotations


class TrainingInputError(ValueError):
    """Invalid training input or configuration; message is safe to show."""
