"""Exceptions whose messages are crafted for API clients.

Arbitrary exception text can leak paths or data fragments, so the training job
sanitizes unknown failures to a generic message. Errors of the types below are
the deliberate exception: their messages are written for the user (wrong column
name, too few rows, bad cv_folds, ...) and safe to surface on /train/status.
"""

from __future__ import annotations


class UserFacingError(Exception):
    """A failure whose message was written for the operator: safe to show as it is.

    The job runner surfaces these verbatim and sanitizes everything else.
    """


class TrainingInputError(UserFacingError, ValueError):
    """Invalid training input or configuration; message is safe to show."""


class TrainingProcessError(UserFacingError, RuntimeError):
    """The training's child process ended without a result — killed (most often by the
    out-of-memory killer) or crashed. The message says so and what to do about it."""
