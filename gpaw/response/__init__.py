"""GPAW Response core functionality."""
from __future__ import annotations
from .groundstate import (ResponseGroundStateAdapter,
                          ResponseGroundStateAdaptable)  # noqa
from .context import (ResponseContext,
                      ResponseContextInput, TXTFilename, timer)  # noqa

__all__ = ['ResponseGroundStateAdapter', 'ResponseGroundStateAdaptable',
           'ResponseContext', 'ResponseContextInput', 'TXTFilename', 'timer']


def ensure_gs_and_context(gs: ResponseGroundStateAdaptable,
                          context: ResponseContextInput)\
        -> tuple[ResponseGroundStateAdapter, ResponseContext]:
    return (ResponseGroundStateAdapter.from_input(gs),
            ResponseContext.from_input(context))
