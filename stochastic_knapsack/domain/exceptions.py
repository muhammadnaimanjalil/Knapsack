"""Application-specific exceptions suitable for concise user-facing errors."""


class InputDataError(ValueError):
    """Raised when input data or configuration violates a model requirement."""


class OptimizationError(RuntimeError):
    """Raised when an optimization algorithm cannot prove a valid solution."""
