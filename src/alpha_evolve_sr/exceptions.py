"""Custom exception hierarchy for alpha_evolve_sr."""


class AlphaEvolveSRError(Exception):
    """Base exception for all alpha_evolve_sr errors."""


class SpecificationError(AlphaEvolveSRError):
    """Raised when a specification file is invalid or cannot be parsed."""


class SandboxTimeoutError(AlphaEvolveSRError):
    """Raised when sandbox code execution exceeds the timeout."""


class LLMProviderError(AlphaEvolveSRError):
    """Raised when an LLM provider request fails after retries."""


class CheckpointError(AlphaEvolveSRError):
    """Raised when checkpoint save/load operations fail."""


class ComplexityError(AlphaEvolveSRError):
    """Raised when complexity scoring fails."""
