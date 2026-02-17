"""Custom exception hierarchy for alpha_evolve_sr."""


class AlphaEvolveSRError(Exception):
    """Base exception for all alpha_evolve_sr errors."""


class LLMProviderError(AlphaEvolveSRError):
    """Raised when an LLM provider request fails after retries."""


class CheckpointError(AlphaEvolveSRError):
    """Raised when checkpoint save/load operations fail."""
