"""Shared test fixtures for alpha_evolve_sr."""

import pytest

from alpha_evolve_sr.code_manipulation import EvaluatedProgram, ParsedFunction, Program
from alpha_evolve_sr.config import ProgramsDatabaseConfig


# Keep Function as an alias for backward compatibility in tests.
Function = ParsedFunction


@pytest.fixture
def sample_function() -> ParsedFunction:
    """A simple ParsedFunction for testing."""
    return ParsedFunction(
        name="equation",
        args="rho, s, params",
        body="    return rho * params[0], 0",
        return_type="tuple",
        docstring="A simple test equation.",
    )


@pytest.fixture
def sample_program(sample_function: ParsedFunction) -> Program:
    """A minimal Program containing one function."""
    return Program(
        preface='import numpy as np\n\n"""Test specification."""',
        functions=[sample_function],
    )


@pytest.fixture
def db_config() -> ProgramsDatabaseConfig:
    """A database config with small values for fast tests."""
    return ProgramsDatabaseConfig(
        functions_per_prompt=2,
        num_islands=3,
        reset_period=50,
        cluster_sampling_temperature_init=0.1,
        cluster_sampling_temperature_period=20,
    )


SAMPLE_SPEC = '''\
"""Test specification for symbolic regression."""
import numpy as np

# @evaluate.run
def evaluate(data: dict) -> float:
    """Evaluate the equation."""
    return -1.0, None

# @equation.evolve
def equation(x, params):
    """Compute output."""
    return params[0] * x
'''
