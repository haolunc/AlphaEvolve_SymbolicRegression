"""Shared test fixtures for alpha_evolve_sr."""

import pytest

from alpha_evolve_sr.code_manipulation import (
    EvaluatedProgram,
    ParsedFunction,
    Program,
    text_to_function,
)
from alpha_evolve_sr.config import ProgramsDatabaseConfig
from alpha_evolve_sr.messages import EvalResult, ExecutionResult, LLMResponse, SampleMessage


# --- Split problem constants (replacing old monolithic SAMPLE_SPEC) ---

SAMPLE_PROMPT = "Test specification for symbolic regression."

SAMPLE_EVALUATE_CODE = '''\
import numpy as np

def evaluate(data: dict) -> float:
    """Evaluate the equation."""
    return -1.0, None
'''

SAMPLE_EQUATION_CODE = '''\
def equation(x, params):
    """Compute output."""
    return params[0] * x
'''

SAMPLE_SEED_FUNCTION = text_to_function(SAMPLE_EQUATION_CODE)


def make_eval_result(score=-1.0):
    """Build a valid EvalResult for testing."""
    func = ParsedFunction(
        name="equation",
        args="x, params",
        body="    return params[0] * x",
    )
    return EvalResult(
        function=func,
        execution_result=ExecutionResult(
            score=score, optimized_params=None,
        ),
        evaluate_time=0.2,
        complexity=5,
        complexity_detail={},
    )


def make_sample_message(island_id=0):
    """Build a valid SampleMessage for testing."""
    return SampleMessage(
        llm_response=LLMResponse(
            response_text="return x",
            input_tokens=10,
            output_tokens=20,
            token_cost=0.001,
        ),
        island_id=island_id,
        sample_time=0.1,
    )


def make_evaluated_program(gsn, score=-1.0, complexity=5):
    """Build an EvaluatedProgram for testing."""
    parsed = ParsedFunction(
        name="equation", args="x, params", body="    return params[0] * x",
    )
    return EvaluatedProgram(
        parsed=parsed,
        score=score,
        optimized_params=[1.0, 2.0],
        complexity=complexity,
        complexity_detail={"BinOp": 3},
        global_sample_nums=gsn,
        sample_time=0.1,
        evaluate_time=0.2,
        token_usage=(10, 20),
        token_cost=0.001,
    )


def finished_future(value):
    """Return a Future already resolved with *value*."""
    from concurrent.futures import Future
    f = Future()
    f.set_result(value)
    return f


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
