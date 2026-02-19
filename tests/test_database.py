"""Tests for the programs database."""

import pytest

from alpha_evolve_sr.code_manipulation import EvaluatedProgram, ParsedFunction
from alpha_evolve_sr.config import ProgramsDatabaseConfig
from alpha_evolve_sr.database import Island
from alpha_evolve_sr.messages import EvalResult, ExecutionResult, LLMResponse, SampleMessage
from tests.conftest import SAMPLE_PROMPT, SAMPLE_SEED_FUNCTION


def _make_result(function, island_id=None, score=-1.0, complexity=5):
    """Build an (EvalResult, SampleMessage | None) pair for testing."""
    eval_result = EvalResult(
        function=function,
        execution_result=ExecutionResult(
            score=score, optimized_params=None,
            complexity=complexity, complexity_detail={},
        ),
        evaluate_time=0.1,
    )
    sample_msg = None
    if island_id is not None:
        sample_msg = SampleMessage(
            llm_response=LLMResponse(
                response_text="return x",
                input_tokens=10,
                output_tokens=20,
                token_cost=0.001,
            ),
            island_id=island_id,
            sample_time=0.1,
        )
    return eval_result, sample_msg


@pytest.fixture
def db(tmp_path):
    """Create a small database for testing."""
    config = ProgramsDatabaseConfig(
        functions_per_prompt=2,
        num_islands=2,
        reset_period=100,
        cluster_sampling_temperature_init=0.1,
        cluster_sampling_temperature_period=20,
    )
    log_dir = str(tmp_path / "logs")

    from alpha_evolve_sr.database import ProgramsDatabase
    database = ProgramsDatabase(config, SAMPLE_PROMPT, log_dir)

    database.register_program(*_make_result(SAMPLE_SEED_FUNCTION, island_id=None, score=-1.0, complexity=5))
    return database


class TestProgramsDatabase:
    def test_get_prompt_returns_prompt(self, db):
        prompt = db.get_prompt()
        assert hasattr(prompt, "code")
        assert hasattr(prompt, "island_id")
        assert isinstance(prompt.code, str)
        assert len(prompt.code) > 0

    def test_register_increments_sample_count(self, db):
        initial_count = db.sample_count
        func = ParsedFunction(
            name="equation",
            args="x, params",
            body="    return x * params[0]",
        )
        db.register_program(*_make_result(func, island_id=0, score=-0.5, complexity=3))
        assert db.sample_count == initial_count + 1

    def test_island_reset(self, db):
        """Register enough programs to trigger a reset."""
        for i in range(db._config.reset_period + 2):
            func = ParsedFunction(
                name="equation",
                args="x, params",
                body=f"    return x * {i}",
            )
            db.register_program(
                *_make_result(func, island_id=i % db._config.num_islands, score=float(-100 + i), complexity=3),
            )
        # Should not raise and sample count should be correct
        assert db.sample_count > db._config.reset_period

    def test_finalize_writes_file(self, db):
        """finalize() writes the best-program-per-complexity file."""
        db.finalize()
        import os
        output_path = os.path.join(db._profiler._log_dir, "best_programs_per_complexity.txt")
        assert os.path.exists(output_path)


class TestClusterPruning:
    """Tests for Island bin max-size pruning."""

    def test_bin_respects_max_size(self):
        """Adding more programs than max_size triggers pruning."""
        island = Island(functions_per_prompt=2, complexity_bin_size=10, cluster_max_size=5)

        for i in range(10):
            island.register(gsn=i, score=float(i), complexity=5)

        assert island.num_programs <= 5

    def test_pruning_keeps_highest_scores(self):
        """Pruning should discard the lowest-scoring programs."""
        island = Island(functions_per_prompt=2, complexity_bin_size=10, cluster_max_size=5)

        for i in range(10):
            island.register(gsn=i, score=float(i), complexity=5)

        # The 5 highest scores should be 5.0, 6.0, 7.0, 8.0, 9.0
        cbin = 5 // 10  # = 0
        scores = sorted(s for _, s in island._bins[cbin])
        assert scores == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestParetoFront:
    """Tests for Pareto front tracking."""

    def test_initial_program_on_front(self, db):
        """The first registered program should be on the Pareto front."""
        assert len(db.pareto_front) == 1

    def test_dominated_program_not_added(self, db):
        """A program dominated by an existing front member is not added."""
        func = ParsedFunction(name="equation", args="x, params", body="    return x")
        # Worse score, same complexity -> dominated
        db.register_program(*_make_result(func, island_id=0, score=-2.0, complexity=5))
        assert len(db.pareto_front) == 1

    def test_non_dominated_extends_front(self, db):
        """A non-dominated program extends the front."""
        func = ParsedFunction(name="equation", args="x, params", body="    return x")
        # Better score, higher complexity -> non-dominated
        db.register_program(*_make_result(func, island_id=0, score=-0.5, complexity=20))
        assert len(db.pareto_front) == 2

    def test_dominating_program_prunes_front(self, db):
        """Adding a program that dominates existing members prunes them."""
        func = ParsedFunction(name="equation", args="x, params", body="    return x")
        # Better score AND lower complexity -> dominates the initial
        db.register_program(*_make_result(func, island_id=0, score=-0.5, complexity=3))
        assert len(db.pareto_front) == 1
        assert db.pareto_front[0].score == -0.5

    def test_front_sorted_by_complexity(self, db):
        func = ParsedFunction(name="equation", args="x, params", body="    return x")
        for c, s in [(20, -0.5), (3, -2.0), (10, -0.8)]:
            db.register_program(*_make_result(func, island_id=0, score=s, complexity=c))
        complexities = [p.complexity for p in db.pareto_front]
        assert complexities == sorted(complexities)

    def test_pareto_aware_get_prompt(self, tmp_path):
        """get_prompt works with pareto_aware=True."""
        config = ProgramsDatabaseConfig(
            functions_per_prompt=2, num_islands=2, reset_period=100,
            cluster_sampling_temperature_init=0.1, cluster_sampling_temperature_period=20,
            pareto_aware=True,
        )
        from alpha_evolve_sr.database import ProgramsDatabase
        database = ProgramsDatabase(config, SAMPLE_PROMPT, str(tmp_path / "logs"))

        # Register a few programs with different complexities to build a Pareto front
        for c, s in [(5, -1.0), (15, -0.5), (25, -0.3)]:
            database.register_program(
                *_make_result(SAMPLE_SEED_FUNCTION, island_id=None, score=s, complexity=c),
            )

        prompt = database.get_prompt()
        assert len(prompt.code) > 0

    def test_finalize_writes_pareto_file(self, db, tmp_path):
        """finalize() writes pareto_front.py when front is non-empty."""
        import os
        db.finalize()
        pareto_path = os.path.join(db._profiler._log_dir, "pareto_front.py")
        assert os.path.exists(pareto_path)


class TestIslandProperties:
    """Tests for Island public properties."""

    def test_num_clusters(self, db):
        """num_clusters reflects the number of complexity bins in use."""
        island = db._islands[0]
        assert island.num_clusters >= 1

    def test_num_programs(self, db):
        """num_programs reflects how many programs have been registered."""
        island = db._islands[0]
        assert island.num_programs >= 1
