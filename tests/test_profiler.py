"""Tests for profiler module."""

from __future__ import annotations

import os

from alpha_evolve_sr.code_manipulation import EvaluatedProgram, ParsedFunction
from alpha_evolve_sr.profiler import Profiler


def _make_program(sample_order: int, score: float = -1.0) -> EvaluatedProgram:
    """Helper to create an EvaluatedProgram with the given sample order and score."""
    parsed = ParsedFunction(
        name="equation",
        args="x, params",
        body="    return x",
    )
    return EvaluatedProgram(
        parsed=parsed,
        score=score,
        optimized_params=None,
        complexity=5,
        complexity_detail={"BinOp": 1},
        global_sample_nums=sample_order,
        sample_time=0.1,
        evaluate_time=0.2,
        token_usage=(10, 20),
        token_cost=0.001,
    )


class TestProfilerCounter:
    """Tests for Profiler counter logic."""

    def test_sequential_samples_all_logged(self, tmp_path):
        """Register 3 samples in order — all should produce JSON files."""
        profiler = Profiler(num_islands=2, log_dir=str(tmp_path / "logs"))

        for i in range(1, 4):
            profiler.register_function(_make_program(sample_order=i))

        json_dir = tmp_path / "logs" / "samples"
        json_files = list(json_dir.glob("samples_*.json"))
        assert len(json_files) == 3

    def test_out_of_order_samples_not_lost(self, tmp_path):
        """Register samples as [1, 3, 2] — all 3 should be logged."""
        profiler = Profiler(num_islands=2, log_dir=str(tmp_path / "logs"))

        for order in [1, 3, 2]:
            profiler.register_function(_make_program(sample_order=order))

        json_dir = tmp_path / "logs" / "samples"
        json_files = list(json_dir.glob("samples_*.json"))
        assert len(json_files) == 3

    def test_high_water_mark_tracked(self, tmp_path):
        """_num_samples should track the highest sample order seen."""
        profiler = Profiler(num_islands=2, log_dir=str(tmp_path / "logs"))

        profiler.register_function(_make_program(sample_order=1))
        assert profiler._num_samples == 1

        profiler.register_function(_make_program(sample_order=5))
        assert profiler._num_samples == 5

        # Out-of-order sample — high-water mark shouldn't decrease
        profiler.register_function(_make_program(sample_order=3))
        assert profiler._num_samples == 5
