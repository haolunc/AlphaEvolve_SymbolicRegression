"""Tests for profiler module."""

from __future__ import annotations

from alpha_evolve_sr.profiler import ProfileMetrics, TensorBoardWriter


class TestTensorBoardWriter:
    """Tests for TensorBoardWriter.write(ProfileMetrics)."""

    def test_write_completes_without_error(self, tmp_path):
        """A basic write with ProfileMetrics should not raise."""
        writer = TensorBoardWriter(
            log_dir=str(tmp_path / "logs"),
        )
        metrics = ProfileMetrics(
            num_samples=1,
            best_score=-1.0,
            tot_token_cost=0.001,
            success_count=1,
            failed_count=0,
            tot_sample_time=0.1,
            tot_evaluate_time=0.2,
        )
        writer.write(metrics)

    def test_write_with_optional_fields(self, tmp_path):
        """Write with pareto_front, best_score_per_island, island_sizes."""
        from alpha_evolve_sr.database import ParetoEntry

        writer = TensorBoardWriter(
            log_dir=str(tmp_path / "logs"), log_frequency=1,
        )
        metrics = ProfileMetrics(
            num_samples=1,
            best_score=-0.5,
            tot_token_cost=0.01,
            success_count=5,
            failed_count=2,
            tot_sample_time=1.0,
            tot_evaluate_time=2.0,
            pareto_front=[
                ParetoEntry(cbin=0, score=-1.0, gsn=1),
                ParetoEntry(cbin=1, score=-0.5, gsn=2),
            ],
            best_score_per_island=[-1.0, -0.5],
            island_sizes=[10, 20],
        )
        writer.write(metrics)
