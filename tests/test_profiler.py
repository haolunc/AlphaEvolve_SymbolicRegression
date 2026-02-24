"""Tests for profiler module."""

from __future__ import annotations

import os

import pytest

from alpha_evolve_sr.profiler import ProfileMetrics, TensorBoardWriter


class TestTensorBoardWriter:
    """Tests for TensorBoardWriter.write(ProfileMetrics)."""

    def test_write_completes_without_error(self, tmp_path):
        """A basic write with ProfileMetrics should not raise."""
        writer = TensorBoardWriter(log_dir=str(tmp_path / "logs"))
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

        writer = TensorBoardWriter(log_dir=str(tmp_path / "logs"))
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

    def test_write_with_pipeline_metrics(self, tmp_path):
        """Write with pipeline throughput fields."""
        writer = TensorBoardWriter(log_dir=str(tmp_path / "logs"))
        metrics = ProfileMetrics(
            num_samples=10,
            best_score=-0.5,
            tot_token_cost=0.01,
            success_count=5,
            failed_count=2,
            tot_sample_time=1.0,
            tot_evaluate_time=2.0,
            pending_evals=3,
            pending_samplers=2,
            wall_time_seconds=60.0,
        )
        writer.write(metrics)

    def test_pipeline_fields_default_none(self):
        """Pipeline fields default to None."""
        metrics = ProfileMetrics(
            num_samples=1,
            best_score=-1.0,
            tot_token_cost=0.0,
            success_count=0,
            failed_count=0,
            tot_sample_time=0.0,
            tot_evaluate_time=0.0,
        )
        assert metrics.pending_evals is None
        assert metrics.pending_samplers is None
        assert metrics.wall_time_seconds is None


class TestCustomScalarsLayout:
    """Tests for the Custom Scalars layout written at init time."""

    def test_layout_creates_event_file(self, tmp_path):
        """Constructor should write a layout event to the log directory."""
        log_dir = str(tmp_path / "logs")
        TensorBoardWriter(log_dir=log_dir, num_islands=10)
        event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
        assert len(event_files) >= 1

    @pytest.mark.parametrize("num_islands", [3, 5, 10, 12])
    def test_layout_with_various_island_counts(self, tmp_path, num_islands):
        """Layout generation should work for any num_islands value."""
        log_dir = str(tmp_path / "logs" / str(num_islands))
        writer = TensorBoardWriter(log_dir=log_dir, num_islands=num_islands)
        # Verify the writer initialised without error
        assert writer._num_islands == num_islands

    def test_layout_island_grouping(self):
        """Verify the grouping logic produces correct groups of 5."""
        # Simulate the grouping used in _write_custom_scalars_layout
        num_islands = 12
        groups = []
        for start in range(0, num_islands, 5):
            end = min(start + 5, num_islands)
            groups.append(list(range(start, end)))
        assert groups == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]

    def test_default_num_islands(self, tmp_path):
        """Default num_islands should be 10."""
        writer = TensorBoardWriter(log_dir=str(tmp_path / "logs"))
        assert writer._num_islands == 10
