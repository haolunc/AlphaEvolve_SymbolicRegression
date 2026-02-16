"""Tests for workers module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alpha_evolve_sr.config import RunConfig


class TestMonitoringWorkerLogPath:
    """Verify monitoring_worker uses run_config.log_path for logger."""

    def test_log_path_from_run_config(self, tmp_path):
        """monitoring_worker should pass run_config.log_path to setup_file_logger."""
        log_dir = str(tmp_path / "custom_logs")
        run_config = RunConfig(
            log_path=log_dir,
            num_samplers=1,
            num_evaluators=1,
        )

        # Create a termination event that fires immediately
        import multiprocessing as mp
        term_event = mp.Event()
        term_event.set()

        perf_queue = MagicMock()
        perf_queue.get_nowait.side_effect = Exception("empty")

        with patch("alpha_evolve_sr.workers.setup_file_logger") as mock_sfl:
            mock_sfl.return_value = MagicMock()
            from alpha_evolve_sr.workers import monitoring_worker
            monitoring_worker(run_config, perf_queue, term_event)

        mock_sfl.assert_called_once_with("monitor", log_dir)

    def test_log_path_defaults_to_logger(self, tmp_path):
        """If run_config has no log_path, fallback to './logger'."""
        run_config = RunConfig(num_samplers=1, num_evaluators=1)

        import multiprocessing as mp
        term_event = mp.Event()
        term_event.set()

        perf_queue = MagicMock()
        perf_queue.get_nowait.side_effect = Exception("empty")

        with patch("alpha_evolve_sr.workers.setup_file_logger") as mock_sfl:
            mock_sfl.return_value = MagicMock()
            from alpha_evolve_sr.workers import monitoring_worker
            monitoring_worker(run_config, perf_queue, term_event)

        mock_sfl.assert_called_once_with("monitor", "./logger")
