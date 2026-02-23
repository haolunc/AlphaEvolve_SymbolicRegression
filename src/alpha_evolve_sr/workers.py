"""Worker functions for the distributed multiprocessing pipeline."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from multiprocessing import Event, Queue, Value

from . import code_manipulation
from . import evaluator as evaluator_mod
from . import sampler as sampler_mod
from .config import EvaluatorConfig, RunConfig, SamplerConfig, WorkerConfig
from .database import ProgramsDatabase
from .logging_config import get_logger, setup_file_logger
from .messages import PerfMessage, SampleMessage

logger = get_logger("workers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close_queues(queues: list[Queue], worker_logger: logging.Logger) -> None:
    """Close a list of multiprocessing queues, ignoring errors."""
    for q in queues:
        try:
            q.close()
            q.join_thread()
        except Exception:
            worker_logger.debug("Error closing queue", exc_info=True)


def _maybe_report_perf(
    start_time: float,
    interval: int,
    perf_queue: Queue,
    worker_type: str,
    worker_id: int,
    stats: dict,
) -> float:
    """Send a PerfMessage if *interval* seconds have elapsed. Returns updated start_time."""
    if time.time() - start_time > interval:
        perf_queue.put(PerfMessage(worker_type=worker_type, worker_id=worker_id, stats=stats))
        return time.time()
    return start_time


# ---------------------------------------------------------------------------
# Database worker
# ---------------------------------------------------------------------------

def database_worker(
    run_config: RunConfig,
    prompt_text: str,
    prompt_queue: Queue,
    prompt_pending_count: Value,
    result_queue: Queue,
    initial_result_queue: Queue,
    termination_event: Event,
    perf_queue: Queue,
) -> None:
    """Worker function for the database process."""
    wc = run_config.worker
    wlog = get_logger("database_worker")
    wlog.info("Database worker started (PID: %d)", os.getpid())

    # Initialize: resume from checkpoint or wait for initial eval
    initial_result = None
    if not run_config.resume_from_ckpt:
        wlog.info("Waiting for initial program evaluation...")
        initial_result = initial_result_queue.get()
        wlog.info("Initial program registered")

    database = ProgramsDatabase.restore_or_create(
        run_config.database, prompt_text, run_config.log_dir,
        ckpt_dir=run_config.save_ckpt_dir,
        max_samples=run_config.max_samples,
        resume_path=run_config.resume_from_ckpt, initial_result=initial_result,
        run_config=run_config,
    )

    prompts_generated = 0
    results_processed = 0
    start_time = time.time()
    show_prompt = False

    try:
        while not termination_event.is_set():
            # Fill prompt queue
            while prompt_pending_count.value < run_config.num_samplers and not termination_event.is_set():
                prompt = database.get_prompt()
                if not show_prompt:
                    logger.debug("First prompt:\n%s", prompt.code)
                    show_prompt = True
                prompt_queue.put(prompt)
                with prompt_pending_count.get_lock():
                    prompt_pending_count.value += 1
                prompts_generated += 1

            # Process results
            try:
                eval_result, sample_msg = result_queue.get(timeout=0.1)
                database.register_program(eval_result, sample_msg)
                results_processed += 1

                if database.should_stop:
                    wlog.info("Reached max samples (%d), setting termination event", run_config.max_samples)
                    termination_event.set()
            except mp.queues.Empty:
                pass
            except Exception as e:
                wlog.error("Database worker error: %s", e)

            # Periodic performance report
            start_time = _maybe_report_perf(
                start_time, wc.perf_report_interval_seconds, perf_queue,
                "database", 0, {
                    "prompts_generated": prompts_generated,
                    "results_processed": results_processed,
                    "global_sample_nums": database.sample_count,
                },
            )
    except KeyboardInterrupt:
        wlog.info("KeyboardInterrupt received, shutting down database worker")
    finally:
        database.finalize()
        wlog.info("Database worker shutting down")
        _close_queues([prompt_queue, result_queue, initial_result_queue, perf_queue], wlog)


# ---------------------------------------------------------------------------
# Sampler worker
# ---------------------------------------------------------------------------

def sampler_worker(
    worker_id: int,
    num_evaluators: int,
    prompt_queue: Queue,
    prompt_pending_count: Value,
    sample_queue: Queue,
    sample_pending_count: Value,
    termination_event: Event,
    perf_queue: Queue,
    worker_config: WorkerConfig | None = None,
    sampler_config: SamplerConfig | None = None,
) -> None:
    """Worker function for sampler processes."""
    wc = worker_config or WorkerConfig()
    sc = sampler_config or SamplerConfig()
    wlog = get_logger(f"sampler_worker_{worker_id}")
    wlog.info("Sampler worker %d started (PID: %d)", worker_id, os.getpid())

    llm = sampler_mod.LLM(config=sc)

    prompts_processed = 0
    samples_generated = 0
    start_time = time.time()
    network_error_count = 0

    try:
        while not termination_event.is_set():
            if sample_pending_count.value >= num_evaluators:
                wlog.info("Sampler %d: %d samples pending, waiting", worker_id, sample_pending_count.value)
                time.sleep(3)
                continue
            try:
                prompt = prompt_queue.get(timeout=0.5)
                with prompt_pending_count.get_lock():
                    prompt_pending_count.value -= 1

                reset_time = time.time()
                all_samples_info = llm.draw_samples(prompt.code)
                sample_time = (time.time() - reset_time) / sc.samples_per_prompt

                if all_samples_info is None:
                    network_error_count += 1
                    wlog.warning("Sampler %d: network error #%d", worker_id, network_error_count)
                    if network_error_count > 10:
                        wlog.warning("Sampler %d: too many network errors, shutting down", worker_id)
                        termination_event.set()
                    continue

                for sample_info in all_samples_info:
                    if sample_info:
                        sample_queue.put(SampleMessage(
                            llm_response=sample_info,
                            island_id=prompt.island_id,
                            sample_time=sample_time,
                        ))
                        samples_generated += 1
                        with sample_pending_count.get_lock():
                            sample_pending_count.value += 1

                prompts_processed += 1

                start_time = _maybe_report_perf(
                    start_time, wc.perf_report_interval_seconds, perf_queue,
                    "sampler", worker_id, {
                        "prompts_processed": prompts_processed,
                        "samples_generated": samples_generated,
                    },
                )
            except mp.queues.Empty:
                pass
            except Exception as e:
                wlog.warning("Sampler worker %d error: %s", worker_id, e)
    except KeyboardInterrupt:
        wlog.info("KeyboardInterrupt received, shutting down sampler worker %d", worker_id)
    except Exception as e:
        wlog.error("Sampler worker %d error: %s", worker_id, e)
    finally:
        llm.clean()
        _close_queues([prompt_queue, sample_queue, perf_queue], wlog)
        wlog.info("Sampler worker %d shutting down", worker_id)


# ---------------------------------------------------------------------------
# Evaluator worker
# ---------------------------------------------------------------------------

def evaluator_worker(
    worker_id: int,
    evaluate_code: str,
    seed_function: code_manipulation.ParsedFunction,
    data_dict: dict,
    sample_queue: Queue,
    sample_pending_count: Value,
    result_queue: Queue,
    initial_result_queue: Queue,
    termination_event: Event,
    perf_queue: Queue,
    process_initial: bool = False,
    worker_config: WorkerConfig | None = None,
    evaluator_config: EvaluatorConfig | None = None,
) -> None:
    """Worker function for evaluator processes."""
    wc = worker_config or WorkerConfig()
    wlog = get_logger(f"evaluator_worker_{worker_id}")
    wlog.info("Evaluator worker %d started (PID: %d)", worker_id, os.getpid())

    eval_instance = evaluator_mod.Evaluator(
        evaluate_code, seed_function, data_dict,
        config=evaluator_config,
    )

    if process_initial:
        wlog.info("Processing initial program")
        eval_result = eval_instance.initialize()
        initial_result_queue.put(eval_result)
        wlog.info("Initial program evaluation complete")

    samples_processed = 0
    successful_evaluations = 0
    failed_evaluations = 0
    start_time = time.time()

    try:
        while not termination_event.is_set():
            try:
                sample_msg = sample_queue.get(timeout=0.5)
                with sample_pending_count.get_lock():
                    sample_pending_count.value -= 1

                eval_result = eval_instance.analyse(sample_msg)
                if eval_result is not None:
                    result_queue.put((eval_result, sample_msg))
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1

                samples_processed += 1

                start_time = _maybe_report_perf(
                    start_time, wc.perf_report_interval_seconds, perf_queue,
                    "evaluator", worker_id, {
                        "samples_processed": samples_processed,
                        "successful_evaluations": successful_evaluations,
                        "failed_evaluations": failed_evaluations,
                    },
                )
            except mp.queues.Empty:
                pass
    except KeyboardInterrupt:
        wlog.info("KeyboardInterrupt received, shutting down evaluator worker %d", worker_id)
    except Exception as e:
        wlog.error("Evaluator worker %d error: %s", worker_id, e)
    finally:
        eval_instance.clean()
        _close_queues([sample_queue, result_queue, initial_result_queue, perf_queue], wlog)
        wlog.info("Evaluator worker %d shutting down", worker_id)


# ---------------------------------------------------------------------------
# Monitoring worker
# ---------------------------------------------------------------------------

def monitoring_worker(
    run_config: RunConfig,
    perf_queue: Queue,
    termination_event: Event,
) -> None:
    """Worker function for monitoring process performance."""
    wc = run_config.worker
    log_dir = run_config.log_dir or "./logger"
    wlog = setup_file_logger("monitor", log_dir)
    wlog.info("Monitoring worker started (PID: %d)", os.getpid())

    performance_data: dict = {
        "database": {"worker_id": 0, "stats": {}},
        "sampler": {i: {"stats": {}} for i in range(run_config.num_samplers)},
        "evaluator": {i: {"stats": {}} for i in range(run_config.num_evaluators)},
    }

    last_report_time = time.time()

    try:
        while not termination_event.is_set():
            try:
                while True:
                    perf_data = perf_queue.get_nowait()
                    worker_type = perf_data.worker_type
                    worker_id = perf_data.worker_id
                    stats = perf_data.stats
                    if worker_type == "database":
                        performance_data["database"]["stats"] = stats
                    else:
                        performance_data[worker_type][worker_id]["stats"] = stats
            except mp.queues.Empty:
                pass

            current_time = time.time()
            if current_time - last_report_time >= wc.monitor_interval_seconds:
                wlog.info("===== Performance Report =====")
                db_stats = performance_data["database"]["stats"]
                if db_stats:
                    wlog.info(
                        "Database: Prompts generated: %d, Results processed: %d, Total samples: %d",
                        db_stats.get("prompts_generated", 0),
                        db_stats.get("results_processed", 0),
                        db_stats.get("global_sample_nums", 0),
                    )
                wlog.info("Samplers:")
                for sid, sdata in performance_data["sampler"].items():
                    s = sdata["stats"]
                    if s:
                        wlog.info(
                            "  Sampler %d: Prompts: %d, Samples: %d",
                            sid, s.get("prompts_processed", 0), s.get("samples_generated", 0),
                        )
                wlog.info("Evaluators:")
                for eid, edata in performance_data["evaluator"].items():
                    s = edata["stats"]
                    if s:
                        wlog.info(
                            "  Evaluator %d: Processed: %d, Success: %d/%d",
                            eid, s.get("samples_processed", 0),
                            s.get("successful_evaluations", 0), s.get("samples_processed", 0),
                        )
                wlog.info("==============================")
                last_report_time = current_time

            time.sleep(10)
    except KeyboardInterrupt:
        wlog.info("KeyboardInterrupt received, shutting down monitoring worker")
    except Exception as e:
        wlog.error("Monitoring worker error: %s", e)
    finally:
        _close_queues([perf_queue], wlog)
        wlog.info("Monitoring worker shutting down")
