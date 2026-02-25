"""Entry point for the alpha_evolve_sr pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import os
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pandas as pd

from . import code_manipulation
from . import sampler as sampler_mod
from .complexity import complexity_score
from .config import EvaluatorConfig, RunConfig
from .database import ProgramsDatabase
from .evaluator import Evaluator
from .logging_config import configure_logging, get_logger
from .messages import EvalResult, SampleMessage

logger = get_logger("cli")


# ---------------------------------------------------------------------------
# Thread-local evaluator management
# ---------------------------------------------------------------------------

_eval_tls = threading.local()


def _init_eval_thread(
    evaluate_code: str,
    seed_function: code_manipulation.ParsedFunction,
    data_dict: dict,
    config: EvaluatorConfig,
) -> None:
    """ThreadPoolExecutor initializer: create a persistent Evaluator per thread."""
    _eval_tls.evaluator = Evaluator(evaluate_code, seed_function, data_dict, config=config)


def _eval_thread_analyse(
    sample_msg: SampleMessage,
) -> tuple[EvalResult, SampleMessage]:
    """Evaluate a sample in the calling thread's Evaluator."""
    result = _eval_tls.evaluator.analyse(sample_msg)
    return (result, sample_msg)


def _eval_thread_initialize() -> EvalResult:
    """Evaluate the seed function in the calling thread's Evaluator."""
    return _eval_tls.evaluator.initialize()


# ---------------------------------------------------------------------------
# Complexity attachment (runs in main thread)
# ---------------------------------------------------------------------------

def _attach_complexity(eval_result: EvalResult) -> EvalResult:
    """Compute complexity and attach it to *eval_result*."""
    if eval_result.execution_result is None:
        return eval_result
    try:
        c_val, c_detail = complexity_score(str(eval_result.function), return_breakdown=True)
    except Exception as e:
        c_val, c_detail = 200, {"error": f"Complexity evaluation failed: {e}"}
        logger.warning("Complexity evaluation failed: %s", e)
        return dataclasses.replace(
            eval_result, complexity=c_val, complexity_detail=c_detail,
            error_type=eval_result.error_type or "complexity",
            error_message=eval_result.error_message or str(e),
        )
    return dataclasses.replace(eval_result, complexity=c_val, complexity_detail=c_detail)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_problem(
    problem_dir: str,
    data_folder: str,
) -> tuple[str, str, code_manipulation.ParsedFunction, dict]:
    """Load a problem directory and training data.

    Returns:
        prompt_text, evaluate_code, seed_function, data_dict
    """
    with open(os.path.join(problem_dir, "prompt.txt"), encoding="utf-8") as f:
        prompt_text = f.read()

    with open(os.path.join(problem_dir, "evaluate.py"), encoding="utf-8") as f:
        evaluate_code = f.read()

    with open(os.path.join(problem_dir, "equation.py"), encoding="utf-8") as f:
        equation_code = f.read()

    seed_function = code_manipulation.text_to_function(equation_code)

    df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    data_dict = {col: df[col].values for col in df.columns}

    return prompt_text, evaluate_code, seed_function, data_dict


# ---------------------------------------------------------------------------
# Sampler task (submitted to ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _sampler_task(
    llm: sampler_mod.LLM,
    prompt_code: str,
    island_id: int,
) -> SampleMessage | None:
    """Make a single LLM call and wrap the result in a ``SampleMessage``."""
    t0 = time.time()
    resp = llm.query(prompt_code)
    if resp is None:
        return None
    return SampleMessage(
        llm_response=resp,
        island_id=island_id,
        sample_time=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------

def _cleanup_eval_threads(eval_pool: ThreadPoolExecutor, num_workers: int) -> None:
    """Clean up Sandbox subprocesses in all evaluator threads."""
    futs = []
    for _ in range(num_workers):
        futs.append(eval_pool.submit(lambda: _eval_tls.evaluator.clean()))
    for f in futs:
        try:
            f.result(timeout=5)
        except Exception:
            pass


def _drain_pipeline(
    pending_sampler_futures: set[Future],
    pending_eval_futures: set[Future],
    eval_pool: ThreadPoolExecutor,
    database: ProgramsDatabase,
    force_event: threading.Event,
) -> None:
    """Drain all in-flight sampler and eval futures, registering results.

    Called on normal exit (max_samples reached) and on first Ctrl+C.
    A second Ctrl+C sets *force_event*, which aborts draining immediately.
    """
    # Phase 1: wait for sampler futures → submit results to eval
    for fut in list(pending_sampler_futures):
        if force_event.is_set():
            return
        try:
            msg = fut.result(timeout=120)
            if msg is not None:
                logger.info(
                    "Drain sample completed: island=%d, tokens=(%d,%d), time=%.2fs",
                    msg.island_id,
                    msg.llm_response.input_tokens,
                    msg.llm_response.output_tokens,
                    msg.sample_time,
                )
                ef = eval_pool.submit(_eval_thread_analyse, msg)
                pending_eval_futures.add(ef)
        except Exception as e:
            logger.error("Drain sampler error: %s", e)

    # Phase 2: wait for all eval futures → register
    for ef in list(pending_eval_futures):
        if force_event.is_set():
            return
        try:
            eval_result, sample_msg = ef.result(timeout=60)
            eval_result = _attach_complexity(eval_result)
            database.register_program(eval_result, sample_msg)
            if eval_result.execution_result is not None:
                logger.info(
                    "Drain eval success: score=%.6g, complexity=%s, eval_time=%.2fs, gsn=%d",
                    eval_result.execution_result.score,
                    eval_result.complexity,
                    eval_result.evaluate_time,
                    database.sample_count,
                )
            else:
                logger.info(
                    "Drain eval failed: error_type=%s, error=%s, gsn=%d",
                    eval_result.error_type,
                    eval_result.error_message,
                    database.sample_count,
                )
        except Exception as e:
            logger.error("Drain eval error: %s", e)


def run_pipeline(
    run_config: RunConfig,
    prompt_text: str,
    evaluate_code: str,
    seed_function: code_manipulation.ParsedFunction,
    data_dict: dict,
) -> None:
    """Run the sampling-evaluation pipeline.

    Architecture::

        Main Thread (orchestrator + database + profiler + complexity)
        ├── ThreadPoolExecutor (num_samplers threads for LLM I/O)
        └── ThreadPoolExecutor (num_evaluators threads for sandbox eval)
            └── Each thread has Evaluator (thread-local)
                └── Sandbox(mp.Pool(1))  ← only mp.Pool left
    """
    eval_pool = ThreadPoolExecutor(
        max_workers=run_config.num_evaluators,
        initializer=_init_eval_thread,
        initargs=(evaluate_code, seed_function, data_dict, run_config.evaluator),
    )

    try:
        # Seed evaluation (blocking)
        initial_result = None
        if not run_config.resume_from_ckpt:
            initial_result = eval_pool.submit(_eval_thread_initialize).result()
            if initial_result is None or initial_result.execution_result is None:
                err_detail = ""
                if initial_result is not None:
                    err_detail = f": [{initial_result.error_type}] {initial_result.error_message}"
                raise RuntimeError(
                    f"Seed function evaluation failed{err_detail}. "
                    "Cannot start without a valid seed program."
                )
            initial_result = _attach_complexity(initial_result)
            logger.info("Initial seed evaluation complete: execution_score=%s, complexity=%s",
                initial_result.execution_result.score, initial_result.complexity)

        database = ProgramsDatabase.restore_or_create(
            run_config.database, prompt_text, run_config.log_dir,
            seed_function=seed_function,
            ckpt_dir=run_config.save_ckpt_dir,
            max_samples=run_config.max_samples,
            resume_path=run_config.resume_from_ckpt,
            initial_result=initial_result,
        )

        llm = sampler_mod.LLM(config=run_config.sampler)
        sampler_pool = ThreadPoolExecutor(max_workers=run_config.num_samplers)

        pending_sampler_futures: set[Future] = set()
        pending_eval_futures: set[Future] = set()
        max_pending = run_config.num_evaluators * 2  # backpressure

        pipeline_start = time.time()

        # -- Two-phase Ctrl+C shutdown via signal handler --
        graceful_shutdown = threading.Event()
        force_shutdown = threading.Event()
        original_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):  # noqa: ARG001
            if graceful_shutdown.is_set():
                logger.info("Second Ctrl+C: forcing immediate shutdown")
                force_shutdown.set()
            else:
                logger.info("First Ctrl+C: draining in-flight work...")
                graceful_shutdown.set()

        signal.signal(signal.SIGINT, _sigint_handler)

        try:
            while not database.should_stop and not graceful_shutdown.is_set():
                # 1. Collect completed evals -> register in DB
                done_eval_futures: set[Future] = set()
                for fut in pending_eval_futures:
                    if fut.done():
                        done_eval_futures.add(fut)
                        try:
                            eval_result, sample_msg = fut.result()
                            eval_result = _attach_complexity(eval_result)
                            database.register_program(eval_result, sample_msg)
                            if eval_result.execution_result is not None:
                                logger.info(
                                    "Eval success: score=%.6g, complexity=%s, eval_time=%.2fs, gsn=%d",
                                    eval_result.execution_result.score,
                                    eval_result.complexity,
                                    eval_result.evaluate_time,
                                    database.sample_count,
                                )
                            else:
                                logger.info(
                                    "Eval failed: error_type=%s, error=%s, gsn=%d",
                                    eval_result.error_type,
                                    eval_result.error_message,
                                    database.sample_count,
                                )
                        except Exception as e:
                            logger.error("Eval worker error: %s", e)
                pending_eval_futures -= done_eval_futures

                # 2. Collect completed samples -> submit to eval pool
                done_futures: set[Future] = set()
                for fut in pending_sampler_futures:
                    if fut.done():
                        done_futures.add(fut)
                        try:
                            msg = fut.result()
                            if msg is not None:
                                logger.info(
                                    "Sample completed: island=%d, tokens=(%d,%d), time=%.2fs",
                                    msg.island_id,
                                    msg.llm_response.input_tokens,
                                    msg.llm_response.output_tokens,
                                    msg.sample_time,
                                )
                                ef = eval_pool.submit(_eval_thread_analyse, msg)
                                pending_eval_futures.add(ef)
                        except Exception as e:
                            logger.error("Sampler error: %s", e)
                pending_sampler_futures -= done_futures

                # 3. Submit new prompts (if under backpressure limit)
                while (
                    len(pending_eval_futures) < max_pending
                    and len(pending_sampler_futures) < run_config.num_samplers
                    and not database.should_stop
                ):
                    prompt = database.get_prompt()
                    for _ in range(run_config.sampler.samples_per_prompt):
                        fut = sampler_pool.submit(
                            _sampler_task, llm, prompt.code, prompt.island_id,
                        )
                        pending_sampler_futures.add(fut)

                # 4. Update pipeline stats for profiler
                database.update_pipeline_stats(
                    pending_evals=len(pending_eval_futures),
                    pending_samplers=len(pending_sampler_futures),
                    wall_time_seconds=time.time() - pipeline_start,
                )

                time.sleep(0.05)  # avoid busy-wait

            # --- Drain phase (normal exit & first Ctrl+C) ---
            if not force_shutdown.is_set():
                reason = "Ctrl+C" if graceful_shutdown.is_set() else "max_samples reached"
                logger.info(
                    "Pipeline stopping (%s). Draining %d samplers, %d evals...",
                    reason, len(pending_sampler_futures), len(pending_eval_futures),
                )
                _drain_pipeline(
                    pending_sampler_futures, pending_eval_futures,
                    eval_pool, database, force_shutdown,
                )
        except KeyboardInterrupt:
            # Safety net: if a KeyboardInterrupt slips through before the
            # signal handler is fully installed, just force-shutdown.
            logger.info("KeyboardInterrupt during drain, forcing shutdown")
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            sampler_pool.shutdown(wait=False, cancel_futures=True)
            llm.clean()
            database.finalize()
    finally:
        _cleanup_eval_threads(eval_pool, run_config.num_evaluators)
        eval_pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="AlphaEvolve Symbolic Regression")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    run_config = RunConfig.from_yaml(args.config)
    run_config.validate()
    configure_logging(log_file=os.path.join(run_config.log_dir, "pipeline.log"))

    if run_config.resume_from_ckpt:
        logger.info("Resuming from checkpoint: %s", run_config.resume_from_ckpt)
    else:
        logger.info("Fresh Start without resume!")

    mp.set_start_method("spawn")

    prompt_text, evaluate_code, seed_function, data_dict = load_problem(
        run_config.problem_dir, run_config.data_folder,
    )

    logger.info(
        "Running pipeline with %d sampler threads, %d evaluator threads",
        run_config.num_samplers, run_config.num_evaluators,
    )
    run_pipeline(
        run_config, prompt_text, evaluate_code, seed_function, data_dict,
    )
