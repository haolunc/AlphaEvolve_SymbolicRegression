"""Entry point for the alpha_evolve_sr pipeline (distributed and non-distributed)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import pandas as pd

from . import checkpoint, code_manipulation
from . import evaluator as evaluator_mod
from . import sampler as sampler_mod
from .config import RunConfig
from .controller import EvolutionController
from .exceptions import SpecificationError
from .logging_config import configure_logging, get_logger
from .workers import database_worker, evaluator_worker, monitoring_worker, sampler_worker

logger = get_logger("cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_function_names(specification: str) -> tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run."""
    run_functions = list(code_manipulation.yield_decorated(specification, "evaluate", "run"))
    if len(run_functions) != 1:
        raise SpecificationError("Expected 1 function decorated with `# @evaluate.run`.")
    evolve_functions = list(code_manipulation.yield_decorated(specification, "equation", "evolve"))
    if len(evolve_functions) != 1:
        raise SpecificationError("Expected 1 function decorated with `# @equation.evolve`.")
    return evolve_functions[0], run_functions[0]


def load_data(spec_path: str, data_folder: str) -> tuple[str, dict]:
    """Loads specification and training data."""
    with open(spec_path, encoding="utf-8") as f:
        spec = f.read()

    df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    data_dict = {col: df[col].values for col in df.columns}

    return spec, data_dict


# ---------------------------------------------------------------------------
# Distributed mode
# ---------------------------------------------------------------------------

def main_distributed(
    run_config: RunConfig,
    specification: str,
    data_dict: dict,
) -> None:
    """Launches a distributed pipeline experiment."""
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    prompt_queue = mp.Queue()
    prompt_pending_count = mp.Value("i", 0)
    sample_pending_count = mp.Value("i", 0)
    sample_queue = mp.Queue()
    result_queue = mp.Queue()
    initial_result_queue = mp.Queue()
    perf_queue = mp.Queue()

    termination_event = mp.Event()

    processes: list[mp.Process] = []

    # Monitoring
    monitor_process = mp.Process(target=monitoring_worker, args=(run_config, perf_queue, termination_event))
    monitor_process.start()
    processes.append(monitor_process)

    process_initial = not run_config.resume_from_ckpt

    # Evaluators
    for i in range(run_config.num_evaluators):
        p = mp.Process(
            target=evaluator_worker,
            args=(
                i, template, function_to_evolve, function_to_run, data_dict,
                sample_queue, sample_pending_count, result_queue, initial_result_queue,
                termination_event, perf_queue, i == 0 and process_initial,
            ),
            kwargs={"evaluator_config": run_config.evaluator},
        )
        p.start()
        processes.append(p)

    if process_initial:
        time.sleep(2)

    # Database
    db_process = mp.Process(
        target=database_worker,
        args=(
            run_config, template, function_to_evolve,
            prompt_queue, prompt_pending_count, result_queue, initial_result_queue,
            termination_event, perf_queue,
        ),
    )
    db_process.start()
    processes.append(db_process)

    # Samplers
    for i in range(run_config.num_samplers):
        p = mp.Process(
            target=sampler_worker,
            args=(
                i, run_config.num_evaluators,
                prompt_queue, prompt_pending_count, sample_queue, sample_pending_count,
                termination_event, perf_queue,
            ),
            kwargs={"sampler_config": run_config.sampler},
        )
        p.start()
        processes.append(p)

    try:
        while not termination_event.is_set():
            time.sleep(5)

        for q in (prompt_queue, sample_queue, result_queue, initial_result_queue, perf_queue):
            try:
                q.close()
                q.join_thread()
            except Exception:
                logger.debug("Error closing queue", exc_info=True)

        for p in processes:
            p.join(10)
            if p.is_alive():
                logger.warning("%s still alive, terminating", p.name)
                p.terminate()

        logger.info("All processes have completed, main_distributed is exiting")
    except KeyboardInterrupt:
        termination_event.set()
        time.sleep(5)
        for process in processes:
            if process.is_alive():
                process.terminate()
        logger.info("Terminated due to KeyboardInterrupt, main_distributed is exiting")


# ---------------------------------------------------------------------------
# Non-distributed (single-process) mode
# ---------------------------------------------------------------------------

def main_single(
    run_config: RunConfig,
    specification: str,
    input_data: dict,
) -> None:
    """Launches a single-process experiment."""
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    evaluators = evaluator_mod.Evaluator(
        template, function_to_evolve, function_to_run, input_data,
        config=run_config.evaluator,
    )

    controller = EvolutionController(
        run_config.database, template, function_to_evolve, run_config.log_path,
        ckpt_dir=run_config.save_ckpt_dir, ckpt_interval=run_config.save_ckpt_interval,
        max_samples=run_config.max_samples,
        profiler_config=run_config.profiler,
    )

    initial_result = None
    if not run_config.resume_from_ckpt:
        initial_body = template.get_function(function_to_evolve).body
        initial_result = evaluators.analyse(initial_body, island_id=None, version_generated=None)

    controller.initialize(
        resume_path=run_config.resume_from_ckpt, initial_result=initial_result,
    )

    llm = sampler_mod.LLM(config=run_config.sampler)

    try:
        while True:
            prompt = controller.get_prompt()
            reset_time = time.time()
            all_samples_info = llm.draw_samples(prompt.code)
            sample_time = (time.time() - reset_time) / run_config.sampler.samples_per_prompt

            for sample_info in all_samples_info:
                if not sample_info:
                    continue
                try:
                    eval_result = evaluators.analyse(
                        sample_info.response_text,
                        prompt.island_id,
                        prompt.version_generated,
                        sample_time,
                        (sample_info.input_tokens, sample_info.output_tokens),
                        sample_info.token_cost,
                    )
                    if eval_result is not None:
                        controller.register_eval_result(eval_result)
                    else:
                        logger.warning("Error analysing sample: %s", sample_info.response_text)
                except Exception:
                    logger.warning("Error analysing sample: %s", getattr(sample_info, "response_text", "unknown"))

            controller.maybe_checkpoint()
            if controller.should_stop:
                break
    finally:
        controller.finalize()
        logger.info("Best program per complexity file written")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="AlphaEvolve Symbolic Regression")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None, help="Path to checkpoint directory to resume from",
    )
    args = parser.parse_args()

    configure_logging()

    run_config = RunConfig.from_yaml(args.config)

    # CLI override for resume
    if args.resume_from_ckpt:
        run_config.resume_from_ckpt = args.resume_from_ckpt

    run_config.validate()

    if run_config.resume_from_ckpt:
        logger.info("Resuming from checkpoint: %s", run_config.resume_from_ckpt)
        run_config = checkpoint.load_config(run_config.resume_from_ckpt)
        # Re-apply the CLI override after loading saved config
        if args.resume_from_ckpt:
            run_config.resume_from_ckpt = args.resume_from_ckpt

    # Derive log_path and save_ckpt_dir from log_folder if not set
    if run_config.log_path is None and run_config.log_folder:
        run_config.log_path = "./log/" + run_config.log_folder

    if run_config.save_ckpt_dir is None and run_config.log_folder:
        run_config.save_ckpt_dir = "./log/" + run_config.log_folder + "/checkpoints"

    if run_config.save_ckpt_dir:
        logger.info("Saving config to %s", run_config.save_ckpt_dir)
        checkpoint.save_config(run_config, run_config.save_ckpt_dir)

    mp.set_start_method("spawn")

    spec, data_dict = load_data(run_config.spec_path, run_config.data_folder)

    if run_config.distributed:
        logger.info("Running in distributed mode")
        main_distributed(run_config, spec, data_dict)
    else:
        logger.info("Running in non-distributed mode")
        main_single(run_config, spec, data_dict)
