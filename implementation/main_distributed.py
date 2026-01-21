"""A distributed implementation of the physr pipeline.

The workflow is as follows:
1. An initial program is evaluated by the first evaluator process
2. The database process registers the initial program and begins generating prompts
3. Sampler processes take prompts from the database and generate program samples
4. Evaluator processes evaluate the samples and send results back to the database
5. The monitoring process tracks performance statistics and reports them periodically

Inter-process communication is implemented using multiprocessing queues:
- prompt_queue: Database -> Samplers
- sample_queue: Samplers -> Evaluators  
- result_queue: Evaluators -> Database
- initial_result_queue: Initial evaluation result
- perf_queue: Performance stats from all processes -> Monitoring process

The system terminates when the maximum number of samples is reached or when interrupted.
"""
from collections.abc import Sequence
from typing import Any, Dict, List
from argparse import ArgumentParser
import time
import multiprocessing as mp
import os
import numpy as np
import json
import pickle
from dataclasses import asdict

import code_manipulation
from config import ProgramsDatabaseConfig
import evaluator
import programs_database
import sampler
import pandas as pd
import logging
from logging_utils import setup_console_logger
import checkpoint_util
from distribution_util import database_worker, evaluator_worker, sampler_worker, monitoring_worker

def _extract_function_names(specification: str) -> tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run."""
    run_functions = list(
        code_manipulation.yield_decorated(specification, "evaluate", "run")
    )
    if len(run_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@evaluate.run`.")
    evolve_functions = list(
        code_manipulation.yield_decorated(specification, "equation", "evolve")
    )
    if len(evolve_functions) != 1:
        raise ValueError("Expected 1 function decorated with `@equation.evolve`.")
    return evolve_functions[0], run_functions[0]

def load_data(spec_path: str, data_folder: str) -> tuple[str, dict]:
    """Loads data from the specified path and problem name."""
    with open(spec_path, encoding="utf-8") as f:
        spec = f.read()

    df = pd.read_csv("./data/" + data_folder + "/train.csv")
    data_dict = {col: df[col].values for col in df.columns}

    return spec, data_dict


def main_distributed(arguments, config: ProgramsDatabaseConfig, specification: str, data_dict: dict, logger: logging.Logger):
    """Launches a distributed physr experiment."""
    
    # Extract function names from the specification
    function_to_evolve, function_to_run = _extract_function_names(specification)

    # Parse the specification into a template program
    template = code_manipulation.text_to_program(specification)
    
    # Create queues for inter-process communication
    prompt_queue = mp.Queue()  # Database -> Samplers
    prompt_pending_count = mp.Value('i', 0)
    sample_pending_count = mp.Value('i', 0)
    sample_queue = mp.Queue()  # Samplers -> Evaluators
    result_queue = mp.Queue()  # Evaluators -> Database
    initial_result_queue = mp.Queue()  # For the initial program evaluation
    perf_queue = mp.Queue()  # For performance monitoring
    
    # Create termination event
    termination_event = mp.Event()
    
    # Start processes
    processes = []
    
    # Start monitoring process
    monitor_process = mp.Process(
        target=monitoring_worker,
        args=(arguments, perf_queue, termination_event)
    )
    monitor_process.start()
    processes.append(monitor_process)
    
    if arguments.resume_from_ckpt:
        process_initial = False
    else:
        process_initial = True  # By default, process the initial program
    
    # Start evaluator processes - start one first to handle the initial program
    for i in range(args.num_evaluators):
        evaluator_process = mp.Process(
            target=evaluator_worker,
            args=(
                i,
                template,
                function_to_evolve,
                function_to_run,
                data_dict,
                sample_queue,
                sample_pending_count,
                result_queue,
                initial_result_queue,
                termination_event,
                perf_queue,
                i == 0 and process_initial  # Only the first evaluator processes the initial program if needed
            )
        )
        evaluator_process.start()
        processes.append(evaluator_process)
    
    # If processing initial program, give the first evaluator a moment to process it
    if process_initial:
        time.sleep(2)
    
    # Start database process
    database_process = mp.Process(
        target=database_worker,
        args=(
            arguments,
            config,
            template,
            function_to_evolve,
            prompt_queue,
            prompt_pending_count,
            result_queue,
            initial_result_queue,
            termination_event,
            perf_queue
        )
    )
    database_process.start()
    processes.append(database_process)
    
    # Start sampler processes
    for i in range(args.num_samplers):
        sampler_process = mp.Process(
            target=sampler_worker,
            args=(
                arguments.samples_per_prompt,
                i,
                arguments.num_evaluators,
                prompt_queue,
                prompt_pending_count,
                sample_queue,
                sample_pending_count,
                termination_event,
                perf_queue
            )
        )
        sampler_process.start()
        processes.append(sampler_process)
    
    try:
        # Wait for all processes to complete
        while not termination_event.is_set():
            time.sleep(5)
        for q in (prompt_queue, sample_queue, result_queue,
                initial_result_queue, perf_queue):
            try:
                q.close()           # 关闭管道
                q.join_thread()     # 等待 FeederThread 退出
            except:
                pass

        for p in processes:
            p.join(10)
            if p.is_alive():
                logger.warning("%s still alive, terminating", p.name)
                p.terminate()
        # If all processes have completed, return from the function
        logger.info("All processes have completed, main_distributed is exiting")
        return
    except KeyboardInterrupt:
        termination_event.set()
        # Give processes a chance to shut down gracefully
        time.sleep(5)
        
        # Check if any processes are still alive
        for process in processes:
            if process.is_alive():
                process.terminate()
        
        # Return from the function after handling KeyboardInterrupt
        logger.info("Terminated due to KeyboardInterrupt, main_distributed is exiting")
        return

def main(args, config: ProgramsDatabaseConfig, specification: str, input: dict, logger: logging.Logger):
    """Launches a physr experiment."""

    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    database = None

    evaluators = evaluator.Evaluator(
        template,
        function_to_evolve,
        function_to_run,
        input
    )
    
    # Check if we're resuming from a checkpoint
    if args.resume_from_ckpt:
        try:
            # Initialize database from loaded state
            database = checkpoint_util.load_checkpoint(args.resume_from_ckpt)
            logger.info(f"Database initialized from checkpoint with {database._global_sample_nums} samples")
            
        except Exception as e:
            logger.error(f"Failed to initialize database from checkpoint: {e}")
            logger.info("Initializing new database instead")

    if database is None:
        # Initialize new database
        database = programs_database.ProgramsDatabase(
            config, template, function_to_evolve, args.log_path
        )
        
        # We send the initial implementation to be analysed by one of the evaluators.
        initial = template.get_function(function_to_evolve).body
        eval_result = evaluators.analyse(initial, island_id=None, version_generated=None)
        database.register_program(eval_result["function"], eval_result["island_id"], eval_result["result_per_test"], evaluate_time=eval_result["evaluate_time"])

    llm = sampler.LLM(args.samples_per_prompt)

    last_checkpoint_time = time.time()

    try:
        while True:
            prompt = database.get_prompt()
            # logger.debug(f"Got prompt for island {prompt.island_id}, version {prompt.version_generated}, prompt: \n{prompt.code}")

            reset_time = time.time()
            all_samples_info = llm.draw_samples(prompt.code)
            sample_time = (time.time() - reset_time) / args.samples_per_prompt
            # This loop can be executed in parallel on remote evaluator machines.
            for sample_info in all_samples_info:
                try:
                    eval_result = evaluators.analyse(
                        sample_info[0], prompt.island_id, prompt.version_generated, sample_time, sample_info[1]
                    )
                    if eval_result is not None:
                        database.register_program(eval_result["function"], eval_result["island_id"], eval_result["result_per_test"], sample_time=eval_result["sample_time"], evaluate_time=eval_result["evaluate_time"], sample_token_usage=eval_result["sample_token_usage"])
                    else:
                        logger.warning("Error analyse sample: %s", sample_info[0])
                except Exception as e:
                    logger.warning("Error analyse sample: %s", sample_info[0])

            # Save checkpoint periodically if checkpoint path is specified
            current_time = time.time()
            if args.save_ckpt_dir and (current_time - last_checkpoint_time) > args.save_ckpt_interval:
                checkpoint_path = os.path.join(args.save_ckpt_dir, f"checkpoint_{database._global_sample_nums}.pkl")
                checkpoint_util.save_checkpoint(database, checkpoint_path)
                last_checkpoint_time = current_time
                logger.info(f"Checkpoint saved to {checkpoint_path}")
                
            if database._global_sample_nums >= args.max_samples:
                break
    finally:
        # Save final checkpoint on shutdown if checkpoint path is specified
        if args.save_ckpt_dir:
            checkpoint_path = os.path.join(args.save_ckpt_dir, f"checkpoint_final.pkl")
            checkpoint_util.save_checkpoint(database, checkpoint_path)
            logger.info(f"Final checkpoint saved to {checkpoint_path}")
        database._profiler.write_best_program_per_c_file()
        logger.info("Best program per c file written")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resume_from_ckpt", type=str, help="Path to checkpoint file to resume from", default=None)
    parser.add_argument("--max_samples", type=int, default=3600)
    parser.add_argument("--spec_path", type=str, default=None)
    parser.add_argument("--problem_name", type=str, default="oscillator1")
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--log_folder", type=str)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--save_ckpt_dir", type=str, help="Directory to save checkpoints to", default=None)
    parser.add_argument("--distributed", action="store_true", help="Run in distributed mode")
    parser.add_argument("--no-distributed", dest="distributed", action="store_false", help="Run in non-distributed mode")
    parser.set_defaults(distributed=True)
    parser.add_argument("--num_samplers", type=int, default=8)
    parser.add_argument("--num_evaluators", type=int, default=8)
    parser.add_argument("--samples_per_prompt", type=int, default=1)
    parser.add_argument("--save_ckpt_interval", type=int, help="Interval (in seconds) between checkpoint saves", default=300)
    args = parser.parse_args()


    # Initialize the main logger
    logger = setup_console_logger("main_distributed")

    if args.resume_from_ckpt:
        logger.info(f"Resuming from checkpoint: {args.resume_from_ckpt}")
        checkpoint_util.load_config(args, logger)
        with open(os.path.join(args.resume_from_ckpt, "database_config.json"), "r") as f:
            data = json.load(f)
        dbconfig = ProgramsDatabaseConfig(**data)
    else:
        dbconfig = ProgramsDatabaseConfig()

    if args.save_ckpt_dir is None:
        args.save_ckpt_dir = "./log/" + args.log_folder + "/checkpoints"

    if args.log_path is None:
        args.log_path = "./log/" + args.log_folder

    if args.save_ckpt_dir:
        logger.info(f"Saving config to {args.save_ckpt_dir}")
        checkpoint_util.save_config(args, args.save_ckpt_dir)
        with open(os.path.join(args.save_ckpt_dir, "database_config.json"), "w") as f:
            json.dump(asdict(dbconfig), f, indent=2)
    # Set multiprocessing start method to 'spawn' for better cross-platform compatibility
    mp.set_start_method('spawn')

    spec, data_dict = load_data(args.spec_path, args.data_folder)
    
    # Run the distributed pipeline
    if args.distributed==True:
        print("Distributed mode")
        main_distributed(args, dbconfig, spec, data_dict, logger)
    else:
        print("Non-distributed mode")
        main(args, dbconfig, spec, data_dict, logger)

