from multiprocessing import Queue, Event, Value
import checkpoint_util
import programs_database
import sampler
import evaluator
from logging_utils import setup_logger, setup_console_logger
from config import ProgramsDatabaseConfig
import multiprocessing as mp
import os
import time

log_dir = "./logger"


def database_worker(
    args,
    config: ProgramsDatabaseConfig,
    template: str,
    function_to_evolve: str,
    prompt_queue: Queue,
    prompt_pending_count,
    result_queue: Queue,
    initial_result_queue: Queue,
    termination_event,
    perf_queue: Queue,
):
    """Worker function for the database process."""
    logger = setup_console_logger(f"database_worker")
    logger.info(f"Database worker started (PID: {os.getpid()})")
    
    # Initialize the database
    database = None
    
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
        
        logger.info("Waiting for initial program evaluation...")
        eval_result = initial_result_queue.get()
        database.register_program(
            eval_result["function"], 
            eval_result["island_id"], 
            eval_result["result_per_test"],
            sample_time=eval_result["sample_time"], 
            evaluate_time=eval_result["evaluate_time"]
        )
        logger.info("Initial program registered")
   
    # Stats for monitoring
    prompts_generated = 0
    results_processed = 0
    start_time = time.time()
    last_checkpoint_time = time.time()
    show_prompt = False
    # Main database loop
    try:
        while not termination_event.is_set():
            # Generate and put prompts in the queue
            # The database should keep the prompt queue filled but not overfilled
            while prompt_pending_count.value < args.num_samplers and not termination_event.is_set():
                prompt = database.get_prompt()
                if not show_prompt:
                    print(prompt.code)
                    show_prompt = True
                prompt_queue.put(prompt)
                with prompt_pending_count.get_lock():
                    prompt_pending_count.value += 1
                prompts_generated += 1
                
            # Process results as they come in
            try:
                # Non-blocking get with timeout to allow checking termination_event
                eval_result = result_queue.get(timeout=0.1)
                database.register_program(
                    eval_result["function"],
                    eval_result["island_id"],
                    eval_result["result_per_test"],
                    eval_result["sample_time"],
                    eval_result["evaluate_time"],
                    eval_result["sample_token_usage"]
                )
                results_processed += 1
                
                # Check if we've reached the maximum number of samples
                if database._global_sample_nums >= args.max_samples:
                    logger.info(f"Reached max samples ({args.max_samples}), setting termination event")
                    termination_event.set()
                    
            except mp.queues.Empty:
                # No results available at the moment, continue
                pass
            except Exception as e:
                logger.error(f"Database worker error: {e}")
                
            # Save checkpoint periodically if checkpoint path is specified
            current_time = time.time()
            if args.save_ckpt_dir and (current_time - last_checkpoint_time) > args.save_ckpt_interval:
                checkpoint_path = os.path.join(args.save_ckpt_dir, f"checkpoint_{database._global_sample_nums}.pkl")
                checkpoint_util.save_checkpoint(database, checkpoint_path)
                last_checkpoint_time = current_time
                logger.info(f"Checkpoint saved to {checkpoint_path}")
                
            # Report performance stats to monitoring process
            if time.time() - start_time > 150:  # Report stats every 60 seconds
                logger.info(f"prompt_pending: {prompt_pending_count.value}")
                perf_queue.put({
                    "worker_type": "database",
                    "worker_id": 0,
                    "stats": {
                        "prompts_generated": prompts_generated,
                        "results_processed": results_processed,
                        "global_sample_nums": database._global_sample_nums,
                    }
                })
                start_time = time.time()
                
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down database worker")
    finally:
        # Save final checkpoint on shutdown if checkpoint path is specified
        if args.save_ckpt_dir:
            checkpoint_path = os.path.join(args.save_ckpt_dir, f"checkpoint_final.pkl")
            checkpoint_util.save_checkpoint(database, checkpoint_path)
            logger.info(f"Final checkpoint saved to {checkpoint_path}")
        database._profiler.write_best_program_per_c_file()
        logger.info("Best program per c file written")
        logger.info("Database worker shutting down")
        for q in [prompt_queue, result_queue, initial_result_queue, perf_queue]:
            try:
                q.close()
                q.join_thread() 
            except:
                pass


def sampler_worker(
    samples_per_prompt: int,
    worker_id: int,
    num_evaluators: int,
    prompt_queue: Queue,
    prompt_pending_count,
    sample_queue: Queue,
    sample_pending_count,
    termination_event,
    perf_queue: Queue,
):
    """Worker function for sampler processes."""
    logger = setup_console_logger(f"sampler_worker_{worker_id}")
    logger.info(f"Sampler worker {worker_id} started (PID: {os.getpid()})")
    
    # Initialize LLM sampler
    llm = sampler.LLM(samples_per_prompt)
    
    # Stats for monitoring
    prompts_processed = 0
    samples_generated = 0
    start_time = time.time()
    
    network_error_count = 0

    try:
        while not termination_event.is_set():
            if sample_pending_count.value >= num_evaluators:
                logger.info(f"Sampler worker {worker_id} has {sample_pending_count.value} samples in the queue, waiting for them to be processed")
                time.sleep(3)
                continue
            try:
                # Get a prompt from the queue (with timeout to check termination)
                prompt = prompt_queue.get(timeout=0.5)
                with prompt_pending_count.get_lock():
                    prompt_pending_count.value -= 1
                # Generate samples
                reset_time = time.time()
                all_samples_info = llm.draw_samples(prompt.code)
                sample_time = (time.time() - reset_time) / samples_per_prompt

                if all_samples_info is None:
                    network_error_count += 1
                    logger.warning(f"Sampler worker {worker_id} encountered network error {network_error_count} times")

                    if network_error_count > 10:
                        logger.warning(f"Sampler worker {worker_id} encountered too many network errors, shutting down")
                        termination_event.set()

                    continue

                
                # Put samples in the sample queue
                for sample_info in all_samples_info:
                    if sample_info:  # Ensure sample is not None
                        sample_info = {
                            "sample": sample_info["response_text"],
                            "island_id": prompt.island_id,
                            "version_generated": prompt.version_generated,
                            "sample_time": sample_time,
                            "sample_token_usage": (sample_info["input_tokens"], sample_info["output_tokens"])
                        }
                        sample_queue.put(sample_info)
                        samples_generated += 1
                        with sample_pending_count.get_lock():
                            sample_pending_count.value += 1
                
                prompts_processed += 1
                
                # Report performance stats
                if time.time() - start_time > 150:  # Report stats every 20 seconds
                    perf_queue.put({
                        "worker_type": "sampler",
                        "worker_id": worker_id,
                        "stats": {
                            "prompts_processed": prompts_processed,
                            "samples_generated": samples_generated,
                        }
                    })
                    start_time = time.time()
                    
            except mp.queues.Empty:
                # No prompts available at the moment, continue
                pass
            except Exception as e:
                logger.warning(f"Sampler worker {worker_id} error: {e}")
                
    except KeyboardInterrupt:
        logger.info(f"KeyboardInterrupt received, shutting down sampler worker {worker_id}")
    except Exception as e:
        logger.error(f"Sampler worker {worker_id} error: {e}")
    finally:
        llm.clean()  # Clean up LLM resources
        for q in [prompt_queue, sample_queue, perf_queue]:
            try:
                q.close()
                q.join_thread() 
            except:
                pass
        logger.info(f"Sampler worker {worker_id} shutting down")

def evaluator_worker(
    worker_id: int,
    template: str,
    function_to_evolve: str,
    function_to_run: str,
    data_dict: dict,
    sample_queue: Queue,
    sample_pending_count,
    result_queue: Queue,
    initial_result_queue: Queue,
    termination_event,
    perf_queue: Queue,
    process_initial: bool = False
):
    """Worker function for evaluator processes."""
    logger = setup_console_logger(f"evaluator_worker_{worker_id}")
    logger.info(f"Evaluator worker {worker_id} started (PID: {os.getpid()})")
    
    # Initialize evaluator
    eval_instance = evaluator.Evaluator(
        template,
        function_to_evolve,
        function_to_run,
        data_dict
    )
    
    # Process the initial program if this is the first evaluator
    if process_initial:
        logger.info("Processing initial program")
        initial = template.get_function(function_to_evolve).body
        eval_result = eval_instance.analyse(initial, island_id=None, version_generated=None)
        initial_result_queue.put(eval_result)
        logger.info("Initial program evaluation complete")
    
    # Stats for monitoring
    samples_processed = 0
    successful_evaluations = 0
    failed_evaluations = 0
    start_time = time.time()
    
    try:
        while not termination_event.is_set():
            try:
                # Get a sample from the queue (with timeout to check termination)
                sample_info = sample_queue.get(timeout=0.5)
                with sample_pending_count.get_lock():
                    sample_pending_count.value -= 1
                # Evaluate the sample
                eval_result = eval_instance.analyse(
                    sample_info["sample"],
                    sample_info["island_id"],
                    sample_info["version_generated"],
                    sample_info["sample_time"],
                    sample_info["sample_token_usage"]
                )
                if eval_result is not None:
                    result_queue.put(eval_result)
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
                
                samples_processed += 1
                
                # Report performance stats
                if time.time() - start_time > 150:  # Report stats every 20 seconds
                    perf_queue.put({
                        "worker_type": "evaluator",
                        "worker_id": worker_id,
                        "stats": {
                            "samples_processed": samples_processed,
                            "successful_evaluations": successful_evaluations,
                            "failed_evaluations": failed_evaluations,
                        }
                    })
                    start_time = time.time()
                    
            except mp.queues.Empty:
                # No samples available at the moment, continue
                pass
                
    except KeyboardInterrupt:
        logger.info(f"KeyboardInterrupt received, shutting down evaluator worker {worker_id}")
    except Exception as e:
        logger.error(f"Evaluator worker {worker_id} error: {e}")
    finally:
        for q in [sample_queue, result_queue, initial_result_queue, perf_queue]:
            try:
                q.close()
                q.join_thread() 
            except:
                pass
        logger.info(f"Evaluator worker {worker_id} shutting down")

def monitoring_worker(
    arguments,
    perf_queue: Queue,
    termination_event,
    monitor_interval: int = 300
):
    """Worker function for monitoring process performance."""
    logger = setup_logger("monitor", log_dir)
    logger.info(f"Monitoring worker started (PID: {os.getpid()})")
    
    # Initialize performance tracking
    performance_data = {
        "database": {"worker_id": 0, "stats": {}},
        "sampler": {i: {"stats": {}} for i in range(arguments.num_samplers)},
        "evaluator": {i: {"stats": {}} for i in range(arguments.num_evaluators)}
    }
    
    last_report_time = time.time()
    
    try:
        while not termination_event.is_set():
            # Collect performance data from the queue
            try:
                while True:  # Process all available performance data
                    perf_data = perf_queue.get_nowait()
                    worker_type = perf_data["worker_type"]
                    worker_id = perf_data["worker_id"]
                    stats = perf_data["stats"]
                    
                    if worker_type == "database":
                        performance_data["database"]["stats"] = stats
                    else:
                        performance_data[worker_type][worker_id]["stats"] = stats
            except mp.queues.Empty:
                # No more performance data to process at the moment
                pass
            
            # Report performance data at the specified interval
            current_time = time.time()
            if current_time - last_report_time >= monitor_interval:
                logger.info("===== Performance Report =====")
                
                # Report database stats
                db_stats = performance_data["database"]["stats"]
                if db_stats:
                    logger.info(f"Database: Prompts generated: {db_stats.get('prompts_generated', 0)}, "
                                f"Results processed: {db_stats.get('results_processed', 0)}, "
                                f"Total samples: {db_stats.get('global_sample_nums', 0)}")
                
                # Report sampler stats
                logger.info("Samplers:")
                for sampler_id, sampler_data in performance_data["sampler"].items():
                    stats = sampler_data["stats"]
                    if stats:
                        logger.info(f"  Sampler {sampler_id}: Prompts processed: {stats.get('prompts_processed', 0)}, "
                                    f"Samples generated: {stats.get('samples_generated', 0)}")
                
                # Report evaluator stats
                logger.info("Evaluators:")
                for eval_id, eval_data in performance_data["evaluator"].items():
                    stats = eval_data["stats"]
                    if stats:
                        logger.info(f"  Evaluator {eval_id}: Samples processed: {stats.get('samples_processed', 0)}, "
                                    f"Success rate: {stats.get('successful_evaluations', 0)}/{stats.get('samples_processed', 0)}")
                
                logger.info("==============================")
                last_report_time = current_time
            
            # Sleep briefly to prevent busy waiting
            time.sleep(10)
                
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down monitoring worker")
    except Exception as e:
        logger.error(f"Monitoring worker error: {e}")
    finally:
        for q in [perf_queue]:
            try:
                q.close()
            except:
                pass
        logger.info("Monitoring worker shutting down")
