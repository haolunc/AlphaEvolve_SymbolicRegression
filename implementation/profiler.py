# profile the experiment with tensorboard

from __future__ import annotations

import os.path
from typing import List, Dict
import json
import code_manipulation
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Profiler:
    def __init__(
        self,
        num_islands: int,
        log_dir: str,
        max_log_nums: int | None = None,
        log_frequency: int = 100,  # Default logging frequency: every 10 steps
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
            log_frequency: log to tensorboard every log_frequency steps.
        """
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, "samples")
        os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums

        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None

        # the best expression at each complexity
        self._best_score_per_c: Dict[int, float] = {}
        self._best_progstr_per_c: Dict[int, str] = {}
        self._best_progorder_per_c: Dict[int, int] = {}

        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._tot_token_cost = 0

        self._log_frequency = log_frequency

        # # per island info
        # self._best_score_per_island: list[float] = [-float("inf")] * num_islands
        # self._num_programs_per_island: list[int] = [0] * num_islands

        self._writer: SummaryWriter | None = None
        self._init_writer()

    def _init_writer(self):
        self._writer = SummaryWriter(log_dir=self._log_dir)

    
    def __getstate__(self):
        state = self.__dict__.copy()
        # 删除不能被序列化的 writer
        state['_writer'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_writer()

    def _write_tensorboard(self, cur_sample_score: float):
            
        self._writer.add_scalar(
            "Best Score of Function",
            self._cur_best_program_score,
            global_step=self._num_samples,
        )

        if self._num_samples % self._log_frequency == 0:
            # ② 分复杂度曲线：一根曲线一个 complexity
            # scalars = {f"C={c}": s for c, s in self._best_score_per_c.items()}
            # self._writer.add_scalars("Best Score / Complexity", scalars, global_step=self._num_samples)
            
            # Group complexities by fives and select the best score from each group
            grouped_best_scores = {}
            for c, s in self._best_score_per_c.items():
                group_key = c // 5
                group_name = f"C={group_key*5}-{group_key*5+4}"
                if group_name not in grouped_best_scores or s > grouped_best_scores[group_name]:
                    grouped_best_scores[group_name] = s
            
            # Add grouped scalars to tensorboard
            if grouped_best_scores:
                self._writer.add_scalars("Best Score / Complexity Group (by 5)", grouped_best_scores, global_step=self._num_samples)
            
            fig, ax = plt.subplots()
            ax.scatter(list(self._best_score_per_c.keys()), list(self._best_score_per_c.values()))
            ax.set_xlabel("Complexity")
            ax.set_ylabel("Best score so far")
            ax.set_title("Pareto front (step {})".format(self._num_samples))
            self._writer.add_figure("Score_vs_Complexity", fig, global_step=self._num_samples)
            plt.close(fig)

            table_lines = ["|Complexity|Score|Program|", "|---|---|---|"]
            for c, s in sorted(self._best_score_per_c.items()):
                progorder = self._best_progorder_per_c[c]
                table_lines.append(f"|{c}|{s:.4g}|{progorder}|")
            self._writer.add_text("Best Program by Complexity", "\n".join(table_lines), global_step=self._num_samples)

        self._writer.add_scalar(
            "Total Token Cost",
            self._tot_token_cost,
            global_step=self._num_samples,
        )


        self._writer.add_scalars(
            "Legal/Illegal Function",
            {
                "legal function num": self._evaluate_success_program_num,
                "illegal function num": self._evaluate_failed_program_num,
            },
            global_step=self._num_samples,
        )

        self._writer.add_scalars(
            "Total Sample/Evaluate Time",
            {
                "sample time": self._tot_sample_time,
                "evaluate time": self._tot_evaluate_time,
            },
            global_step=self._num_samples,
        )

        # Log the function_str
        if self._cur_best_program_sample_order == self._num_samples:
            self._writer.add_text(
                "Best Function String",
                self._cur_best_program_str,
                global_step=self._num_samples,
            )

        # if cur_sample_score is not None:
        #     self._writer.add_scalar(
        #         "Sample Score",
        #         cur_sample_score,
        #         global_step=self._num_samples,
        #     )

        # # Log best score per island
        # scores_dict = {f"Island {i}": score for i, score in enumerate(self._best_score_per_island)}
        # self._writer.add_scalars(
        #     "Best Score Per Island",
        #     scores_dict,
        #     global_step=self._num_samples,
        # )

        # # Log number of programs per island
        # nums_dict = {f"Island {i}": num for i, num in enumerate(self._num_programs_per_island)}
        # self._writer.add_scalars(
        #     "Number of Programs Per Island",
        #     nums_dict,
        #     global_step=self._num_samples,
        # )

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders > self._num_samples:
            self._num_samples += 1
            # for island_id in register_island_id:
            #     self._num_programs_per_island[island_id] += 1
            self._write_json_and_verbose(programs)
            self._write_tensorboard(programs.score)
        # for island_id in register_island_id:
        #     if programs.score > self._best_score_per_island[island_id]:
        #         self._best_score_per_island[island_id] = programs.score


    def _write_json_and_verbose(self, programs: code_manipulation.Function):
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        sample_time = programs.sample_time
        evaluate_time = programs.evaluate_time
        token_usage = programs.token_usage
        token_cost = programs.token_cost
        optimized_params = programs.optimized_params
        function_str = str(programs)
        score = programs.score
        complexity = programs.complexity
        complexity_detail = programs.complexity_detail
        # log attributes of the function
        print(f"================= Evaluated Function =================")
        print(f"{function_str}")
        print(f"------------------------------------------------------")
        print(f"Score        : {str(score)}")
        print(f"Complexity   : {str(complexity)}")
        print(f"Optimized params: {str(optimized_params)}")
        print(f"Sample time  : {str(sample_time)}")
        print(f"Evaluate time: {str(evaluate_time)}")
        print(f"Token usage  : {str(token_usage)}")
        print(f"Token cost   : {str(token_cost)}")
        print(f"Sample orders: {str(sample_order)}")
        print(f"======================================================\n\n")
        content = {
            "sample_order": sample_order,
            "score": score,
            "optimized_params": optimized_params.tolist() if optimized_params is not None else None,
            "complexity": complexity,
            "complexity_detail": complexity_detail,
            "sample_time": sample_time,
            "evaluate_time": evaluate_time,
            "token_usage": token_usage,
            "token_cost": token_cost,
            "function": function_str
        }
        path = os.path.join(self._json_dir, f"samples_{sample_order}.json")
        with open(path, "w") as json_file:
            json.dump(content, json_file, indent=4)

        function_path = os.path.join(self._json_dir, f"equation_{sample_order}.py")
        programs.save_to_file(function_path)

        # update best function in curve
        if score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_order
            self._cur_best_program_str = function_str

        # update per complexity info
        if complexity is not None:
            if complexity not in self._best_score_per_c:
                self._best_score_per_c[complexity] = score
                self._best_progstr_per_c[complexity] = function_str 
                self._best_progorder_per_c[complexity] = sample_order
            elif score > self._best_score_per_c[complexity]:
                self._best_score_per_c[complexity] = score
                self._best_progstr_per_c[complexity] = function_str
                self._best_progorder_per_c[complexity] = sample_order

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1
        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
        if token_cost:
            self._tot_token_cost += token_cost

    def write_best_program_per_c_file(self):
        """Writes the best program found for each complexity to a text file."""
        output_path = os.path.join(self._log_dir, "best_programs_per_complexity.txt")
        with open(output_path, "w") as f:
            for c in sorted(self._best_score_per_c.keys()):
                score = self._best_score_per_c[c]
                order = self._best_progorder_per_c[c]
                prog_str = self._best_progstr_per_c[c]
                f.write(f"{c},{score:.4g},{order}\n{prog_str}\n") # Use comma separation
        print(f"Best programs per complexity saved to {output_path}")