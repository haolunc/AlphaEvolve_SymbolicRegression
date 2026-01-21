# Copyright 2023 DeepMind Technologies Limited
# Copyright 2026 Haolun Cai
# This file has been modified by Haolun Cai for AlphaEvolve_SymbolicRegression on Jan 21, 2026.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import dataclasses

@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
    """Configuration of a ProgramsDatabase.

    Attributes:
      functions_per_prompt: Number of previous programs to include in prompts.
      num_islands: Number of islands to maintain as a diversity mechanism.
      reset_period: How often (in seconds) the weakest islands should be reset.
      cluster_sampling_temperature_init: Initial temperature for softmax sampling
          of clusters within an island.
      cluster_sampling_temperature_period: Period of linear decay of the cluster
          sampling temperature.
    """

    functions_per_prompt: int = 4
    num_islands: int = 10
    reset_period: int = 700
    cluster_sampling_temperature_init: float = 0.005
    cluster_sampling_temperature_period: int = 200
    cost_per_ktoken: tuple[float, float] = (0.006, 0.024)
    # RMB
    # qwen-max: 0.0024, 0.0096
    # qwen-plus: 0.0008, 0.002
    # qwen-turbo: 0.0003, 0.0006
    # deepseek-chat: 0.002, 0.008
    # deepseek-reasoner: 0.004, 0.016
    # Dollar
    # gpt5-mini(0.00025, 0.002)
    # gemini-3-pro: (0.002, 0.012)