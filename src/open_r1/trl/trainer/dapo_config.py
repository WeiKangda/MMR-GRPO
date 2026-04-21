# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class DAPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`DAPOTrainer`].

    DAPO (Dynamic Adaptive Policy Optimization) extends GRPO with dynamic sampling
    that filters trajectories based on reward variance to improve training efficiency.

    Only the parameters specific to DAPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`DAPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation.
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM.

        > DAPO-specific parameters

        enable_dynamic_sampling (`bool`, *optional*, defaults to `True`):
            Whether to enable DAPO's dynamic sampling mechanism. When enabled, filters out prompts with zero
            reward variance (all generations have same reward), keeping only diverse reward groups.
        filter_metric (`str`, *optional*, defaults to `"seq_final_reward"`):
            Metric to use for filtering in dynamic sampling. Options: "seq_final_reward", "seq_reward".
            seq_final_reward uses the sum of token-level rewards (after KL penalty if enabled).
            seq_reward uses the sum of token-level scores (before KL penalty).
        max_num_gen_batches (`int`, *optional*, defaults to `5`):
            Maximum number of generation batches to attempt when dynamic sampling is enabled.
            If set to 0 or negative, unlimited generation batches are allowed until batch is filled.
            Prevents infinite loops when data is too difficult.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer.
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.9`):
            α parameter from the TR-DPO paper for reference model mixing.
        ref_model_sync_steps (`int`, *optional*, defaults to `64`):
            τ parameter from the TR-DPO paper for reference model sync frequency.

        > Parameters for diversity-aware reward reweighting (MMR)

        SMI_reweighting (`bool`, *optional*, defaults to `False`):
            Use SMI reweighting.
        MMR_reweighting (`bool`, *optional*, defaults to `False`):
            Use MMR (Maximal Marginal Relevance) reweighting for diversity-aware reward adjustment.
        MMR_STD_reweighting (`bool`, *optional*, defaults to `False`):
            Use MMR with adaptive lambda based on reward std.
        MMR_SIGMOID_reweighting (`bool`, *optional*, defaults to `False`):
            Use MMR with adaptive lambda based on sigmoid(std/mean).
        lambda_div (`float`, *optional*, defaults to `0.7`):
            Diversity weight for MMR reweighting. 1.0 = pure reward, 0.0 = pure diversity.
        mmr_std_temp (`float`, *optional*, defaults to `1.0`):
            Temperature for adaptive lambda in MMR_STD reweighting.
        extractor_name (`str`, *optional*, defaults to `'jina'`):
            Embedding model to use for computing completion similarity. Options: 'jina', 'nomic'.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log the completions during training.
        log_reward_reweighting (`bool`, *optional*, defaults to `False`):
            Whether to log rewards before and after reweighting for analysis.
        log_reward_plots (`bool`, *optional*, defaults to `False`):
            Whether to generate and log reward visualization plots.
        reward_plot_frequency (`int`, *optional*, defaults to `10`):
            Frequency (in steps) for logging reward plots.
        save_reward_data (`bool`, *optional*, defaults to `False`):
            Whether to save raw reward data to CSV files.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `DAPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM."
        },
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size."
        },
    )

    # DAPO-specific parameters
    enable_dynamic_sampling: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable DAPO's dynamic sampling mechanism. When enabled, filters out prompts with "
            "zero reward variance, keeping only diverse reward groups for more efficient training."
        },
    )
    filter_metric: str = field(
        default="seq_final_reward",
        metadata={
            "help": "Metric to use for filtering in dynamic sampling. Options: 'seq_final_reward' (sum of "
            "token-level rewards after KL penalty), 'seq_reward' (sum of token-level scores before KL penalty)."
        },
    )
    max_num_gen_batches: int = field(
        default=5,
        metadata={
            "help": "Maximum number of generation batches to attempt when dynamic sampling is enabled. "
            "If <= 0, unlimited generation batches are allowed until batch is filled. "
            "Prevents infinite loops when data is too difficult."
        },
    )
    filter_reward_index: int = field(
        default=0,
        metadata={
            "help": "Index of reward function to use for dynamic sampling filtering. "
            "Default 0 assumes accuracy_reward is the first reward function. "
            "Set to -1 to use the combined weighted reward (original behavior). "
            "This allows filtering on accuracy while training with multiple weighted rewards."
        },
    )
    epsilon_low: float = field(
        default=0.2,
        metadata={
            "help": "Lower bound for PPO clipping. DAPO uses asymmetric clipping with epsilon_low=0.2."
        },
    )
    epsilon_high: float = field(
        default=0.28,
        metadata={
            "help": "Upper bound for PPO clipping (clip-higher strategy). DAPO uses epsilon_high=0.28 to "
            "prevent entropy collapse while allowing exploration."
        },
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates."
        },
    )
    ref_model_sync_steps: int = field(
        default=64,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy."
        },
    )

    # Parameters for diversity-aware reward reweighting
    SMI_reweighting: bool = field(
        default=False,
        metadata={
            "help": "use SMI rewighting."
        },
    )
    MMR_reweighting: bool = field(
        default=False,
        metadata={
            "help": "use MMR (Maximal Marginal Relevance) reweighting for diversity-aware reward adjustment."
        },
    )
    MMR_STD_reweighting: bool = field(
        default=False,
        metadata={
            "help": "use MMR with adaptive lambda based on reward std for diversity-aware reward adjustment."
        },
    )
    MMR_SIGMOID_reweighting: bool = field(
        default=False,
        metadata={
            "help": "use MMR with adaptive lambda based on sigmoid(std/mean) for diversity-aware reward adjustment."
        },
    )
    lambda_div: float = field(
        default=0.7,
        metadata={
            "help": "Diversity weight for MMR reweighting. 1.0 = pure reward, 0.0 = pure diversity."
        },
    )
    mmr_std_temp: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for adaptive lambda in MMR_STD reweighting. Higher temp = more sensitive to reward std."
        },
    )
    extractor_name: str = field(
        default='jina',
        metadata={
            "help": "use compeletion embedding modle."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to log the completions during training."},
    )
    log_reward_reweighting: bool = field(
        default=False,
        metadata={
            "help": "Whether to log rewards before and after reweighting for analysis. "
                    "Logs are saved to WANDB_DIR in offline mode and to output_dir."
        },
    )
    log_reward_plots: bool = field(
        default=False,
        metadata={
            "help": "Whether to generate and log reward visualization plots. "
                    "Plots saved to: (1) output_dir/reward_plots/, (2) wandb offline logs in WANDB_DIR."
        },
    )
    reward_plot_frequency: int = field(
        default=10,
        metadata={
            "help": "Frequency (in steps) for logging reward plots. Only applies if log_reward_plots is True."
        },
    )
    save_reward_data: bool = field(
        default=False,
        metadata={
            "help": "Whether to save raw reward data to CSV files in output_dir/reward_data/ for offline analysis."
        },
    )
