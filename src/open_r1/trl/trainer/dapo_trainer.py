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

import logging
import os
import textwrap
import uuid
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
from .dapo_config import DAPOConfig
from .dapo_dataloader import DynamicSamplingDataLoader
from .grpo_trainer import RepeatRandomSampler, diverse_adjust_rewards_fast, diverse_adjust_rewards_fast_sigmoid, diverse_adjust_rewards_fast_std
from .utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

# Import reward visualization utilities
from src.open_r1.utils.reward_visualization import (
    create_advantage_plot,
    create_reward_comparison_plot,
    log_advantage_metrics,
    log_reward_reweighting_metrics,
    save_reward_data_to_csv,
)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# Global EMA for MMR std reweighting
_lambda_ema = None


class DAPOTrainer(Trainer):
    """
    Trainer for the Dynamic Adaptive Policy Optimization (DAPO) method.

    DAPO extends policy optimization with four key techniques:
    1. Token-Level Policy Gradient Loss: Better handles long chain-of-thought responses
    2. Clip-Higher: Asymmetric clipping (ε_low=0.2, ε_high=0.28) to prevent entropy collapse
    3. Dynamic Sampling: Filters prompts with identical outputs for efficiency
    4. Overlong Reward Shaping: Reduces reward noise in long responses

    Key differences from GRPO:
    - Token-level loss normalization instead of sample-level
    - Asymmetric PPO clipping with higher upper bound
    - Dynamic filtering of non-diverse prompt groups

    Example:

    ```python
    from datasets import load_dataset
    from trl import DAPOTrainer, DAPOConfig

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        return [float(len(set(completion))) for completion in completions]

    config = DAPOConfig(
        enable_dynamic_sampling=True,
        epsilon_low=0.2,
        epsilon_high=0.28,
    )

    trainer = DAPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions for computing rewards.
        args ([`DAPOConfig`], *optional*):
            Configuration for this trainer.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Training dataset with a `"prompt"` column.
        eval_dataset:
            Evaluation dataset.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Tokenizer with left padding.
        reward_processing_classes:
            Processing classes for reward functions.
        callbacks:
            List of callbacks.
        optimizers:
            Optimizer and scheduler tuple.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration.
    """

    _tag_names = ["trl", "dapo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: DAPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DAPOConfig(f"{model_name}-DAPO")

        # Initialize embedding extractor for MMR reweighting
        self.extractor_name = args.extractor_name
        if args.extractor_name == "jina":
            self.sentence_extractor = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
            )
        elif args.extractor_name == "nomic":
            self.sentence_extractor = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
            )
        else:
            raise NotImplementedError("Embedding model is not implemented.")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(f"Invalid `torch_dtype`: {torch_dtype}")

            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` but model is already instantiated."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match "
                    f"number of reward functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("Number of reward processing classes must match number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # MMR reweighting settings
        self.SMI_reweighting = args.SMI_reweighting
        self.MMR_reweighting = args.MMR_reweighting
        self.MMR_STD_reweighting = args.MMR_STD_reweighting
        self.MMR_SIGMOID_reweighting = args.MMR_SIGMOID_reweighting
        self.lambda_div = args.lambda_div
        self.mmr_std_temp = args.mmr_std_temp

        # DAPO-specific settings
        self.enable_dynamic_sampling = args.enable_dynamic_sampling
        self.filter_metric = args.filter_metric
        self.max_num_gen_batches = args.max_num_gen_batches
        self.epsilon_low = getattr(args, 'epsilon_low', 0.2)
        self.epsilon_high = getattr(args, 'epsilon_high', 0.28)

        # Data collator
        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.use_vllm = args.use_vllm
        self.beta = args.beta

        # Suppress token estimation warning
        model.warnings_issued["estimate_tokens"] = True

        # Initialize metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Validate batch sizes
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Valid values: {possible_values}."
            )

        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Valid values: {possible_values}."
                )

        # Set device-specific seed
        set_seed(args.seed, device_specific=True)

        # vLLM setup
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError("vLLM is not available. Please install with `pip install vllm`.")

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"

                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(f"Requested vllm device ({vllm_device}) is not available.")

                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"Device {vllm_device} is also used for training. Consider using a dedicated device for vLLM."
                    )

                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    # NOTE: vLLM will load base model weights during initialization
                    # We override them in _move_model_to_vllm() before each generation
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = -1  # Initialize to -1 to ensure first load
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Set model_accepts_loss_kwargs for gradient accumulation scaling
        self.model_accepts_loss_kwargs = False

        # Add tags
        self.model.add_model_tags(self._tag_names)

        # Prepare reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        # Prepare reward models
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return RepeatRandomSampler(dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)

    def _move_model_to_vllm(self):
        print(f"[DEBUG _move_model_to_vllm] Called at global_step={self.state.global_step}")
        print(f"[DEBUG _move_model_to_vllm] Last loaded step: {self._last_loaded_step}")

        # 🔍 DIAGNOSTIC: Check if this is a resume scenario
        is_resume = self.state.global_step > 0
        if is_resume:
            print(f"[⚠️  RESUME DETECTED] Moving model to vLLM after checkpoint resume at step {self.state.global_step}")

        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                print(f"[DEBUG _move_model_to_vllm] Model is PEFT, merging adapter")
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                print(f"[DEBUG _move_model_to_vllm] State dict has {len(state_dict)} keys")
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
                print(f"[DEBUG _move_model_to_vllm] After filtering: {len(state_dict)} keys")
            else:
                state_dict = unwrapped_model.state_dict()
                print(f"[DEBUG _move_model_to_vllm] Non-PEFT model, got {len(state_dict)} weights from state_dict")

                # DIAGNOSTIC: Show RAW keys with quotes to see exact format
                print(f"\n[DEBUG] RAW state_dict keys (first 15):")
                for i, key in enumerate(list(state_dict.keys())[:15]):
                    print(f"  [{i}] '{key}'")

                # DIAGNOSTIC: Check for problematic prefixes
                double_model_keys = [k for k in state_dict.keys() if k.startswith("model.model.")]
                module_model_keys = [k for k in state_dict.keys() if k.startswith("module.model.")]
                if double_model_keys:
                    print(f"[ERROR] Found {len(double_model_keys)} keys with 'model.model.' prefix!")
                    print(f"  Examples: {double_model_keys[:3]}")
                if module_model_keys:
                    print(f"[ERROR] Found {len(module_model_keys)} keys with 'module.model.' prefix!")
                    print(f"  Examples: {module_model_keys[:3]}")

                # 🔧 FIX: Normalize state_dict keys to handle checkpoint resume
                # vLLM expects keys with "model." prefix (e.g., "model.embed_tokens.weight")
                # BUT: We must avoid creating double prefixes like "model.model." during resume!

                normalized_state_dict = {}
                problem_keys = []

                for k, v in state_dict.items():
                    original_k = k
                    new_k = k

                    # STEP 1: Strip ALL wrapper prefixes first
                    prefixes_to_remove = ["module.", "_orig_mod.", "model."]
                    for prefix in prefixes_to_remove:
                        if new_k.startswith(prefix):
                            new_k = new_k[len(prefix):]
                            print(f"[DEBUG normalization] Stripped '{prefix}' from '{original_k}' -> '{new_k}'")

                    # STEP 2: Now add back exactly ONE "model." prefix for vLLM
                    # Exception: lm_head should NOT have "model." prefix in vLLM
                    if "lm_head" in new_k:
                        # Keep lm_head at top level
                        pass
                    else:
                        # All other keys need "model." prefix
                        new_k = f"model.{new_k}"

                    # 🔍 DIAGNOSTIC: Detect problematic keys
                    if "model.model." in new_k or "module.model." in new_k:
                        problem_keys.append(new_k)

                    normalized_state_dict[new_k] = v

                if problem_keys:
                    print(f"[❌ ERROR] Found {len(problem_keys)} keys with double prefixes!")
                    print(f"   Examples: {problem_keys[:5]}")
                    print(f"   This WILL cause model corruption! Check normalization logic.")

                state_dict = normalized_state_dict
                print(f"[DEBUG _move_model_to_vllm] After normalization: {len(state_dict)} keys")

                # Show sample keys after normalization
                sample_keys = list(state_dict.keys())[:3]
                print(f"[DEBUG _move_model_to_vllm] Sample normalized keys: {sample_keys}")
                print(f"[DEBUG _move_model_to_vllm] Final state_dict size: {len(state_dict)} keys")
                print(f"[INFO] vLLM will automatically fuse QKV and gate_up weights during loading")

                # DIAGNOSTIC: Show FINAL keys being sent to vLLM
                print(f"\n[DEBUG] FINAL state_dict keys (first 15):")
                for i, key in enumerate(list(state_dict.keys())[:15]):
                    print(f"  [{i}] '{key}'")

            if self.accelerator.is_main_process:
                print(f"[DEBUG _move_model_to_vllm] Loading {len(state_dict)} weights into vLLM")
                print(f"[DEBUG _move_model_to_vllm] Providing unfused HF format - vLLM will fuse internally")

                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

                # Show sample keys being loaded
                checkpoint_keys = list(state_dict.keys())
                print(f"[DEBUG _move_model_to_vllm] Checkpoint has {len(checkpoint_keys)} keys to load")
                print(f"[DEBUG _move_model_to_vllm] First 10 checkpoint keys: {checkpoint_keys[:10]}")

                # Show a few sample weights
                print(f"[DEBUG _move_model_to_vllm] Sample checkpoint weight stats:")
                for k in checkpoint_keys[:3]:
                    v = state_dict[k]
                    print(f"  {k}: shape={v.shape}, mean={v.float().mean().item():.6f}, std={v.float().std().item():.6f}")

                # Verify vLLM model structure before loading
                print(f"[DEBUG _move_model_to_vllm] vLLM model type: {type(llm_model).__name__}")

                # DIAGNOSTIC: Check a weight BEFORE loading
                try:
                    before_o_proj = llm_model.model.layers[0].self_attn.o_proj.weight.data.clone()
                    before_mean = before_o_proj.float().mean().item()
                    before_std = before_o_proj.float().std().item()
                    before_min = before_o_proj.float().min().item()
                    before_max = before_o_proj.float().max().item()
                    print(f"[DEBUG] BEFORE load - layer 0 o_proj: mean={before_mean:.6f}, std={before_std:.6f}, min={before_min:.6f}, max={before_max:.6f}")
                except Exception as e:
                    print(f"[DEBUG] Could not access before weight: {e}")
                    before_o_proj = None

                # Load weights - vLLM will handle the fusion internally
                print(f"[DEBUG _move_model_to_vllm] Calling load_weights()...")
                try:
                    llm_model.load_weights(state_dict.items())
                    print(f"[DEBUG _move_model_to_vllm] load_weights() completed successfully")

                    # DIAGNOSTIC: Check the same weight AFTER loading
                    if before_o_proj is not None:
                        try:
                            after_o_proj = llm_model.model.layers[0].self_attn.o_proj.weight.data
                            after_mean = after_o_proj.float().mean().item()
                            after_std = after_o_proj.float().std().item()
                            after_min = after_o_proj.float().min().item()
                            after_max = after_o_proj.float().max().item()
                            weights_changed = not torch.equal(before_o_proj, after_o_proj)
                            print(f"[DEBUG] AFTER load - layer 0 o_proj: mean={after_mean:.6f}, std={after_std:.6f}, min={after_min:.6f}, max={after_max:.6f}")
                            print(f"[DEBUG] Weights actually changed: {weights_changed}")

                            # Compare with checkpoint value
                            if 'model.layers.0.self_attn.o_proj.weight' in state_dict:
                                checkpoint_o_proj = state_dict['model.layers.0.self_attn.o_proj.weight']
                                checkpoint_mean = checkpoint_o_proj.float().mean().item()
                                checkpoint_min = checkpoint_o_proj.float().min().item()
                                checkpoint_max = checkpoint_o_proj.float().max().item()
                                weights_match = torch.allclose(checkpoint_o_proj.cpu(), after_o_proj.cpu(), rtol=1e-4)
                                max_diff = (checkpoint_o_proj.cpu() - after_o_proj.cpu()).abs().max().item()
                                print(f"[DEBUG] Checkpoint o_proj: mean={checkpoint_mean:.6f}, min={checkpoint_min:.6f}, max={checkpoint_max:.6f}")
                                print(f"[DEBUG] Loaded weights match checkpoint: {weights_match}")
                                print(f"[DEBUG] Max absolute difference: {max_diff:.10f}")

                            # 🔍 CRITICAL: Check embedding weight specifically
                            try:
                                vllm_embed = llm_model.model.embed_tokens.weight.data
                                checkpoint_embed = state_dict['model.embed_tokens.weight']

                                embed_mean = vllm_embed.float().mean().item()
                                embed_std = vllm_embed.float().std().item()
                                checkpoint_embed_mean = checkpoint_embed.float().mean().item()
                                checkpoint_embed_std = checkpoint_embed.float().std().item()

                                embed_match = torch.allclose(checkpoint_embed.cpu(), vllm_embed.cpu(), rtol=1e-4)
                                embed_diff = (checkpoint_embed.cpu() - vllm_embed.cpu()).abs().max().item()

                                print(f"\n[🔍 EMBEDDING CHECK]")
                                print(f"  Checkpoint embed: mean={checkpoint_embed_mean:.6f}, std={checkpoint_embed_std:.6f}")
                                print(f"  vLLM embed:       mean={embed_mean:.6f}, std={embed_std:.6f}")
                                print(f"  Embeddings match: {embed_match}")
                                print(f"  Max difference: {embed_diff:.10f}")

                                if not embed_match or embed_diff > 1e-5:
                                    print(f"  ⚠️  WARNING: Embedding weights DO NOT match checkpoint!")
                                    print(f"  This will cause garbage generation!")
                            except Exception as e:
                                print(f"[DEBUG] Could not check embedding: {e}")

                        except Exception as e:
                            print(f"[DEBUG] Could not verify after weight: {e}")

                except Exception as e:
                    print(f"[ERROR _move_model_to_vllm] Failed to load weights: {e}")
                    raise
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()
                print(f"[DEBUG _move_model_to_vllm] Adapter unmerged")

    def get_train_dataloader(self):
        """
        Returns the training dataloader, wrapped with dynamic sampling if enabled.
        """
        base_dataloader = super().get_train_dataloader()

        if self.enable_dynamic_sampling:
            logger.info("Wrapping dataloader with dynamic sampling")
            return DynamicSamplingDataLoader(
                base_dataloader,
                trainer=self,
                max_attempts=self.max_num_gen_batches
            )
        else:
            return base_dataloader

    def _generate_and_filter_prompts(self, batch_prompts):
        """
        Generate completions for prompts and filter out zero-variance groups.

        This method is called by DynamicSamplingDataLoader to:
        1. Generate m completions per prompt
        2. Compute rewards for all completions
        3. Filter out prompts where all rewards are identical (zero advantage)
        4. Return only valid prompts with their cached data

        Args:
            batch_prompts: List of prompt dictionaries from dataloader

        Returns:
            List of valid prompt data dictionaries, each containing:
            - 'prompt': Original prompt dict
            - 'cached_completions': Generated completion texts
            - 'cached_rewards': Computed rewards
            - 'cached_prompt_ids': Tokenized prompt IDs
            - 'cached_completion_ids': Tokenized completion IDs
            - etc.
        """
        device = self.accelerator.device

        # Extract prompts
        prompts = [x["prompt"] for x in batch_prompts]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch_prompts]

        # Tokenize prompts
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)  # Use parent class method
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions (m per prompt, where m = num_generations)
        # This generates m completions for EACH prompt in the batch
        # So if batch has n prompts, we get n*m completions total
        if self.args.use_vllm:
            # For vLLM, we need to repeat prompts m times
            prompts_text_repeated = [p for p in prompts_text for _ in range(self.num_generations)]
            all_prompts_text = gather_object(prompts_text_repeated)

            if self.accelerator.is_main_process:
                if self.state.global_step != self._last_loaded_step:
                    self._move_model_to_vllm()
                    self._last_loaded_step = self.state.global_step

                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.outputs[0].token_ids for out in outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)

            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_repeated),
                (self.accelerator.process_index + 1) * len(prompts_text_repeated),
            )
            completion_ids = completion_ids[process_slice]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        else:
            # For regular generation, need to repeat inputs
            prompt_ids_repeated = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_mask_repeated = prompt_mask.repeat_interleave(self.num_generations, dim=0)

            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids_repeated,
                    attention_mask=prompt_mask_repeated,
                    generation_config=self.generation_config
                )

            prompt_length = prompt_ids_repeated.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Decode completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Prepare for reward computation
        if is_conversational(batch_prompts[0]):
            # Repeat prompts for conversational format
            prompts_repeated = [p for p in prompts for _ in range(self.num_generations)]
            completions = []
            for prompt, completion in zip(prompts_repeated, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            prompts_repeated = prompts * self.num_generations
            completions = completions_text

        # Compute rewards for all completions
        rewards_per_func = torch.zeros(len(prompts) * self.num_generations, len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(batch_prompts[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts_repeated, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts_repeated, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = Trainer._prepare_inputs(self, reward_inputs)  # Use parent class method
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in batch_prompts[0] if key not in ["prompt", "completion"]]
                # DEBUG LOGGING (commented out for production)
                # print(f"[DEBUG dapo_trainer] batch_prompts[0] keys: {list(batch_prompts[0].keys())}")
                # print(f"[DEBUG dapo_trainer] Keys extracted for reward_kwargs: {keys}")
                # print(f"[DEBUG dapo_trainer] 'solution' in keys: {'solution' in keys}")

                # Repeat additional kwargs
                reward_kwargs = {key: [example[key] for example in batch_prompts for _ in range(self.num_generations)] for key in keys}

                # DEBUG LOGGING (commented out for production)
                # print(f"[DEBUG dapo_trainer] reward_kwargs keys: {list(reward_kwargs.keys())}")
                # if 'solution' in reward_kwargs:
                #     print(f"[DEBUG dapo_trainer] Number of solutions: {len(reward_kwargs['solution'])}")
                #     if len(reward_kwargs['solution']) > 0:
                #         first_sol = str(reward_kwargs['solution'][0])
                #         print(f"[DEBUG dapo_trainer] First solution: {first_sol[:100]}..." if len(first_sol) > 100 else f"[DEBUG dapo_trainer] First solution: {first_sol}")
                # else:
                #     print(f"[DEBUG dapo_trainer] WARNING: 'solution' NOT in reward_kwargs!")
                #     print(f"[DEBUG dapo_trainer] Available keys: {list(reward_kwargs.keys())}")

                output_reward_func = reward_func(prompts=prompts_repeated, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather and compute final rewards
        rewards_per_func = gather(rewards_per_func)

        # Determine which reward to use for filtering
        filter_reward_index = getattr(self.args, 'filter_reward_index', 0)
        if filter_reward_index >= 0 and filter_reward_index < rewards_per_func.size(1):
            # Use specific reward function for filtering (e.g., accuracy_reward)
            filter_rewards = rewards_per_func[:, filter_reward_index]
            logger.debug(f"Dynamic sampling: Filtering on reward function index {filter_reward_index}")
        else:
            # Use combined weighted reward for filtering (original behavior)
            filter_rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
            logger.debug("Dynamic sampling: Filtering on combined weighted reward")

        # Always compute combined rewards for actual training
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Reshape: (num_prompts, num_generations)
        num_prompts_local = len(batch_prompts)

        # Filter based on filter_rewards variance
        filter_rewards_grouped = filter_rewards.view(-1, self.num_generations)[:num_prompts_local]
        reward_stds = filter_rewards_grouped.std(dim=1)
        valid_mask = reward_stds > 1e-8  # Non-zero variance

        # Log filtering statistics with detailed reward information
        kept_ratio = valid_mask.float().mean().item()
        self._metrics["dapo/kept_prompts_ratio"].append(kept_ratio)
        self._metrics["dapo/avg_reward_std"].append(reward_stds[valid_mask].mean().item() if valid_mask.any() else 0.0)
        self._metrics["dapo/filter_reward_index"].append(filter_reward_index if filter_reward_index >= 0 else -1)

        # Log detailed reward info for each prompt
        logger.info(f"\n{'='*80}\nDynamic Sampling Batch Analysis:")
        for i in range(num_prompts_local):
            start_idx = i * self.num_generations
            end_idx = (i + 1) * self.num_generations
            prompt_rewards = filter_rewards[start_idx:end_idx].cpu().tolist()

            if valid_mask[i]:
                # Non-zero variance - show all rewards
                logger.info(
                    f"  Prompt {i+1}/{num_prompts_local}: KEPT (variance={reward_stds[i].item():.4f}) | "
                    f"Rewards: [{', '.join([f'{r:.3f}' for r in prompt_rewards])}]"
                )
            else:
                # Zero variance - check if all 0 or all 1 (or other constant)
                unique_val = prompt_rewards[0]
                if abs(unique_val - 0.0) < 1e-6:
                    status = "SKIPPED (all rewards = 0.0)"
                elif abs(unique_val - 1.0) < 1e-6:
                    status = "SKIPPED (all rewards = 1.0)"
                else:
                    status = f"SKIPPED (all rewards = {unique_val:.3f})"
                logger.info(f"  Prompt {i+1}/{num_prompts_local}: {status}")

        logger.info(
            f"Summary: {valid_mask.sum().item()}/{num_prompts_local} prompts kept "
            f"(ratio: {kept_ratio:.2%})"
        )
        logger.info(f"{'='*80}\n")

        # Build list of valid prompts with cached data
        valid_data = []
        for i in range(num_prompts_local):
            if not valid_mask[i]:
                continue  # Skip zero-variance prompts

            # Extract data for this prompt
            start_idx = i * self.num_generations
            end_idx = (i + 1) * self.num_generations

            valid_data.append({
                'original_prompt': batch_prompts[i],
                'prompt_text': prompts_text[i],
                'cached_completions_text': completions_text[start_idx:end_idx],
                'cached_rewards': rewards[start_idx:end_idx],
                'cached_prompt_ids': prompt_ids[i:i+1],
                'cached_prompt_mask': prompt_mask[i:i+1],
                'cached_completion_ids': completion_ids[start_idx:end_idx],
            })

        return valid_data

    def _prepare_inputs_from_cache(self, cached_data_list: list[dict[str, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare training inputs from cached data (from dynamic sampling).

        Args:
            cached_data_list: List of dicts containing pre-computed completions and rewards

        Returns:
            Dictionary with all necessary data for training step
        """
        device = self.accelerator.device

        # Stack all the cached data
        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids_list = []
        all_completions_text = []
        all_rewards = []

        for cached_data in cached_data_list:
            # Each cached_data contains m generations for one prompt
            all_prompt_ids.append(cached_data['cached_prompt_ids'].repeat(self.num_generations, 1))
            all_prompt_masks.append(cached_data['cached_prompt_mask'].repeat(self.num_generations, 1))
            all_completion_ids_list.append(cached_data['cached_completion_ids'])
            all_completions_text.extend(cached_data['cached_completions_text'])
            all_rewards.append(cached_data['cached_rewards'])

        # Pad prompts and completions to same length (different prompts from dynamic sampling may have different lengths)
        # Flatten the list of prompt tensors for padding
        all_prompt_ids_flat = [p for prompt_batch in all_prompt_ids for p in prompt_batch]
        all_prompt_masks_flat = [m for mask_batch in all_prompt_masks for m in mask_batch]

        # Flatten completion_ids list for padding
        all_completion_ids_flat = [c for completion_batch in all_completion_ids_list for c in completion_batch]

        # Pad to max length in this batch
        prompt_ids = pad(all_prompt_ids_flat, padding_value=self.processing_class.pad_token_id, padding_side='left').to(device)
        prompt_mask = pad(all_prompt_masks_flat, padding_value=0, padding_side='left').to(device)
        completion_ids = pad(all_completion_ids_flat, padding_value=self.processing_class.pad_token_id, padding_side='right').to(device)

        # Concatenate rewards
        rewards = torch.cat(all_rewards, dim=0).to(device)

        # Compute completion masks (mask after EOS)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate for full sequence
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute reference log probs in micro-batches to reduce peak memory
        micro_batch_size = 12  # Process 12 sequences at a time (2 prompts × 6 generations)
        ref_per_token_logps_list = []

        with torch.inference_mode():
            for i in range(0, len(prompt_completion_ids), micro_batch_size):
                end_idx = min(i + micro_batch_size, len(prompt_completion_ids))

                if self.ref_model is not None:
                    ref_logps_chunk = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_ids[i:end_idx],
                        attention_mask[i:end_idx],
                        logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_logps_chunk = self._get_per_token_logps(
                            self.model,
                            prompt_completion_ids[i:end_idx],
                            attention_mask[i:end_idx],
                            logits_to_keep
                        )

                ref_per_token_logps_list.append(ref_logps_chunk)

        # Concatenate all chunks
        ref_per_token_logps = torch.cat(ref_per_token_logps_list, dim=0)

        # Compute group-wise statistics for advantages
        # Rewards are already computed and cached
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize to get advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Log metrics
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def _prepare_inputs(self, inputs: Union[dict[str, Any], list[dict[str, Any]]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare inputs for training.

        If dynamic sampling is enabled, `inputs` will be a list of cached data dicts from
        the DynamicSamplingDataLoader. Otherwise, it's a standard batch dict.

        This method either:
        1. Uses cached data from dynamic sampling (already filtered)
        2. Generates completions and computes rewards normally (no dynamic sampling)
        """
        device = self.accelerator.device

        # Check if this is cached data from dynamic sampling
        if isinstance(inputs, list) and len(inputs) > 0 and 'cached_completions_text' in inputs[0]:
            # This is from dynamic sampling - use cached data
            return self._prepare_inputs_from_cache(inputs)

        # Standard path: generate and compute (no caching)
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)  # Use parent class method
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask after EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute reference log probs
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Compute rewards
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # MMR reweighting (same as GRPO)
        if self.SMI_reweighting or self.MMR_reweighting or self.MMR_STD_reweighting or self.MMR_SIGMOID_reweighting:
            rewards_before_reweighting = rewards.clone() if self.args.log_reward_reweighting else None

            if isinstance(completions[0], list):
                completions_flat = [
                    " ".join([turn.get("content", "") for turn in comp if isinstance(turn, dict)])
                    for comp in completions
                ]
            elif isinstance(completions[0], dict):
                completions_flat = [comp.get("content", "") for comp in completions]
            else:
                completions_flat = [str(c) for c in completions]

            if self.extractor_name == "nomic":
                embeddings = self.sentence_extractor.encode(completions_flat, max_tokens_per_text=3584, device=device)
            elif self.extractor_name == "jina":
                embeddings = self.sentence_extractor.encode(completions_flat, max_length=3584, device=device)
            else:
                raise NotImplementedError("Embedding model not implemented.")

            embeddings = torch.from_numpy(embeddings).to(device)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            if self.SMI_reweighting:
                similarity_matrix = embeddings @ embeddings.T
                similarity_sums = similarity_matrix.sum(dim=1)
                diversity_weights = 1.0 / (similarity_sums + 1e-6)
                diversity_weights = gather(diversity_weights)
                rewards = rewards * diversity_weights

            if self.MMR_reweighting or self.MMR_STD_reweighting or self.MMR_SIGMOID_reweighting:
                all_embeddings = gather(embeddings)
                all_rewards = rewards

                lambda_div_used = None
                if self.accelerator.is_main_process:
                    if self.MMR_SIGMOID_reweighting:
                        adjusted_rewards, lambda_div_used = diverse_adjust_rewards_fast_sigmoid(
                            all_rewards, all_embeddings
                        )
                    elif self.MMR_STD_reweighting:
                        adjusted_rewards, lambda_div_used = diverse_adjust_rewards_fast_std(
                            all_rewards, all_embeddings, self.mmr_std_temp
                        )
                    else:
                        adjusted_rewards = diverse_adjust_rewards_fast(all_rewards, all_embeddings, self.lambda_div)
                        lambda_div_used = self.lambda_div
                else:
                    adjusted_rewards = torch.zeros_like(all_rewards)

                adjusted_rewards_list = [adjusted_rewards]
                broadcast_object_list(adjusted_rewards_list, from_process=0)
                rewards = adjusted_rewards_list[0].to(self.accelerator.device)

                if self.args.log_reward_reweighting:
                    lambda_list = [lambda_div_used if lambda_div_used is not None else 0.0]
                    broadcast_object_list(lambda_list, from_process=0)
                    lambda_div_used = lambda_list[0]

                    if self.accelerator.is_main_process:
                        log_reward_reweighting_metrics(
                            self._metrics, rewards_before_reweighting, rewards, lambda_div_used
                        )

        # Compute group-wise statistics for advantage calculation
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize to get advantages (DAPO uses same advantage as GRPO but different loss)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep local portion
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute DAPO loss with token-level policy gradient and asymmetric clipping.

        Key differences from GRPO:
        1. Token-level normalization: Loss is averaged over total tokens, not samples
        2. Asymmetric clipping: Different ε_low and ε_high (0.2 vs 0.28)
        3. Clip-higher strategy prevents entropy collapse while allowing exploration
        """
        if return_outputs:
            raise ValueError("The DAPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Micro-batch the training forward pass to reduce peak memory
        # Use gradient checkpointing to avoid storing intermediate logits for backward pass
        micro_batch_size = 12  # Process 12 sequences at a time (2 prompts × 6 generations)
        per_token_logps_list = []

        for i in range(0, len(input_ids), micro_batch_size):
            end_idx = min(i + micro_batch_size, len(input_ids))

            # Wrap in gradient checkpointing to save memory during backward pass
            # This recomputes the forward pass during backward instead of storing intermediates
            chunk_logps = torch.utils.checkpoint.checkpoint(
                self._get_per_token_logps,
                model,
                input_ids[i:end_idx],
                attention_mask[i:end_idx],
                logits_to_keep,
                use_reentrant=False  # Use new non-reentrant API (more memory efficient)
            )
            per_token_logps_list.append(chunk_logps)

        per_token_logps = torch.cat(per_token_logps_list, dim=0)

        # Compute KL divergence
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Compute importance sampling ratio
        advantages = inputs["advantages"]
        ratio = torch.exp(per_token_logps - per_token_logps.detach())  # π_θ / π_θ_old

        # DAPO: Asymmetric clipping with clip-higher strategy
        ratio_clipped = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)

        # Token-level policy gradient loss
        per_token_loss = -torch.min(
            ratio * advantages.unsqueeze(1),
            ratio_clipped * advantages.unsqueeze(1)
        )

        # Add KL penalty (Removed from DAPO)
        # per_token_loss = per_token_loss + self.beta * per_token_kl

        # DAPO: Token-level normalization (key difference from GRPO)
        # Instead of averaging over samples, we average over total valid tokens
        total_tokens = completion_mask.sum()
        loss = (per_token_loss * completion_mask).sum() / (total_tokens + 1e-8)

        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Log clipping statistics
        clipped_ratio = (ratio != ratio_clipped).float()
        self._metrics["clip_fraction"].append(
            self.accelerator.gather_for_metrics((clipped_ratio * completion_mask).sum() / (total_tokens + 1e-8)).mean().item()
        )

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """Creates a model card for DAPO training."""
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{yu2025dapo,
                title        = {{DAPO: An Open-Source LLM Reinforcement Learning System at Scale}},
                author       = {Qiying Yu and Zheng Zhang and others},
                year         = 2025,
                eprint       = {arXiv:2503.14476},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="DAPO",
            trainer_citation=citation,
            paper_title="DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
            paper_id="2503.14476",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
