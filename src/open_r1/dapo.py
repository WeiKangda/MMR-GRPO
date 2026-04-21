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

"""
DAPO Training Script

This script trains models using DAPO (Dynamic Adaptive Policy Optimization).
DAPO extends policy optimization with:
1. Token-Level Policy Gradient Loss (better for long CoT)
2. Clip-Higher Strategy (ε_low=0.2, ε_high=0.28)
3. Dynamic Sampling (filters non-diverse prompts)
4. Support for MMR Diversity Reweighting

Usage examples:

# Standard DAPO with dynamic sampling
python src/open_r1/dapo.py --config configs/dapo_default.yaml

# DAPO without dynamic sampling (more like GRPO)
python src/open_r1/dapo.py --config configs/dapo_no_sampling.yaml --enable_dynamic_sampling false

# DAPO with MMR reweighting
python src/open_r1/dapo.py --config configs/dapo_mmr.yaml --MMR_STD_reweighting true

# Compare: DAPO vs DAPO without dynamic sampling vs DAPO with MMR
# See configs/ directory for example configurations
"""

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch

# Apply patch for DynamicSamplingDataLoader checkpoint resumption
# CRITICAL: This MUST happen BEFORE importing transformers!
# Transformers imports skip_first_batches at module level, so we need to patch it first
from accelerate.data_loader import skip_first_batches as original_skip_first_batches

def patched_skip_first_batches(dataloader, num_batches):
    """Patched version that handles DynamicSamplingDataLoader specially."""
    # Import here to avoid circular dependency
    from src.open_r1.trl.trainer.dapo_dataloader import DynamicSamplingDataLoader

    if isinstance(dataloader, DynamicSamplingDataLoader):
        dataloader.skip_batches(num_batches)
        return dataloader
    else:
        return original_skip_first_batches(dataloader, num_batches)

# Apply the patch to accelerate.data_loader BEFORE transformers imports it
import accelerate.data_loader
accelerate.data_loader.skip_first_batches = patched_skip_first_batches

# NOW it's safe to import transformers (it will get our patched version)
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# Also patch transformers.trainer directly in case it cached the reference
import transformers.trainer
transformers.trainer.skip_first_batches = patched_skip_first_batches

from src.open_r1.configs import DAPOConfig
from src.open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from src.open_r1.trl import DAPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from src.open_r1.utils import get_tokenizer
from src.open_r1.utils.callbacks import get_callbacks
from src.open_r1.utils.wandb_logging import init_wandb_training

logger = logging.getLogger(__name__)


@dataclass
class DAPOScriptArguments(ScriptArguments):
    """
    Script arguments for the DAPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps',
            'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', "
            "'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Log DAPO-specific settings
    logger.info("=" * 80)
    logger.info("DAPO Configuration:")
    logger.info(f"  - Dynamic Sampling: {training_args.enable_dynamic_sampling}")
    if training_args.enable_dynamic_sampling:
        logger.info(f"    - Filter Metric: {training_args.filter_metric}")
        logger.info(f"    - Max Gen Batches: {training_args.max_num_gen_batches}")
    logger.info(f"  - Clip-Higher Strategy:")
    logger.info(f"    - Epsilon Low: {training_args.epsilon_low}")
    logger.info(f"    - Epsilon High: {training_args.epsilon_high}")
    logger.info(f"  - Token-Level Loss: Enabled (DAPO default)")
    logger.info(f"  - MMR Reweighting:")
    logger.info(f"    - SMI: {training_args.SMI_reweighting}")
    logger.info(f"    - MMR: {training_args.MMR_reweighting}")
    logger.info(f"    - MMR STD: {training_args.MMR_STD_reweighting}")
    logger.info(f"    - MMR SIGMOID: {training_args.MMR_SIGMOID_reweighting}")
    if training_args.MMR_reweighting:
        logger.info(f"    - Lambda Div: {training_args.lambda_div}")
    if training_args.MMR_STD_reweighting:
        logger.info(f"    - MMR STD Temp: {training_args.mmr_std_temp}")
    logger.info("=" * 80)

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions (same as GRPO)
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the DAPO trainer
    #############################
    logger.info("*** Initializing DAPO Trainer ***")
    trainer = DAPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # 🔍 DIAGNOSTIC: Check model state BEFORE training (especially after resume)
    if checkpoint is not None:
        logger.info("=" * 80)
        logger.info("🔍 CHECKPOINT RESUME DIAGNOSTICS")
        logger.info(f"Resuming from: {checkpoint}")
        logger.info("=" * 80)

        # Check model state dict keys
        model_state_dict = trainer.model.state_dict()
        logger.info(f"\n📊 Model state dict has {len(model_state_dict)} keys")

        # Check for problematic prefixes
        double_model_keys = [k for k in model_state_dict.keys() if k.startswith("model.model.")]
        module_model_keys = [k for k in model_state_dict.keys() if k.startswith("module.model.")]
        if double_model_keys:
            logger.error(f"❌ Found {len(double_model_keys)} keys with 'model.model.' prefix!")
            logger.error(f"   Examples: {double_model_keys[:3]}")
        if module_model_keys:
            logger.error(f"❌ Found {len(module_model_keys)} keys with 'module.model.' prefix!")
            logger.error(f"   Examples: {module_model_keys[:3]}")

        # Show first 15 keys
        logger.info(f"\n📋 First 15 state dict keys:")
        for i, key in enumerate(list(model_state_dict.keys())[:15]):
            logger.info(f"   [{i:2d}] '{key}'")

        # Check embedding and lm_head weights
        embed_keys = [k for k in model_state_dict.keys() if "embed_tokens" in k or "embed_in" in k or "wte" in k]
        lm_head_keys = [k for k in model_state_dict.keys() if "lm_head" in k or "embed_out" in k]

        logger.info(f"\n🔍 Embedding layer keys ({len(embed_keys)}):")
        for key in embed_keys:
            weight = model_state_dict[key]
            logger.info(f"   - {key}: shape={weight.shape}, mean={weight.float().mean().item():.6f}, std={weight.float().std().item():.6f}")

        logger.info(f"\n🔍 LM head keys ({len(lm_head_keys)}):")
        for key in lm_head_keys:
            weight = model_state_dict[key]
            logger.info(f"   - {key}: shape={weight.shape}, mean={weight.float().mean().item():.6f}, std={weight.float().std().item():.6f}")

        # Check a few critical layer weights
        layer_0_keys = [k for k in model_state_dict.keys() if ".layers.0." in k or ".layer.0." in k or ".h.0." in k]
        if layer_0_keys:
            logger.info(f"\n🔍 First transformer layer has {len(layer_0_keys)} parameters")
            # Show first 3
            for key in layer_0_keys[:3]:
                weight = model_state_dict[key]
                logger.info(f"   - {key}: shape={weight.shape}, mean={weight.float().mean().item():.6f}")

        # 🧪 WEIGHT SANITY CHECK: Verify weights are not corrupted
        logger.info("\n🧪 Weight sanity check...")

        # Check for NaN/Inf in critical weights
        critical_weights_ok = True
        for key in list(embed_keys) + list(lm_head_keys) + layer_0_keys[:5]:
            weight = model_state_dict[key]
            has_nan = torch.isnan(weight).any().item()
            has_inf = torch.isinf(weight).any().item()

            if has_nan or has_inf:
                logger.error(f"   ❌ CRITICAL: {key} contains {'NaN' if has_nan else 'Inf'} values!")
                critical_weights_ok = False

        if critical_weights_ok:
            logger.info(f"   ✅ All critical weights are numerically valid (no NaN/Inf)")
        else:
            logger.error(f"   ⚠️  WARNING: Model weights contain NaN or Inf - training will likely fail!")

        # NOTE: We skip actual inference test here because:
        # 1. It can fail in distributed settings due to device placement
        # 2. Flash attention doesn't work on CPU
        # 3. The real test is whether training generation works (which will happen in first step)
        logger.info(f"   💡 Actual generation quality will be tested during first training step")

        logger.info("=" * 80 + "\n")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")

    # If using LoRA/PEFT, save both the adapter and the merged model
    if model_args.use_peft and trainer.accelerator.is_main_process:
        # First, save the LoRA adapter separately
        adapter_output_dir = f"{training_args.output_dir}/adapter"
        logger.info(f"Saving LoRA adapter to {adapter_output_dir}")
        trainer.save_model(adapter_output_dir)

        # Then merge and save the full model
        logger.info("Merging LoRA weights with base model...")
        from peft import PeftModel

        model = trainer.model
        if isinstance(model, PeftModel):
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(training_args.output_dir)
            trainer.processing_class.save_pretrained(training_args.output_dir)
            logger.info(f"Merged model saved to {training_args.output_dir}")
        else:
            logger.warning("Model is not a PeftModel, saving normally")
            trainer.save_model(training_args.output_dir)
    else:
        # Standard save for non-LoRA models
        trainer.save_model(training_args.output_dir)

    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1", "dapo"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # Push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((DAPOScriptArguments, DAPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
