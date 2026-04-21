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
Utilities for visualizing reward reweighting and advantages in GRPO training.
Supports both online and offline wandb modes.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


def create_reward_comparison_plot(
    rewards_before: torch.Tensor,
    rewards_after: torch.Tensor,
    step: int,
    lambda_div: Optional[float] = None,
) -> Figure:
    """
    Create a scatter plot comparing rewards before and after reweighting.

    Args:
        rewards_before: Original rewards before reweighting
        rewards_after: Adjusted rewards after reweighting
        step: Current training step
        lambda_div: Lambda diversity parameter (if applicable)

    Returns:
        matplotlib Figure object
    """
    rewards_before_np = rewards_before.detach().cpu().numpy()
    rewards_after_np = rewards_after.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: before vs after
    axes[0].scatter(rewards_before_np, rewards_after_np, alpha=0.5, s=20)
    axes[0].plot(
        [rewards_before_np.min(), rewards_before_np.max()],
        [rewards_before_np.min(), rewards_before_np.max()],
        'r--', label='y=x (no change)'
    )
    axes[0].set_xlabel('Original Rewards', fontsize=12)
    axes[0].set_ylabel('Adjusted Rewards', fontsize=12)
    title = f'Reward Comparison (Step {step})'
    if lambda_div is not None:
        title += f'\nλ_div = {lambda_div:.2f}'
    axes[0].set_title(title, fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distribution comparison
    axes[1].hist(rewards_before_np, bins=30, alpha=0.5, label='Before', density=True)
    axes[1].hist(rewards_after_np, bins=30, alpha=0.5, label='After', density=True)
    axes[1].set_xlabel('Reward Value', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Reward Distributions', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_advantage_plot(
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    step: int,
    num_generations: int,
) -> Figure:
    """
    Create visualization of advantages and their relationship to rewards.

    Args:
        advantages: Computed advantages
        rewards: Final rewards (after reweighting if applicable)
        step: Current training step
        num_generations: Number of generations per prompt

    Returns:
        matplotlib Figure object
    """
    advantages_np = advantages.detach().cpu().numpy()
    rewards_np = rewards.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Advantage distribution
    axes[0].hist(advantages_np, bins=40, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero advantage')
    axes[0].set_xlabel('Advantage Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Advantage Distribution (Step {step})', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reward vs Advantage scatter
    axes[1].scatter(rewards_np, advantages_np, alpha=0.4, s=15)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Reward', fontsize=12)
    axes[1].set_ylabel('Advantage', fontsize=12)
    axes[1].set_title('Reward vs Advantage', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # Box plot per generation group (show first 10 groups if many)
    num_prompts = len(rewards_np) // num_generations
    max_groups_to_show = min(10, num_prompts)

    advantages_grouped = []
    for i in range(max_groups_to_show):
        start_idx = i * num_generations
        end_idx = start_idx + num_generations
        advantages_grouped.append(advantages_np[start_idx:end_idx])

    axes[2].boxplot(advantages_grouped, labels=[f'P{i}' for i in range(max_groups_to_show)])
    axes[2].axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[2].set_xlabel('Prompt Group', fontsize=12)
    axes[2].set_ylabel('Advantage', fontsize=12)
    axes[2].set_title(f'Advantages per Prompt (first {max_groups_to_show})', fontsize=13)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def save_reward_data_to_csv(
    output_dir: str,
    step: int,
    rewards_before: Optional[torch.Tensor] = None,
    rewards_after: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    lambda_div_used: Optional[float] = None,
    prompts: Optional[list] = None,
    completions: Optional[list] = None,
):
    """
    Save reward data to a single CSV file, appending data from each step.
    All training steps are saved to the same file: reward_data/all_rewards.csv

    Args:
        output_dir: Directory to save CSV files
        step: Current training step
        rewards_before: Original rewards (optional)
        rewards_after: Adjusted rewards (optional)
        advantages: Advantages (optional)
        lambda_div_used: Lambda value used (optional)
        prompts: List of prompt texts (optional)
        completions: List of completion texts (optional)
    """
    import csv
    import os

    reward_data_dir = Path(output_dir) / "reward_data"
    reward_data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reward_data_dir / "all_rewards.csv"

    # Determine which data we have
    has_before = rewards_before is not None
    has_after = rewards_after is not None
    has_advantages = advantages is not None
    has_prompts = prompts is not None
    has_completions = completions is not None

    if not (has_before or has_after or has_advantages):
        return  # Nothing to save

    # Convert to numpy
    data_dict = {}
    if has_before:
        data_dict['reward_before'] = rewards_before.detach().cpu().numpy()
    if has_after:
        data_dict['reward_after'] = rewards_after.detach().cpu().numpy()
    if has_advantages:
        data_dict['advantage'] = advantages.detach().cpu().numpy()

    # Determine length
    n_samples = len(next(iter(data_dict.values())))

    # Determine fieldnames - put text fields first for readability
    fieldnames = ['step', 'sample_idx']
    if lambda_div_used is not None:
        fieldnames.append('lambda_div')
    if has_prompts:
        fieldnames.append('prompt')
    if has_completions:
        fieldnames.append('completion')
    fieldnames.extend(data_dict.keys())

    # Check if file exists and if this step has already been logged
    file_exists = os.path.exists(csv_path)
    step_already_logged = False

    if file_exists:
        # Check if this step already exists in the CSV to avoid duplicates on resume
        try:
            import pandas as pd
            existing_df = pd.read_csv(csv_path)
            if 'step' in existing_df.columns:
                step_already_logged = step in existing_df['step'].values
        except Exception:
            # If pandas fails or file is corrupted, continue anyway
            pass

    # Skip if this step was already logged (e.g., when resuming from checkpoint)
    if step_already_logged:
        return

    # Append to CSV file
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        # Write all samples for this step
        for i in range(n_samples):
            row = {'step': step, 'sample_idx': i}
            if lambda_div_used is not None:
                row['lambda_div'] = lambda_div_used
            if has_prompts:
                # Handle both string and list-of-dict prompts
                if isinstance(prompts[i], str):
                    row['prompt'] = prompts[i]
                else:
                    # For conversational format, extract content
                    row['prompt'] = str(prompts[i])
            if has_completions:
                # Handle both string and list-of-dict completions
                if isinstance(completions[i], str):
                    row['completion'] = completions[i]
                elif isinstance(completions[i], list) and len(completions[i]) > 0:
                    # Extract content from conversational format
                    if isinstance(completions[i][0], dict):
                        row['completion'] = completions[i][0].get('content', str(completions[i]))
                    else:
                        row['completion'] = str(completions[i])
                else:
                    row['completion'] = str(completions[i])
            for key, values in data_dict.items():
                row[key] = float(values[i])
            writer.writerow(row)


def log_reward_reweighting_metrics(
    metrics_dict: Dict[str, Any],
    rewards_before: torch.Tensor,
    rewards_after: torch.Tensor,
    lambda_div_used: Optional[float] = None,
):
    """
    Compute and add reward reweighting metrics to the metrics dictionary.

    Note: metrics_dict is expected to be a defaultdict(list) where each metric
    is stored as a list of values to be averaged later.

    Args:
        metrics_dict: Dictionary to update with metrics (defaultdict(list))
        rewards_before: Original rewards
        rewards_after: Adjusted rewards
        lambda_div_used: Lambda value used (for adaptive methods)
    """
    reward_change = rewards_after - rewards_before

    metrics_dict['reward_before_mean'].append(rewards_before.mean().item())
    metrics_dict['reward_before_std'].append(rewards_before.std().item())
    metrics_dict['reward_after_mean'].append(rewards_after.mean().item())
    metrics_dict['reward_after_std'].append(rewards_after.std().item())
    metrics_dict['reward_change_mean'].append(reward_change.mean().item())
    metrics_dict['reward_change_std'].append(reward_change.std().item())
    metrics_dict['reward_change_max'].append(reward_change.max().item())
    metrics_dict['reward_change_min'].append(reward_change.min().item())

    if lambda_div_used is not None:
        metrics_dict['lambda_div_used'].append(lambda_div_used)


def log_advantage_metrics(
    metrics_dict: Dict[str, Any],
    advantages: torch.Tensor,
    rewards: torch.Tensor,
):
    """
    Compute and add advantage metrics to the metrics dictionary.

    Note: metrics_dict is expected to be a defaultdict(list) where each metric
    is stored as a list of values to be averaged later.

    Args:
        metrics_dict: Dictionary to update with metrics (defaultdict(list))
        advantages: Computed advantages
        rewards: Final rewards
    """
    metrics_dict['advantage_mean'].append(advantages.mean().item())
    metrics_dict['advantage_std'].append(advantages.std().item())
    metrics_dict['advantage_max'].append(advantages.max().item())
    metrics_dict['advantage_min'].append(advantages.min().item())

    # Correlation between rewards and advantages
    if len(advantages) > 1:
        corr = np.corrcoef(
            rewards.detach().cpu().numpy(),
            advantages.detach().cpu().numpy()
        )[0, 1]
        metrics_dict['reward_advantage_correlation'].append(float(corr))
