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
Dynamic Sampling DataLoader for DAPO

This dataloader wrapper implements DAPO's dynamic sampling mechanism:
- Generates m completions per prompt
- Filters out prompt-groups with zero reward variance
- Continues sampling until a full batch of valid prompts is collected
"""

import logging
import os
from datetime import datetime
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class DynamicSamplingDataLoader:
    """
    Wraps a base dataloader to implement DAPO's dynamic sampling.

    For each batch request, this dataloader:
    1. Fetches prompts from the base dataloader
    2. For each prompt, generates m completions
    3. Computes rewards for all completions
    4. Filters out prompt-groups where all rewards are identical (zero variance)
    5. Continues until enough valid prompt-groups are collected

    Args:
        base_dataloader: The underlying HuggingFace dataloader
        trainer: Reference to the DAPOTrainer instance
        max_attempts: Maximum number of base batches to try before giving up
    """

    def __init__(self, base_dataloader, trainer, max_attempts: int = 10):
        self.base_dataloader = base_dataloader
        self.trainer = trainer
        self.max_attempts = max_attempts
        self.base_iterator = None

        # Track batches served for checkpoint resumption
        self._batches_served = 0
        self._skip_batches = 0

        # Expose standard DataLoader attributes to prevent AttributeError in skip_first_batches
        # These are needed because skip_first_batches checks for them before our patch can intercept
        self.dataset = getattr(base_dataloader, 'dataset', None)
        self.sampler = getattr(base_dataloader, 'sampler', None)
        self.batch_sampler = getattr(base_dataloader, 'batch_sampler', None)

        # Initialize log file for dynamic sampling statistics
        self.log_file_path = None
        if hasattr(trainer.args, 'output_dir') and trainer.args.output_dir:
            self.log_file_path = os.path.join(trainer.args.output_dir, "dynamic_sampling_log.txt")
            # Create output directory if it doesn't exist
            os.makedirs(trainer.args.output_dir, exist_ok=True)
            # Write header if file doesn't exist
            if not os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'w') as f:
                    f.write("Dynamic Sampling Statistics Log\n")
                    f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"{'Step':<8} {'Attempts':<10} {'Total Prompts':<15} {'Valid Prompts':<15} {'Efficiency':<12}\n")
                    f.write("=" * 80 + "\n")

    def __iter__(self) -> Iterator:
        self.base_iterator = iter(self.base_dataloader)
        self._batches_served = 0
        return self

    def skip_batches(self, num_batches: int):
        """
        Skip the first num_batches when iterating.
        Called by transformers when resuming from checkpoint.
        """
        self._skip_batches = num_batches
        logger.info(f"DynamicSamplingDataLoader: Will skip first {num_batches} batches when iterating")

    def __next__(self) -> dict[str, Any]:
        """
        Returns a batch of valid prompt-groups (with non-zero reward variance).

        Returns:
            Dictionary containing:
            - 'prompts': List of valid prompts
            - 'cached_data': Pre-computed completions, rewards, etc.
        """
        # Handle skipping batches for checkpoint resumption
        # IMPORTANT: We skip base_dataloader batches directly without generation
        # to avoid the massive time cost of generating and discarding batches
        if self._batches_served < self._skip_batches:
            num_to_skip = self._skip_batches - self._batches_served
            logger.info(f"Fast-skipping {num_to_skip} batches from base dataloader for checkpoint resumption")

            # Skip batches in the base iterator without generation
            for i in range(num_to_skip):
                try:
                    _ = next(self.base_iterator)
                    self._batches_served += 1
                    if (i + 1) % 50 == 0:  # Log progress every 50 batches
                        logger.info(f"Skipped {i + 1}/{num_to_skip} batches...")
                except StopIteration:
                    logger.warning(f"Base dataloader exhausted after skipping {i} batches")
                    raise

            logger.info(f"Successfully skipped {num_to_skip} batches")

        # Normal batch generation
        collected_data = self._generate_batch()
        self._batches_served += 1
        return collected_data

    def _generate_batch(self) -> list:
        """
        Internal method to generate a single batch with dynamic sampling logic.
        """
        # DEBUG: Check if this is the first batch after skipping
        if self._batches_served == self._skip_batches and self._skip_batches > 0:
            logger.info(f"[DEBUG dataloader] First _generate_batch() call after skipping {self._skip_batches} batches")

        collected_data = []
        target_size = self.trainer.args.per_device_train_batch_size
        num_attempts = 0
        total_prompts_processed = 0  # Track total prompts seen across all attempts

        while len(collected_data) < target_size and num_attempts < self.max_attempts:
            # Get next batch of prompts from base dataloader
            try:
                base_batch = next(self.base_iterator)

                # DEBUG: Log first batch after skip
                if self._batches_served == self._skip_batches and self._skip_batches > 0 and num_attempts == 1:
                    print(f"[DEBUG dataloader] First base_batch after skip:")
                    print(f"[DEBUG dataloader] batch length: {len(base_batch)}")
                    if len(base_batch) > 0:
                        print(f"[DEBUG dataloader] First sample keys: {list(base_batch[0].keys())}")
                        print(f"[DEBUG dataloader] 'solution' present: {'solution' in base_batch[0]}")
                        if 'solution' in base_batch[0]:
                            sol = str(base_batch[0]['solution'])
                            print(f"[DEBUG dataloader] First solution: {sol[:100]}..." if len(sol) > 100 else f"[DEBUG dataloader] First solution: {sol}")

            except StopIteration:
                # End of epoch - use what we have if anything
                if len(collected_data) > 0:
                    logger.warning(
                        f"End of dataloader reached. Collected {len(collected_data)}/{target_size} "
                        f"valid prompts for this batch."
                    )
                    break
                raise  # Re-raise if no data collected

            num_attempts += 1
            batch_size = len(base_batch)
            total_prompts_processed += batch_size

            # Generate completions and filter zero-variance prompt-groups
            # This calls back into the trainer to handle generation and filtering
            valid_data = self.trainer._generate_and_filter_prompts(base_batch)

            if valid_data:
                collected_data.extend(valid_data)
                logger.info(
                    f"Dynamic sampling attempt {num_attempts}: {len(valid_data)}/{batch_size} valid prompts found. "
                    f"Total collected: {len(collected_data)}/{target_size}, "
                    f"Total prompts processed: {total_prompts_processed}"
                )
            else:
                logger.info(
                    f"Dynamic sampling attempt {num_attempts}: 0/{batch_size} valid prompts found (all zero-variance). "
                    f"Total prompts processed: {total_prompts_processed}"
                )

        # Check if we collected enough prompts
        if len(collected_data) == 0:
            logger.warning(
                f"Dynamic sampling failed to collect any valid prompts after {num_attempts} attempts. "
                f"Skipping this batch and moving to the next one. "
                f"Consider: (1) Increasing max_attempts, (2) Using easier prompts, "
                f"(3) Increasing num_generations, or (4) Disabling dynamic sampling."
            )
            # Recursively try the next batch
            return self._generate_batch()

        if len(collected_data) < target_size:
            logger.warning(
                f"Could only collect {len(collected_data)}/{target_size} valid prompts "
                f"after {num_attempts} attempts. Padding batch by repeating samples."
            )
            # Pad by repeating samples to reach target size
            while len(collected_data) < target_size:
                # Repeat from beginning
                idx = len(collected_data) % len(collected_data) if collected_data else 0
                collected_data.append(collected_data[idx])

        # Trim to exact target size if we overshot
        collected_data = collected_data[:target_size]

        # Calculate efficiency metrics
        efficiency = (target_size / total_prompts_processed * 100) if total_prompts_processed > 0 else 0.0

        # Log final statistics for this batch
        logger.info(
            f"Dynamic sampling summary: Collected {target_size} valid prompts "
            f"from {total_prompts_processed} total prompts in {num_attempts} attempts. "
            f"Efficiency: {efficiency:.1f}% (higher is better)"
        )

        # Write statistics to log file
        if self.log_file_path:
            current_step = getattr(self.trainer.state, 'global_step', 0)
            with open(self.log_file_path, 'a') as f:
                f.write(f"{current_step:<8} {num_attempts:<10} {total_prompts_processed:<15} "
                        f"{target_size:<15} {efficiency:<12.1f}%\n")

        # Save metrics for tracking and logging
        self.trainer._metrics["dapo/num_sampling_attempts"].append(num_attempts)
        self.trainer._metrics["dapo/total_prompts_processed"].append(total_prompts_processed)
        self.trainer._metrics["dapo/sampling_efficiency"].append(efficiency)
        self.trainer._metrics["dapo/valid_prompts_collected"].append(len(collected_data))

        return collected_data

    def __len__(self):
        """
        Return length of base dataloader.
        Note: Actual length may vary due to filtering.
        """
        return len(self.base_dataloader)
