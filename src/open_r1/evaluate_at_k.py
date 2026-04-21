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

"""Custom evaluation tasks with pass@k, avg@k, and majority_vote@k metrics for LightEval."""

import numpy as np
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    MultilingualExtractiveMatchMetric,
)
from lighteval.metrics.metrics_sample import AvgAtN, MajAtN, PassAtK
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# Prompt template adapted from
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# Note that it is important to have the final answer in a box for math-verify to work correctly
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

# Define the majority vote metrics matching evaluate.py configurations

def majority_vote_at_k_latex_gold(sample_params: dict):
    """Create a majority_vote@k metric for math problems with latex gold answers.

    Matches latex_gold_metric from evaluate.py:
    - Gold extraction: LatexExtractionConfig only
    - Pred extraction: ExprExtractionConfig, then LatexExtractionConfig with boxed_match_priority=0
    - Fallback mode: first_match
    - Precision: 5
    - Aggregation: max

    Args:
        sample_params: Dictionary with 'n' (number of samples to generate)

    Returns:
        SampleLevelMetric configured for majority voting on math problems
    """
    n = sample_params.get("n", 1)

    return SampleLevelMetric(
        metric_name=f"majority_vote@{n}",
        sample_level_fn=MajAtN(
            n=n,
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=[LatexExtractionConfig()],
                pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)],
                aggregation_function=max,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def majority_vote_at_k_expr_gold(sample_params: dict):
    """Create a majority_vote@k metric for math problems with expression gold answers.

    Matches expr_gold_metric from evaluate.py:
    - Gold extraction: ExprExtractionConfig only
    - Pred extraction: ExprExtractionConfig, then LatexExtractionConfig with boxed_match_priority=0
    - Fallback mode: first_match
    - Precision: 5
    - Aggregation: max

    Args:
        sample_params: Dictionary with 'n' (number of samples to generate)

    Returns:
        SampleLevelMetric configured for majority voting on math problems
    """
    n = sample_params.get("n", 1)

    return SampleLevelMetric(
        metric_name=f"majority_vote@{n}",
        sample_level_fn=MajAtN(
            n=n,
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=[ExprExtractionConfig()],
                pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)],
                aggregation_function=max,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


# Define pass@k metrics matching evaluate.py configurations

def pass_at_k_latex_gold(sample_params: dict):
    """Create a pass@k metric for math problems with latex gold answers.

    Matches latex_gold_metric from evaluate.py.

    Args:
        sample_params: Dictionary with 'k' and 'n' (number of samples)

    Returns:
        SampleLevelMetric configured for pass@k on math problems
    """
    k = sample_params.get("k", 1)
    n = sample_params.get("n", 1)

    return SampleLevelMetric(
        metric_name=f"pass@{k}",
        sample_level_fn=PassAtK(
            k=k,
            n=n,
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=[LatexExtractionConfig()],
                pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)],
                aggregation_function=max,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def pass_at_k_expr_gold(sample_params: dict):
    """Create a pass@k metric for math problems with expression gold answers.

    Matches expr_gold_metric from evaluate.py.

    Args:
        sample_params: Dictionary with 'k' and 'n' (number of samples)

    Returns:
        SampleLevelMetric configured for pass@k on math problems
    """
    k = sample_params.get("k", 1)
    n = sample_params.get("n", 1)

    return SampleLevelMetric(
        metric_name=f"pass@{k}",
        sample_level_fn=PassAtK(
            k=k,
            n=n,
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=[ExprExtractionConfig()],
                pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)],
                aggregation_function=max,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


# Define avg@n metrics matching evaluate.py configurations

def avg_at_n_latex_gold(sample_params: dict):
    """Create an avg@n metric for math problems with latex gold answers.

    Matches latex_gold_metric from evaluate.py.

    Args:
        sample_params: Dictionary with 'n' (number of samples)

    Returns:
        SampleLevelMetric configured for avg@n on math problems
    """
    n = sample_params.get("n", 1)

    return SampleLevelMetric(
        metric_name=f"avg@{n}",
        sample_level_fn=AvgAtN(
            n=n,
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=[LatexExtractionConfig()],
                pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)],
                aggregation_function=max,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def avg_at_n_expr_gold(sample_params: dict):
    """Create an avg@n metric for math problems with expression gold answers.

    Matches expr_gold_metric from evaluate.py.

    Args:
        sample_params: Dictionary with 'n' (number of samples)

    Returns:
        SampleLevelMetric configured for avg@n on math problems
    """
    n = sample_params.get("n", 1)

    return SampleLevelMetric(
        metric_name=f"avg@{n}",
        sample_level_fn=AvgAtN(
            n=n,
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=[ExprExtractionConfig()],
                pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)],
                aggregation_function=max,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def math_500_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def aime_prompt_fn(line, task_name: str = None):
    query = MATH_QUERY_TEMPLATE.format(Question=line["problem"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["answer"]],
        gold_index=0,
    )


def amc_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def minerva_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def olympiadbench_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )


# Define task configurations with pass@k, avg@k, and majority_vote@k metrics

# MATH-500 with pass@k, avg@k, and majority_vote@k
math_500_at_k = LightevalTaskConfig(
    name="math_500_at_k",
    prompt_function=math_500_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        pass_at_k_latex_gold(sample_params={"k": 1, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 2, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 4, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 8, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 16, "n": 16}),
        avg_at_n_latex_gold(sample_params={"n": 1}),
        avg_at_n_latex_gold(sample_params={"n": 2}),
        avg_at_n_latex_gold(sample_params={"n": 4}),
        avg_at_n_latex_gold(sample_params={"n": 8}),
        avg_at_n_latex_gold(sample_params={"n": 16}),
        majority_vote_at_k_latex_gold(sample_params={"n": 1}),
        majority_vote_at_k_latex_gold(sample_params={"n": 2}),
        majority_vote_at_k_latex_gold(sample_params={"n": 4}),
        majority_vote_at_k_latex_gold(sample_params={"n": 8}),
        majority_vote_at_k_latex_gold(sample_params={"n": 16}),
    ],
    version=1,
)

# AIME 2024 with pass@k, avg@k, and majority_vote@k
aime24_at_k = LightevalTaskConfig(
    name="aime24_at_k",
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        pass_at_k_expr_gold(sample_params={"k": 1, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 2, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 4, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 8, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 16, "n": 16}),
        avg_at_n_expr_gold(sample_params={"n": 1}),
        avg_at_n_expr_gold(sample_params={"n": 2}),
        avg_at_n_expr_gold(sample_params={"n": 4}),
        avg_at_n_expr_gold(sample_params={"n": 8}),
        avg_at_n_expr_gold(sample_params={"n": 16}),
        majority_vote_at_k_expr_gold(sample_params={"n": 1}),
        majority_vote_at_k_expr_gold(sample_params={"n": 2}),
        majority_vote_at_k_expr_gold(sample_params={"n": 4}),
        majority_vote_at_k_expr_gold(sample_params={"n": 8}),
        majority_vote_at_k_expr_gold(sample_params={"n": 16}),
    ],
    version=1,
)

# AIME 2025 with pass@k, avg@k, and majority_vote@k
aime25_at_k = LightevalTaskConfig(
    name="aime25_at_k",
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        pass_at_k_expr_gold(sample_params={"k": 1, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 2, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 4, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 8, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 16, "n": 16}),
        avg_at_n_expr_gold(sample_params={"n": 1}),
        avg_at_n_expr_gold(sample_params={"n": 2}),
        avg_at_n_expr_gold(sample_params={"n": 4}),
        avg_at_n_expr_gold(sample_params={"n": 8}),
        avg_at_n_expr_gold(sample_params={"n": 16}),
        majority_vote_at_k_expr_gold(sample_params={"n": 1}),
        majority_vote_at_k_expr_gold(sample_params={"n": 2}),
        majority_vote_at_k_expr_gold(sample_params={"n": 4}),
        majority_vote_at_k_expr_gold(sample_params={"n": 8}),
        majority_vote_at_k_expr_gold(sample_params={"n": 16}),
    ],
    version=1,
)

# Minerva with pass@k, avg@k, and majority_vote@k
minerva_at_k = LightevalTaskConfig(
    name="minerva_at_k",
    prompt_function=minerva_prompt_fn,
    hf_repo="knoveleng/Minerva-Math",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        pass_at_k_latex_gold(sample_params={"k": 1, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 2, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 4, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 8, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 16, "n": 16}),
        avg_at_n_latex_gold(sample_params={"n": 1}),
        avg_at_n_latex_gold(sample_params={"n": 2}),
        avg_at_n_latex_gold(sample_params={"n": 4}),
        avg_at_n_latex_gold(sample_params={"n": 8}),
        avg_at_n_latex_gold(sample_params={"n": 16}),
        majority_vote_at_k_latex_gold(sample_params={"n": 1}),
        majority_vote_at_k_latex_gold(sample_params={"n": 2}),
        majority_vote_at_k_latex_gold(sample_params={"n": 4}),
        majority_vote_at_k_latex_gold(sample_params={"n": 8}),
        majority_vote_at_k_latex_gold(sample_params={"n": 16}),
    ],
    version=1,
)

# AMC23 with pass@k, avg@k, and majority_vote@k
amc23_at_k = LightevalTaskConfig(
    name="amc23_at_k",
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        pass_at_k_expr_gold(sample_params={"k": 1, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 2, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 4, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 8, "n": 16}),
        pass_at_k_expr_gold(sample_params={"k": 16, "n": 16}),
        avg_at_n_expr_gold(sample_params={"n": 1}),
        avg_at_n_expr_gold(sample_params={"n": 2}),
        avg_at_n_expr_gold(sample_params={"n": 4}),
        avg_at_n_expr_gold(sample_params={"n": 8}),
        avg_at_n_expr_gold(sample_params={"n": 16}),
        majority_vote_at_k_expr_gold(sample_params={"n": 1}),
        majority_vote_at_k_expr_gold(sample_params={"n": 2}),
        majority_vote_at_k_expr_gold(sample_params={"n": 4}),
        majority_vote_at_k_expr_gold(sample_params={"n": 8}),
        majority_vote_at_k_expr_gold(sample_params={"n": 16}),
    ],
    version=1,
)

# OlympiadBench with pass@k, avg@k, and majority_vote@k
olympiadbench_at_k = LightevalTaskConfig(
    name="olympiadbench_at_k",
    prompt_function=olympiadbench_prompt_fn,
    hf_repo="knoveleng/OlympiadBench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        pass_at_k_latex_gold(sample_params={"k": 1, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 2, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 4, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 8, "n": 16}),
        pass_at_k_latex_gold(sample_params={"k": 16, "n": 16}),
        avg_at_n_latex_gold(sample_params={"n": 1}),
        avg_at_n_latex_gold(sample_params={"n": 2}),
        avg_at_n_latex_gold(sample_params={"n": 4}),
        avg_at_n_latex_gold(sample_params={"n": 8}),
        avg_at_n_latex_gold(sample_params={"n": 16}),
        majority_vote_at_k_latex_gold(sample_params={"n": 1}),
        majority_vote_at_k_latex_gold(sample_params={"n": 2}),
        majority_vote_at_k_latex_gold(sample_params={"n": 4}),
        majority_vote_at_k_latex_gold(sample_params={"n": 8}),
        majority_vote_at_k_latex_gold(sample_params={"n": 16}),
    ],
    version=1,
)

# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(math_500_at_k)
TASKS_TABLE.append(aime24_at_k)
TASKS_TABLE.append(aime25_at_k)
TASKS_TABLE.append(minerva_at_k)
TASKS_TABLE.append(amc23_at_k)
TASKS_TABLE.append(olympiadbench_at_k)

# MODULE LOGIC
if __name__ == "__main__":
    print([t.name for t in TASKS_TABLE])
    print(f"Total tasks: {len(TASKS_TABLE)}")
