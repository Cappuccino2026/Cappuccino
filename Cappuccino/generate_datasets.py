import json
import random
from pathlib import Path
from typing import Any, Sequence
import click
import numpy as np
import seaborn as sns
from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from .utils import set_seed

NUM_PERMUTATIONS = 1
SEQ_LEN_UPPER_BOUND = 4096 
DEFAULT_TOKENIZER_NAME = "meta-llama/Meta-Llama-3-70B"
DATASET_INFO = {
    # =======================
    # 1. Summarization
    # =======================

    # 1. EdinburghNLP/xsum
    "xsum": {
        "dataset_name": "EdinburghNLP/xsum",
        "dataset_path": None,
        "dataset_split": "train",
        "dataset_keys": ["document", "summary"],
    },

    # 2. abisee/cnn_dailymail
    "cnn_dailymail": {
        "dataset_name": "abisee/cnn_dailymail",
        "dataset_path": "3.0.0",
        "dataset_split": "train",
        "dataset_keys": ["article", "highlights"],
    },

    # 3. d0rj/wikisum
    "wikisum": {
        "dataset_name": "d0rj/wikisum",
        "dataset_path": None,
        "dataset_split": "train",
        "dataset_keys": ["article", "summary"],
    },

    # 4. knkarthick/samsum
    "samsum": {
        "dataset_name": "knkarthick/samsum",
        "dataset_path": None,
        "dataset_split": "train",
        # columns: dialogue, summary
        "dataset_keys": ["dialogue", "summary"],
    },

    # =======================
    # 2. Math Reasoning
    # =======================

    # 5. openai/gsm8k
    "gsm8k": {
        "dataset_name": "openai/gsm8k",
        "dataset_path": "main",
        "dataset_split": "train",
        "dataset_keys": ["question", "answer"],
    },

    # 6. TIGER-Lab/MathInstruct
    "mathinstruct": {
        "dataset_name": "TIGER-Lab/MathInstruct",
        "dataset_path": None,
        "dataset_split": "train",
        # HF viewer shows instruction / output columns
        "dataset_keys": ["instruction", "output"],
    },

    # 7. meta-math/MetaMathQA
    "metamathqa": {
        "dataset_name": "meta-math/MetaMathQA",
        "dataset_path": None,
        "dataset_split": "train",
        # dataset columns: query (problem text), response (solution), type
        "dataset_keys": ["query", "response"],
    },

    # 8. AI-MO/NuminaMath-CoT
    "numinamath_cot": {
        "dataset_name": "AI-MO/NuminaMath-CoT",
        "dataset_path": None,
        "dataset_split": "train",
        # README: fields include problem / solution (plus source, messages, etc.)
        "dataset_keys": ["problem", "solution"],
    },

    # =======================
    # 3. Code / Software Engineering
    # =======================

    # 9. iamtarun/python_code_instructions_18k_alpaca
    "python_code_instructions_18k_alpaca": {
        "dataset_name": "iamtarun/python_code_instructions_18k_alpaca",
        "dataset_path": None,
        "dataset_split": "train",
        # features: instruction, input, output; use instruction→output for src/tgt
        "dataset_keys": ["instruction", "output"],
    },

    # 10. ise-uiuc/Magicoder-Evol-Instruct-110K
    "magicoder_evol_instruct_110k": {
        "dataset_name": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "dataset_path": None,          # single default subset/split train
        "dataset_split": "train",
        "dataset_keys": ["instruction", "response"],
    },

    # 11. chargoddard/commitpack-ft-instruct
    "commitpack_ft_instruct": {
        "dataset_name": "chargoddard/commitpack-ft-instruct",
        "dataset_path": None,
        "dataset_split": "train",
        # features include: id, rating, language, license, instruction, output, input
        "dataset_keys": ["instruction", "output"],
    },

    # 12. sahil2801/CodeAlpaca-20k
    "codealpaca_20k": {
        "dataset_name": "sahil2801/CodeAlpaca-20k",
        "dataset_path": None,
        "dataset_split": "train",
        # columns: instruction, input, output – follow Alpaca-style instruction/output
        "dataset_keys": ["instruction", "output"],
    },

    # =======================
    # 4. Domain-specific (Law / Bio / Scientific / Meetings)
    # =======================

    # 13. FiscalNote/billsum
    "billsum": {
        "dataset_name": "FiscalNote/billsum",
        "dataset_path": None,
        "dataset_split": "train",
        "dataset_keys": ["text", "summary"],
    },

    # 14. qiaojin/PubMedQA
    "pubmedqa": {
        "dataset_name": "qiaojin/PubMedQA",
        # configs: pqa_labeled / pqa_unlabeled / pqa_artificial; use labeled one
        "dataset_path": "pqa_labeled",
        "dataset_split": "train",
        # main generative fields: question + long_answer
        "dataset_keys": ["question", "long_answer"],
    },

    # 15. ccdv/arxiv-summarization
    "arxiv": {
        "dataset_name": "ccdv/arxiv-summarization",
        "dataset_path": "section",
        "dataset_split": "validation",
        "dataset_keys": ["article", "abstract"],
    },

    # 16. huuuyeah/meetingbank
    "meetingbank": {
        "dataset_name": "huuuyeah/meetingbank",
        "dataset_path": None,
        "dataset_split": "train",  # default split in HF viewer
        # fields: id, uid, transcript, summary
        "dataset_keys": ["transcript", "summary"],
    },
}

def compute_length_stats(tokens: Sequence[float]) -> dict[str, float]:
    """Compute length statistics for a 1D sequence of token lengths."""
    arr = np.asarray(tokens, dtype=np.float64)

    mean = float(np.mean(arr))
    std = float(np.std(arr))  # population std (ddof=0)
    median = float(np.median(arr))
    min_len = float(np.min(arr))
    max_len = float(np.max(arr))

    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))

    cv = float(std / mean) if mean > 0 else 0.0

    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = float(q3 - q1)

    if std > 0:
        centered = arr - mean
        m3 = np.mean(centered ** 3)
        skewness = float(m3 / (std ** 3))
    else:
        skewness = 0.0

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": min_len,
        "max": max_len,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "cv": cv,
        "iqr": iqr,
        "skewness": skewness,
    }

def sample_dataset_lengths(
    dataset_key: str,
    seeds: list[int],
    num_samples: int,
    num_permutations: int = NUM_PERMUTATIONS,
    upper_bound: int = SEQ_LEN_UPPER_BOUND,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
) -> dict[str, Any]:
    """Sample the sample lengths from the dataset length distribution."""
    if dataset_key not in DATASET_INFO:
        msg = f"Dataset {dataset_key} not supported"
        raise ValueError(msg)

    dataset_info = DATASET_INFO[dataset_key]
    name, path, split, keys = (
        dataset_info["dataset_name"],
        dataset_info["dataset_path"],
        dataset_info["dataset_split"],
        dataset_info["dataset_keys"],
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    result: dict[str, Any] = {"seeds": seeds}

    for seed in seeds:
        set_seed(seed)
        raw_dataset = load_dataset(name, path, split=split)
        dataset = raw_dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(num_samples, len(raw_dataset))))
        tokens = [
            min(
                sum(len(tokenizer.encode(sample[key])) for key in keys),
                upper_bound,
            )
            for sample in dataset
        ]
        tokens_copy = tokens.copy()
        stats = compute_length_stats(tokens)
        seed_result: dict[str, Any] = {
            "num_permutations": num_permutations,
            "num_samples": len(tokens),
            **stats,
        }
        for i in range(num_permutations):
            if i > 0:
                random.shuffle(tokens_copy)
            seed_result[f"permutation_{i + 1}"] = tokens_copy.copy()
        result[f"seed_{seed}"] = seed_result
    return result


# -----------------------------
# CLI 主函数
# -----------------------------
@click.command()
@click.option(
    "--seeds",
    type=click.INT,
    multiple=True,
    default=[16],
    help="The seeds for the random number generator.",
)
@click.option(
    "--datasets",
    type=str,
    default="xsum,cnn_dailymail,wikisum",
    help="The datasets to use. Use comma to separate multiple datasets.",
)
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="The number of samples to generate per dataset.",
)
@click.option(
    "--num-permutations",
    type=int,
    default=1,
    help="Number of permutations to generate.",
)
@click.option(
    "--output-dir",
    type=str,
    default="./examples",
    help="The output directory.",
)
@click.option(
    "--tokenizer-name",
    type=str,
    default=DEFAULT_TOKENIZER_NAME,
    help="The tokenizer to use for tokenization.",
)
@click.option(
    "--output-filename",
    type=str,
    default="dataset_distributions.json",
    help="The output filename for the dataset distributions JSON.",
)
def main(
    seeds: tuple[int],
    datasets: str,
    num_samples: int,
    num_permutations: int,
    output_dir: str,
    tokenizer_name: str,
    output_filename: str
) -> None:
    """Generate or load the dataset length distribution."""

    seeds_list = list(seeds)
    dataset_list = datasets.split(",")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {}

    for dataset_key in dataset_list:
        logger.info(f"Processing dataset: {dataset_key}")
        try:
            dataset_result = sample_dataset_lengths(
                    dataset_key=dataset_key,
                    seeds=seeds_list,
                    num_samples=num_samples,
                    num_permutations=num_permutations,
                    upper_bound=SEQ_LEN_UPPER_BOUND,
                    tokenizer_name=tokenizer_name,
                )
            result[dataset_key] = dataset_result
        except KeyError as e:
            logger.error(f"Error processing dataset {dataset_key}: {e}")

    json_path = output_path / output_filename
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False)
    logger.success(f"Results saved to {json_path}")

if __name__ == "__main__":
    main()

'''
python -m Cappuccino.generate_datasets \
    --num-samples 1000 \
    --output-filename Cappuccino_debug.json \
    --datasets gsm8k,wikisum,cnn_dailymail
'''