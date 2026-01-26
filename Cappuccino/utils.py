from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union, Dict, Literal
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from loguru import logger
from dataclasses import dataclass, field
SequenceBatchLayout = Literal["padded", "ragged"]
BatchComputeLayout  = Literal["batched", "packed"]
# Note:
# - SequenceBatchLayout: how raw samples are organized inside a batch (storage layout).
# - BatchComputeLayout: how batched tokens are laid out when passed through the model (compute layout).
ADAPTER_PADDING_MULTIPLE = 64

def load_dataset_list(dataset_path: Optional[Union[str, Path]] = None) -> List[Tuple[str, int, int]]:
    """
    Read a dataset.json and return:
      dataset_list = [(dataset_name, seed_idx, perm_idx), ...]
    where seed_idx and perm_idx are 0-based indices.

    If dataset_path is None, uses ./dataset.json by default.
    """
    path = Path(dataset_path) if dataset_path is not None else (Path.cwd() / "examples" / "dataset_distributions_16all_4096_seqlen_42_seed_1000_samples.json")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_list: List[Tuple[str, int, int]] = []
    for dataset_name, cfg in data.items():
        seeds = cfg.get("seeds", [])
        for seed_idx, seed in enumerate(seeds):
            seed_key = f"seed_{seed}"
            seed_cfg = cfg.get(seed_key, {})
            num_perms = int(seed_cfg.get("num_permutations", 0))
            for perm_idx in range(num_perms):
                dataset_list.append((dataset_name, seed_idx, perm_idx))

    return dataset_list

def json_utils_default(obj: Any) -> str:  # noqa: ANN401
    """Serialize an object to a JSON string.

    Args:
        obj: The object to serialize

    Returns:
        The serialized JSON string.
    """
    if isinstance(obj, set):
        return list(obj)
    return obj

def list_of_ints(arg: str) -> list[int]:
    """Parse a string of ints separated by spaces.

    It can be used as an argument parser type.

    """
    try:
        return [int(x) for x in arg.replace(",", " ").split()]
    except ValueError as e:
        msg = (
            f"Invalid list of ints: {arg}. "
            "Must be a string of ints separated by spaces."
        )
        raise argparse.ArgumentTypeError(msg) from e

def list_of_floats(arg: str) -> list[float]:
    """Parse a string of floats separated by spaces.

    It can be used as an argument parser type.

    """
    try:
        return [float(x) for x in arg.replace(",", " ").split()]
    except ValueError as e:
        msg = (
            f"Invalid list of floats: {arg}. "
            "Must be a string of floats separated by spaces."
        )
        raise argparse.ArgumentTypeError(msg) from e

def set_seed(seed: int) -> None:
    """Set the seed for the random number generator."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

def update_loguru_level(level: str) -> None:
    """Update the loguru level."""
    if level.upper() not in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]:
        msg = (
            f"Invalid loguru level: {level}. "
            f"Must be one of: INFO, DEBUG, WARNING, ERROR, CRITICAL."
        )
        raise ValueError(msg)
    logger.remove()
    logger.add(sys.stderr, level=level)


def stringify_keys(obj: Any) -> Any:  # noqa: ANN401
    """Serialize the keys of a dictionary to strings.

    Args:
        obj: The object to serialize
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            new_k = ", ".join(map(str, k)) if isinstance(k, tuple | list) else k
            new[new_k] = stringify_keys(v)
        return new
    if isinstance(obj, list | tuple):
        return [stringify_keys(i) for i in obj]
    return obj

def _round_up(v: int, multiple: int) -> int:
    return ((v + multiple - 1) // multiple) * multiple

# ========================= 
@dataclass
class MockDataArguments:
    """Arguments pertaining to the mock data."""

    dataset_path: str = field(
        default="",
        metadata={"help": "Path to the mock data."},
    )
    dataset_name: str = field(
        default="",
        metadata={"help": "Name of the dataset to use for the mock data."},
    )
    num_samples: int = field(
        default=1000,
        metadata={"help": "Number of samples to use for the mock data."},
    )
    seed_idx: int = field(
        default=0,
        metadata={"help": "Seed index to use for the mock data."},
    )
    permutation_idx: int = field(
        default=0,
        metadata={"help": "Permutation index to use for the mock data."},
    )
    multi_lora_dataset_schedule_path: str = field(
        default="",
        metadata={"help": "Path to the multi-LoRA dataset schedule."},
    )

    # 是否使用一个固定长度的虚拟数据集
    use_dummy_fixed_length_dataset: bool = field(
        default=False,
        metadata={"help": "Use a dummy fixed-length dataset."},
    )
    dummy_fixed_length_dataset_length: int = field(
        default=1024,
        metadata={"help": "Length of the dummy fixed-length dataset."},
    )

    # 采用构造 batch 的方式
    padding_mode: str = field(
        default="per_adapter_roundup",
        metadata={
            "help": (
                "Padding strategy when constructing synthetic batches. "
                "'per_adapter_roundup': original behavior, sum tokens per adapter "
                "and round up by adapter_padding_multiple; "
                "'max_length': pad every sample in a micro-batch to the maximum "
                "sequence length L_max within that micro-batch."
            )
        },
    )

class MockDataset:
    """A mock dataset that returns a fixed set of length of sequences."""

    # 样本序列长度列表
    def __init__(self, sample_lengths: list[int]) -> None:
        """Initialize the MockDataset with a fixed set of length of sequences.

        Args:
            sample_lengths: The length of the sequences to sample.
        """
        self.sample_lengths = sample_lengths
        self.stats: dict[str, float] = {
            "median": np.median(self.sample_lengths),
            "mean": np.mean(self.sample_lengths),
            "std": np.std(self.sample_lengths),
            "min": np.min(self.sample_lengths),
            "max": np.max(self.sample_lengths),
        }

    # 返回样本数
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.sample_lengths)

    # 返回指定索引或切片的样本序列长度
    def __getitem__(self, idx: int | slice) -> list[int] | int:
        """Get the sample at the specified index or slice.

        Args:
            idx: The index or slice to get.

        Returns:
            The sample(s) at the specified index or slice.
        """
        return self.sample_lengths[idx]

    # 打乱样本序列长度的顺序
    def shuffle(self, seed: int | None = None) -> None:
        """Shuffle the dataset."""
        rng = np.random.default_rng(seed)
        self.sample_lengths = rng.permutation(self.sample_lengths).tolist()


    # 返回指定数据集（dataset_name，seed_idx，permutation_idx ）的样本长度列表
    @classmethod
    def from_dataset_args(cls, dataset_args: MockDataArguments) -> MockDataset:
        """Initialize the MockDataset with a fixed set of length of sequences.

        Args:
            dataset_args: The dataset arguments.
        """
        with Path(dataset_args.dataset_path).open("r") as f:
            dataset_distributions = json.load(f)

        seed_to_data_dict = dataset_distributions[dataset_args.dataset_name]
        # print(f'dataset_args.seed_idx: {dataset_args.seed_idx}')
        seed = seed_to_data_dict["seeds"][dataset_args.seed_idx]
        data_dict = seed_to_data_dict[f"seed_{seed}"]
        sample_lengths = data_dict[f"permutation_{dataset_args.permutation_idx + 1}"]

        logger.success(
            f"Loaded {len(sample_lengths)} samples from {dataset_args.dataset_path}. "
            f"Dataset name: {dataset_args.dataset_name}, "
            f"Seed Index: {dataset_args.seed_idx}, "
            f"Seed: {seed}, "
            f"Permutation Index: {dataset_args.permutation_idx + 1}"
        )
        return cls(sample_lengths)

    @classmethod
    def from_a_fixed_length(cls, length: int, num_samples: int) -> MockDataset:
        """Initialize the MockDataset with a fixed set of length of sequences.

        Args:
            length: The length of the sequences to sample.
            num_samples: The number of samples to sample.
        """
        # 返回一个长度为 num_samples 的列表，列表中的每个元素都是 length
        # 即创建一个固定长度的数据集（所有样本长度相同）
        return cls([length] * num_samples)

@dataclass(slots=True)
class MicroBatchInfo:
    """Information about a micro-batch.

    Args:
        data_indices: List of (adapter_idx, global_batch_idx, sample_idx)
        micro_batch_idx: The index of the micro-batch in the schedule
    """

    # All following fields are sorted by the padded_total_tokens
    # > [adapter_idx, global_batch_idx, sample_idx, num_tokens]
    data_samples: dict[tuple[int, int, int], int]
    max_microbatch_size: int
    adapter_padding_multiple: int
    adapter_group_info: set[int] | None = None
    adapter_global_batch_idx_pairs: set[tuple[int, int]] | None = None
    adapter_token_lengths_pairs: dict[int, list[int]] | None = None
    adapter_num_samples_pairs: dict[int, int] | None = None
    adapter_num_tokens_pairs: dict[int, int] | None = None
    total_tokens: int | None = None
    padded_adapter_num_tokens_pairs: dict[int, int] | None = None
    padded_total_tokens: int | None = None
    # Fixed fields
    micro_batch_idx: int | None = None
    is_empty_marker: bool = False
    sequence_batch_layout: SequenceBatchLayout = "padded"
    batch_compute_layout: BatchComputeLayout = "packed"
    effictive_total_tokens: int | None = None

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self.self_update()

    def self_update(self) -> None:
        """Update the MicroBatchInfo object from the data_samples."""
        self.update_from_data_samples(self.data_samples)

    def update_from_data_samples(
        self, data_samples: dict[tuple[int, int, int], int]
    ) -> None:
        """Update the MicroBatchInfo object from the data_samples."""
        self.data_samples = data_samples

        # Update the adapter group info.
        self.adapter_group_info = {a_idx for a_idx, _, _ in data_samples}
        # 计算有效的总 token 数，用于计算吞吐量
        if self.effictive_total_tokens is None:
            self.effictive_total_tokens = sum(data_samples.values())

        # 如果 sequence_batch_layout 是 padded ,将该 micro-batch 的所有样本长度改为 max_length
        if self.sequence_batch_layout == "padded":
            max_len = max(data_samples.values())
            data_samples = {key: max_len for key in data_samples}
            self.data_samples = data_samples

        # Update the adapter global batch idx pairs.
        self.adapter_global_batch_idx_pairs = {
            (a_idx, global_batch_idx) for a_idx, global_batch_idx, _ in data_samples
        }

        # Update the adapter token lengths pairs.
        self.adapter_token_lengths_pairs = {
            target_adapter_idx: [
                tokens
                for (a_idx, _, _), tokens in data_samples.items()
                if a_idx == target_adapter_idx
            ]
            for target_adapter_idx in self.adapter_group_info
        }

        # Update the adapter num samples pairs.
        self.adapter_num_samples_pairs = {
            a_idx: len(samples)
            for a_idx, samples in self.adapter_token_lengths_pairs.items()
        }

        # Update the adapter num tokens pairs.
        self.adapter_num_tokens_pairs = {
            a_idx: sum(tokens)
            for a_idx, tokens in self.adapter_token_lengths_pairs.items()
        }

        # Total tokens.
        self.total_tokens = sum(self.adapter_num_tokens_pairs.values())

        # Padded adapter num tokens pairs.
        self.padded_adapter_num_tokens_pairs = {
            a_idx: _round_up(num_tokens, self.adapter_padding_multiple)
            for a_idx, num_tokens in self.adapter_num_tokens_pairs.items()
        }

        # Padded total tokens.
        self.padded_total_tokens = sum(self.padded_adapter_num_tokens_pairs.values())

    @classmethod
    def from_raw_data_indices(
        cls,
        raw_data_indices: list[tuple[int, int]],
        local_batch_data: list[list[int]],
        max_microbatch_size: int,
        adapter_padding_multiple: int = ADAPTER_PADDING_MULTIPLE,
        global_batch_idx: int | None = None,
        adapter_mapping: list[int] | None = None,
    ) -> MicroBatchInfo:
        """Create a MicroBatchInfo object from raw data indices."""
        num_adapters = len(local_batch_data)
        if adapter_mapping is None:
            adapter_mapping = list(range(num_adapters))
        if global_batch_idx is None:
            global_batch_idx = 0

        # data_samples collects all the samples in the micro-batch
        # contains list of (adapter_idx, global_batch_idx, sample_idx, num_tokens).
        data_samples = {
            (
                adapter_mapping[adapter_idx],
                global_batch_idx,
                sample_idx,
            ): local_batch_data[adapter_idx][sample_idx]
            for adapter_idx, sample_idx in raw_data_indices
        }

        return cls(
            data_samples=data_samples,
            max_microbatch_size=max_microbatch_size,
            adapter_padding_multiple=adapter_padding_multiple,
        )

    @classmethod
    def shedules_to_adapter_group_step_infos(
        cls,
        schedules: List[Tuple[List[Tuple[int, int, int]], List[int]]],
        aggregated_dataset: List[List[List[int]]],
        sequence_batch_layout: SequenceBatchLayout = "padded",
    ) -> List[AdapterGroupStepInfo]:
        """
        将调度格式转换为 AdapterGroupStepInfoFlexiblePadding 格式。

        Args:
            schedules:
                list[tuple[list[tuple[int, int, int]], list[int]]]
                    一个 tuple 表示一个微批次的信息：
                    - 第 0 项: 微批次中的样本列表
                        [(adapter_idx, global_batch_idx, sample_idx), ...]
                    - 第 1 项: 需要执行优化器步骤的适配器列表 [adapter_idx, ...]
                    （这里构造统计信息时忽略 optimizer_step_adapters）
            aggregated_dataset:
                形状为 [adapter_idx][global_batch_idx][sample_idx] -> num_tokens
            sequence_batch_layout:
                MicroBatch 构造方式：
                - "ragged": 保持原始样本长度不变
                - "padded": micro-batch 内按最大样本长度 padding

        Returns:
            List[AdapterGroupStepInfoFlexiblePadding]:
                每个元素对应一个 AdapterGroupStep（即一个 global_batch_idx）
        """
        print(f"Converting schedules to AdapterGroupStepInfo with data organization mode={sequence_batch_layout}...\n")
        print(f'来自 other/lorafusion\n')
        # 按 global_batch_idx 聚合 micro-batch 列表
        # key: global_batch_idx
        # value: List[MicroBatchInfoFlexiblePadding]
        group_by_global_batch: Dict[int, List[MicroBatchInfo]] = {}

        for micro_batch_idx, (sample_tuples, _optimizer_step_adapters) in enumerate(
            schedules
        ):
            # 如果这个 micro-batch 里没有样本，直接跳过
            if not sample_tuples:
                continue

            # 检查该 micro-batch 中的 global_batch_idx 是否唯一
            global_batch_indices = {gb_idx for _, gb_idx, _ in sample_tuples}
            if len(global_batch_indices) != 1:
                raise ValueError(
                    f"Each micro-batch is expected to have a single global_batch_idx, "
                    f"but got {global_batch_indices} for micro_batch_idx={micro_batch_idx}"
                )
            global_batch_idx = next(iter(global_batch_indices))

            # 构造 data_samples: (adapter_idx, global_batch_idx, sample_idx) -> num_tokens
            data_samples: Dict[tuple[int, int, int], int] = {}
            total_tokens_no_padding = 0

            for adapter_idx, gb_idx, sample_idx in sample_tuples:
                num_tokens = aggregated_dataset[adapter_idx][gb_idx][sample_idx]
                data_samples[(adapter_idx, gb_idx, sample_idx)] = num_tokens
                total_tokens_no_padding += num_tokens

            # 构造 MicroBatchInfoFlexiblePadding
            micro_batch_info = MicroBatchInfo(
                data_samples=data_samples,
                # 这里 max_microbatch_size 主要是一个“容量参考”，
                # 用总 token 数作为上限即可；如果后续不做 partial_merge，不会真正用到它。
                max_microbatch_size=total_tokens_no_padding,
                adapter_padding_multiple=ADAPTER_PADDING_MULTIPLE,
                micro_batch_idx=micro_batch_idx,
                sequence_batch_layout=sequence_batch_layout,
            )

            # 加入对应的 global_batch_idx 分组
            if global_batch_idx not in group_by_global_batch:
                group_by_global_batch[global_batch_idx] = []
            group_by_global_batch[global_batch_idx].append(micro_batch_info)

        # 将每个 global_batch_idx 对应的一组 micro-batch 转为 AdapterGroupStepInfoFlexiblePadding
        adapter_group_step_infos: List[AdapterGroupStepInfo] = []

        # dict 在 3.7+ 是保持插入顺序的，
        # 这里会按“第一次出现该 global_batch_idx 的顺序”产出 step。
        for _, micro_batch_list in group_by_global_batch.items():
            step_info = AdapterGroupStepInfo.from_micro_batch_infos_list(
                micro_batch_list
            )
            adapter_group_step_infos.append(step_info)

        return adapter_group_step_infos

@dataclass(slots=True)
class AdapterGroupStepInfo:
    """Information about an adapter group step."""

    adapter_group_info: set[int]
    internal_adapter_start_end_indices: dict[int, tuple[int, int]]
    micro_batch_infos: list[MicroBatchInfo]

    def self_update(self) -> None:
        """Update the AdapterGroupStepInfo object from the micro_batch_infos."""
        self.update_from_self_micro_batch_infos(self.micro_batch_infos)

    def update_from_self_micro_batch_infos(
        self, micro_batch_infos_list: list[MicroBatchInfo]
    ) -> None:
        """Create the object from a list of MicroBatchInfo objects."""
        # Handle the case where micro_batch_infos_list is empty
        if not micro_batch_infos_list:
            self.adapter_group_info = set()
            self.internal_adapter_start_end_indices = {}
            self.micro_batch_infos = []
            return

        adapter_group_info: set[int] = micro_batch_infos_list[0].adapter_group_info
        # Sort the micro_batch_infos_list by the padded_total_tokens
        # preserve the order if padded_total_tokens is the same, using descending order
        micro_batch_infos_list.sort(
            key=lambda x: x.padded_total_tokens,
            reverse=True,
        )
        internal_adapter_start_end_indices: dict[int, tuple[int, int]] = {
            adapter_idx: (-1, -1) for adapter_idx in adapter_group_info
        }
        used_adapters_list = [
            list(micro_batch_info.adapter_num_tokens_pairs.keys())
            for micro_batch_info in micro_batch_infos_list
        ]

        # Track the start and end indices for each adapter
        for adapter_idx in adapter_group_info:
            # Find the first microbatch that uses this adapter
            for i, used_adapters in enumerate(used_adapters_list):
                if adapter_idx in used_adapters:
                    internal_adapter_start_end_indices[adapter_idx] = (i, -1)
                    break

            # Find the last microbatch that uses this adapter
            for i in range(len(used_adapters_list) - 1, -1, -1):
                if adapter_idx in used_adapters_list[i]:
                    # Update the end index in the tuple
                    start_idx = internal_adapter_start_end_indices[adapter_idx][0]
                    internal_adapter_start_end_indices[adapter_idx] = (start_idx, i)
                    break
        self.adapter_group_info = adapter_group_info
        self.internal_adapter_start_end_indices = internal_adapter_start_end_indices
        self.micro_batch_infos = micro_batch_infos_list

    @classmethod
    def from_micro_batch_infos_list(
        cls,
        micro_batch_infos_list: list[MicroBatchInfo],
    ) -> AdapterGroupStepInfo:
        """Create the object from a list of MicroBatchInfo objects."""
        adapter_group_step_info = cls(
            adapter_group_info=None,
            internal_adapter_start_end_indices=None,
            micro_batch_infos=None,
        )
        adapter_group_step_info.update_from_self_micro_batch_infos(
            micro_batch_infos_list
        )
        return adapter_group_step_info



# =========================
#   Adapter config loader
# =========================

def load_adapter_config(adapter_config_path: str | Path) -> List[Dict[str, Any]]:
    """Load adapter_config.json.

    Expected formats:
      {"adapters": [ {...}, {...} ]}
    or directly:
      [ {...}, {...} ]
    """
    path = Path(adapter_config_path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "adapters" in obj:
        adapters = obj["adapters"]
    elif isinstance(obj, list):
        adapters = obj
    else:
        raise ValueError(
            f"Invalid adapter_config format at {path}. Expect dict with 'adapters' or a list."
        )
    if not isinstance(adapters, list) or not adapters:
        raise ValueError(f"adapter_config contains no adapters: {path}")
    return adapters