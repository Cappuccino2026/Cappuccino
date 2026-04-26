from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
from loguru import logger

from .profiler.mem_cost_model import MemCostModel
from .profiler.time_cost_model import TimeCostModel
from .utils import (
    AdapterGroupStepInfo,
    MicroBatchInfo,
    MockDataArguments,
    MockDataset,
    json_utils_default,
    list_of_ints,
    load_dataset_list,
    stringify_keys,
)

SampleRef = Tuple[int, int, int]
MicroBatch = Tuple[List[SampleRef], List[int]]
Schedule = List[MicroBatch]

GPU_MEMORY_LIMIT_MAP: Dict[str, float] = {
    "a100-40gb": 40.0,
    "a100-80gb": 80.0,
}

GPU_TYPE = "a100-40gb"
DEFAULT_GPU_MEMORY_LIMIT = GPU_MEMORY_LIMIT_MAP.get(GPU_TYPE, 40.0)
MEM_PROFILE_DATA_PATH = "profile_pp_combined.csv"
TIME_PROFILE_DATA_PATH = "profile_pp_combined.csv"
DEFAULT_ADAPTER_RANK = 16
LEGACY_MEMORY_CHECK_ADAPTER_RANK = 8


@dataclass
class SimpleMicroBatchInfo:
    """Metadata for a micro-batch candidate used by the planner."""

    original_index: int
    max_length: int
    total_tokens: int
    microbatch_size: int
    adapter_count: int
    samples: List[SampleRef]
    optimizer_steps: List[int]


def save_schedule(schedule: List[AdapterGroupStepInfo], path: str, output_name: str) -> None:
    """Save a schedule in both pickle and JSON formats."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule_pickle_path = output_dir / f"{output_name}_schedule.pkl"
    schedule_json_path = output_dir / f"{output_name}_schedule.json"

    with schedule_pickle_path.open("wb") as f:
        pickle.dump(schedule, f)
    logger.info("Schedule pickle saved: {}", schedule_pickle_path)

    with schedule_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [stringify_keys(asdict(item)) for item in schedule],
            f,
            default=json_utils_default,
            indent=2,
        )
    logger.info("Schedule JSON saved: {}", schedule_json_path)


class PipelineExecutionConstruction:
    """Construct Cappuccino-optimized pipeline execution schedules."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> None:
        self.model_name = model_name

        # Reserved for future pp_size-specific memory models.
        self.mem_models: Dict[int, Any] = {}

        # Single-GPU profile-based cost models.
        self.mem_model: Optional[Any] = None
        self.time_model: Optional[Any] = None
        self.precomputed_times: List[List[float]] = []
        self.is_padding = False

        self._initialize_time_model()
        self._initialize_mem_model()

        logger.info(
            "PipelineExecutionConstruction initialized | model={}",
            self.model_name,
        )

    def _initialize_time_model(self) -> None:
        """Initialize the single-GPU execution-time cost model."""
        try:
            self.time_model = TimeCostModel(
                model_name=self.model_name,
                csv_name=TIME_PROFILE_DATA_PATH,
            )
            logger.info("TimeCostModel initialized | model={}", self.model_name)
        except Exception as exc:  # pragma: no cover - depends on local profile files.
            logger.warning("Failed to initialize TimeCostModel | error={}", exc)

    def _initialize_mem_model(self) -> None:
        """Initialize the single-GPU memory cost model."""
        try:
            self.mem_model = MemCostModel(
                model_name=self.model_name,
                csv_name=MEM_PROFILE_DATA_PATH,
            )
            logger.info("MemCostModel initialized | model={}", self.model_name)
        except Exception as exc:  # pragma: no cover - depends on local profile files.
            logger.warning("Failed to initialize MemCostModel | error={}", exc)

    def step_level_check_if_fit_memory(
        self,
        batch_microbatches: Schedule,
        aggregated_dataset: List[List[List[int]]],
        pp_size: int = 4,
        gpu_memory_limit: float = DEFAULT_GPU_MEMORY_LIMIT,
    ) -> bool:
        """Return whether all micro-batches in a batch fit the memory budget."""
        if not batch_microbatches:
            return True
        if self.mem_model is None:
            raise RuntimeError("MemCostModel is not initialized.")

        for microbatch_samples, _ in batch_microbatches:
            combined_mbs = len(microbatch_samples)
            combined_max_length = 0
            adapter_set = set()

            for adapter_idx, batch_idx, sample_idx in microbatch_samples:
                sample_length = aggregated_dataset[adapter_idx][batch_idx][sample_idx]
                combined_max_length = max(combined_max_length, sample_length)
                adapter_set.add(adapter_idx)

            combined_adapter_count = len(adapter_set)
            combined_rank = combined_adapter_count * LEGACY_MEMORY_CHECK_ADAPTER_RANK
            mem_usage = self.mem_model.stage_memory_estimate(
                bsz=combined_mbs,
                seqlen=combined_max_length,
                rank=combined_rank,
                tasknum=combined_adapter_count,
                pp_size=pp_size,
            )
            if mem_usage > gpu_memory_limit:
                return False

        return True

    def job_level_check_if_fit_memory(
        self,
        micro_batchsize: int,
        seq_length: int,
        rank: int,
        pp_size: int,
        gpu_memory_limit: float = DEFAULT_GPU_MEMORY_LIMIT,
    ) -> bool:
        """Return whether a single job configuration fits the memory budget."""
        if self.mem_model is None:
            raise RuntimeError("MemCostModel is not initialized.")

        mem_usage = self.mem_model.stage_memory_estimate(
            bsz=micro_batchsize,
            seqlen=seq_length,
            rank=rank,
            pp_size=pp_size,
            tasknum=1,
        )
        return mem_usage <= gpu_memory_limit

    def job_leval_check_if_fit_memory(self, *args: Any, **kwargs: Any) -> bool:
        """Backward-compatible alias for the previous misspelled method name."""
        return self.job_level_check_if_fit_memory(*args, **kwargs)

    def _build_sorted_microbatch_infos(
        self,
        batch_microbatches: Schedule,
        aggregated_dataset: List[List[List[int]]],
    ) -> List[SimpleMicroBatchInfo]:
        """Build micro-batch metadata and sort it by maximum sequence length."""
        microbatch_infos: List[SimpleMicroBatchInfo] = []
        for microbatch_idx, (microbatch_samples, optimizer_steps) in enumerate(batch_microbatches):
            max_length = 0
            adapter_set = set()
            for adapter_idx, batch_idx, sample_idx in microbatch_samples:
                sample_length = aggregated_dataset[adapter_idx][batch_idx][sample_idx]
                max_length = max(max_length, sample_length)
                adapter_set.add(adapter_idx)

            microbatch_infos.append(
                SimpleMicroBatchInfo(
                    original_index=microbatch_idx,
                    max_length=max_length,
                    total_tokens=max_length * len(microbatch_samples),
                    microbatch_size=len(microbatch_samples),
                    adapter_count=len(adapter_set),
                    samples=microbatch_samples,
                    optimizer_steps=optimizer_steps,
                )
            )

        return sorted(microbatch_infos, key=lambda item: item.max_length)

    def _partition_objective(self, segment_costs: List[float], pp_size: int) -> float:
        """Compute the Cappuccino partition objective for a packed schedule."""
        num_segments = len(segment_costs)
        if num_segments <= 0:
            return float("inf")

        max_segment_cost = max(segment_costs)
        iteration_time = (pp_size - 1) * max_segment_cost + sum(segment_costs)
        return iteration_time * (num_segments + pp_size - 1) / num_segments

    def _cappuccino_pipeline_planner(
        self,
        batch_microbatches: Schedule,
        aggregated_dataset: List[List[List[int]]],
        pp_size: int = 4,
        gpu_memory_limit: float = DEFAULT_GPU_MEMORY_LIMIT,
        is_return_cappuccino_without_reorder: bool = True,
    ) -> Union[Tuple[Schedule, Schedule, Optional[float]], Tuple[Schedule, Optional[float]]]:
        """Optimize the micro-batch schedule for one logical batch."""
        if not batch_microbatches:
            if is_return_cappuccino_without_reorder:
                return [], [], None
            return [], None

        start_time = time.time()
        sorted_microbatch_infos = self._build_sorted_microbatch_infos(
            batch_microbatches=batch_microbatches,
            aggregated_dataset=aggregated_dataset,
        )
        num_microbatches = len(sorted_microbatch_infos)

        precomputed_times = self._get_precomputed_times(
            sorted_microbatch_infos=sorted_microbatch_infos,
            aggregated_dataset=aggregated_dataset,
            pp_size=pp_size,
            gpu_memory_limit=gpu_memory_limit,
        )
        t_max_candidates = self._get_t_max_candidates(precomputed_times)

        if not t_max_candidates:
            logger.warning(
                "No feasible t_max candidate found; returning the original batch schedule."
            )
            if is_return_cappuccino_without_reorder:
                return batch_microbatches, batch_microbatches, None
            return batch_microbatches, None

        min_iteration_time = float("inf")
        best_partition: Optional[List[Tuple[int, int]]] = None
        best_t_max = 0.0

        for t_max in t_max_candidates:
            inf = float("inf")
            dp = [[inf] * (num_microbatches + 1) for _ in range(num_microbatches + 1)]
            prev = [[-1] * (num_microbatches + 1) for _ in range(num_microbatches + 1)]
            dp[0][0] = 0.0

            for group_count in range(1, num_microbatches + 1):
                for end_idx in range(1, num_microbatches + 1):
                    start_min = group_count - 1
                    if start_min > end_idx - 1:
                        continue

                    best_val = inf
                    best_start = -1
                    for start_idx in range(start_min, end_idx):
                        segment_cost = precomputed_times[start_idx][end_idx - 1]
                        if segment_cost > t_max:
                            continue

                        previous_cost = dp[group_count - 1][start_idx]
                        if previous_cost == inf:
                            continue

                        candidate_cost = previous_cost + segment_cost
                        if candidate_cost < best_val:
                            best_val = candidate_cost
                            best_start = start_idx

                    dp[group_count][end_idx] = best_val
                    prev[group_count][end_idx] = best_start

            for group_count in range(1, num_microbatches + 1):
                if dp[group_count][num_microbatches] == float("inf"):
                    continue

                iteration_time = (pp_size - 1) * t_max + dp[group_count][num_microbatches]
                overall_iteration_time = iteration_time * (group_count + pp_size - 1) / group_count

                if overall_iteration_time >= min_iteration_time:
                    continue

                partition: List[Tuple[int, int]] = []
                remaining_groups = group_count
                end_idx = num_microbatches
                feasible = True

                while remaining_groups > 0 and end_idx >= 0:
                    start_idx = prev[remaining_groups][end_idx]
                    if start_idx < 0:
                        feasible = False
                        break
                    partition.append((start_idx, end_idx))
                    end_idx = start_idx
                    remaining_groups -= 1

                if not feasible or end_idx != 0 or len(partition) != group_count:
                    continue

                partition.reverse()
                min_iteration_time = overall_iteration_time
                best_partition = partition
                best_t_max = t_max

        if best_partition is None:
            logger.warning(
                "No feasible partition found; returning the original batch schedule."
            )
            if is_return_cappuccino_without_reorder:
                return batch_microbatches, batch_microbatches, None
            return batch_microbatches, None

        optimized_schedule_without_reorder: Schedule = []
        packed_microbatch_times: List[float] = []

        for group_start, group_end in best_partition:
            combined_samples: List[SampleRef] = []
            combined_optimizer_steps: List[int] = []

            for microbatch_idx in range(group_start, group_end):
                microbatch_info = sorted_microbatch_infos[microbatch_idx]
                combined_samples.extend(microbatch_info.samples)
                combined_optimizer_steps.extend(microbatch_info.optimizer_steps)

            optimized_schedule_without_reorder.append(
                (combined_samples, sorted(set(combined_optimizer_steps)))
            )
            packed_microbatch_times.append(precomputed_times[group_start][group_end - 1])

        optimized_schedule_with_reorder = self._reorder_optimized_schedule(
            optimized_schedule=optimized_schedule_without_reorder,
            packed_microbatch_times=packed_microbatch_times,
            pp_size=pp_size,
        )

        logger.debug(
            "Batch planning completed | groups={} | best_t_max={:.6f} | "
            "estimated_iteration_time={:.6f} | planner_overhead_sec={:.4f}",
            len(best_partition),
            best_t_max,
            min_iteration_time,
            time.time() - start_time,
        )

        if is_return_cappuccino_without_reorder:
            return optimized_schedule_without_reorder, optimized_schedule_with_reorder, min_iteration_time
        return optimized_schedule_with_reorder, min_iteration_time

    def _reorder_optimized_schedule(
        self,
        optimized_schedule: Schedule,
        packed_microbatch_times: List[float],
        pp_size: int = 4,
    ) -> Schedule:
        """Apply a V-shaped order to packed micro-batches within one batch.

        The smallest micro-batch is placed first, then subsequent micro-batches are
        alternately placed on the left and right sides. The contents and optimizer
        steps of each packed micro-batch are unchanged.
        """
        del pp_size  # The current reordering strategy is independent of pp_size.

        if not optimized_schedule or not packed_microbatch_times:
            return optimized_schedule

        num_items = len(optimized_schedule)
        if len(packed_microbatch_times) != num_items:
            logger.warning(
                "Skip schedule reordering because length mismatch | schedule_len={} | time_len={}",
                num_items,
                len(packed_microbatch_times),
            )
            return optimized_schedule

        items = [
            (idx, packed_time, optimized_schedule[idx])
            for idx, packed_time in enumerate(packed_microbatch_times)
        ]
        items.sort(key=lambda item: (item[1], item[0]))

        left, right = 0, num_items - 1
        placed: List[Optional[Tuple[float, MicroBatch]]] = [None] * num_items
        place_left = True

        for _, packed_time, microbatch in items:
            if place_left:
                placed[left] = (packed_time, microbatch)
                left += 1
            else:
                placed[right] = (packed_time, microbatch)
                right -= 1
            place_left = not place_left

        return [microbatch for item in placed if item is not None for _, microbatch in [item]]

    def _get_precomputed_times(
        self,
        sorted_microbatch_infos: List[SimpleMicroBatchInfo],
        aggregated_dataset: List[List[List[int]]],
        pp_size: int,
        gpu_memory_limit: float,
    ) -> List[List[float]]:
        """Precompute execution-time estimates for all contiguous packed groups."""
        if self.mem_model is None:
            raise RuntimeError("MemCostModel is not initialized.")
        if self.time_model is None:
            raise RuntimeError("TimeCostModel is not initialized.")

        num_microbatches = len(sorted_microbatch_infos)
        precomputed_times = [
            [float("inf")] * num_microbatches for _ in range(num_microbatches)
        ]

        for start_idx in range(num_microbatches):
            for end_idx in range(start_idx, num_microbatches):
                combined_mbs = 0
                combined_max_length = 0
                adapter_set = set()

                if self.is_padding:
                    for idx in range(start_idx, end_idx + 1):
                        microbatch_info = sorted_microbatch_infos[idx]
                        combined_mbs += microbatch_info.microbatch_size
                        combined_max_length = max(
                            combined_max_length,
                            microbatch_info.max_length,
                        )
                        for adapter_idx, _, _ in microbatch_info.samples:
                            adapter_set.add(adapter_idx)
                else:
                    # Ragged packing models all samples as one packed sequence.
                    combined_mbs = 1
                    total_length = 0
                    for idx in range(start_idx, end_idx + 1):
                        microbatch_info = sorted_microbatch_infos[idx]
                        for adapter_idx, batch_idx, sample_idx in microbatch_info.samples:
                            sample_length = aggregated_dataset[adapter_idx][batch_idx][sample_idx]
                            total_length += sample_length
                            adapter_set.add(adapter_idx)
                    combined_max_length = total_length

                combined_adapter_count = len(adapter_set)
                combined_rank = combined_adapter_count * DEFAULT_ADAPTER_RANK

                try:
                    mem_usage = self.mem_model.stage_memory_estimate(
                        bsz=combined_mbs,
                        seqlen=combined_max_length,
                        rank=combined_rank,
                        tasknum=combined_adapter_count,
                        pp_size=pp_size,
                    )
                except Exception as exc:
                    logger.warning(
                        "Memory estimation failed; skip packed group | range=[{}, {}] | error={}",
                        start_idx,
                        end_idx,
                        exc,
                    )
                    continue

                if mem_usage > gpu_memory_limit:
                    continue

                try:
                    precomputed_times[start_idx][end_idx] = (
                        self.time_model.stage_execution_time_estimate(
                            micro_bsz=combined_mbs,
                            seqlen=combined_max_length,
                            rank=combined_rank,
                            tasknum=combined_adapter_count,
                            pp_size=pp_size,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Time estimation failed; skip packed group | range=[{}, {}] | error={}",
                        start_idx,
                        end_idx,
                        exc,
                    )

        self.precomputed_times = precomputed_times
        return precomputed_times

    def _get_t_max_candidates(self, precomputed_times: List[List[float]]) -> List[float]:
        """Return sorted feasible maximum-stage-time candidates."""
        candidates = set()
        for start_idx in range(len(precomputed_times)):
            for end_idx in range(start_idx, len(precomputed_times)):
                value = precomputed_times[start_idx][end_idx]
                if value != float("inf"):
                    candidates.add(value)
        return sorted(candidates)

    def _generate_mlora_schedule(
        self,
        aggregated_dataset: List[List[List[int]]],
        adapter_to_microbatch_size: List[int],
        max_num_batches_to_schedule: int = 10000,
        **_: Any,
    ) -> Schedule:
        """Generate an mLoRA schedule with per-adapter micro-batch sizes.

        ``aggregated_dataset[i][j][k]`` is the token length of sample ``k`` in
        batch ``j`` for adapter ``i``. This method only uses it to recover the
        nested sample-index structure.
        """
        num_adapters = len(aggregated_dataset)
        if len(adapter_to_microbatch_size) != num_adapters:
            raise ValueError(
                "adapter_to_microbatch_size length mismatch: "
                f"expected={num_adapters}, actual={len(adapter_to_microbatch_size)}"
            )

        schedule: Schedule = []
        num_batches_in_each_adapter = [len(dataset) for dataset in aggregated_dataset]
        max_num_batches = max(num_batches_in_each_adapter) if num_adapters > 0 else 0

        for batch_idx in range(max_num_batches):
            for adapter_idx in range(num_adapters):
                if batch_idx >= num_batches_in_each_adapter[adapter_idx]:
                    continue

                batch = aggregated_dataset[adapter_idx][batch_idx]
                num_samples_in_batch = len(batch)
                microbatch_size = adapter_to_microbatch_size[adapter_idx]
                if microbatch_size <= 0:
                    raise ValueError(
                        f"Adapter {adapter_idx} has a non-positive micro-batch size: "
                        f"{microbatch_size}"
                    )

                num_microbatches = (num_samples_in_batch + microbatch_size - 1) // microbatch_size
                for microbatch_idx in range(num_microbatches):
                    start_sample_idx = microbatch_idx * microbatch_size
                    end_sample_idx = min(start_sample_idx + microbatch_size, num_samples_in_batch)
                    sample_indices = [
                        (adapter_idx, batch_idx, sample_idx)
                        for sample_idx in range(start_sample_idx, end_sample_idx)
                    ]
                    schedule.append((sample_indices, []))

                    if len(schedule) >= max_num_batches_to_schedule:
                        return schedule

                schedule[-1][1].append(adapter_idx)

        return schedule

    def generate_cappuccino_schedule(
        self,
        aggregated_dataset: List[List[List[int]]],
        adapter_to_microbatch_size: List[int],
        pp_size: int = 4,
        max_num_batches_to_schedule: int = 100000,
        is_return_cappuccino_without_reorder: bool = True,
    ) -> Union[Tuple[Schedule, Schedule, float], Tuple[Schedule, float]]:
        """Generate the final Cappuccino-optimized schedule."""
        mlora_schedule = self._generate_mlora_schedule(
            aggregated_dataset=aggregated_dataset,
            adapter_to_microbatch_size=adapter_to_microbatch_size,
            max_num_batches_to_schedule=max_num_batches_to_schedule,
        )

        batch_to_microbatch_indices: Dict[int, List[int]] = {}
        for microbatch_idx, (microbatch_samples, _) in enumerate(mlora_schedule):
            for _, batch_idx, _ in microbatch_samples:
                batch_to_microbatch_indices.setdefault(batch_idx, [])
                if microbatch_idx not in batch_to_microbatch_indices[batch_idx]:
                    batch_to_microbatch_indices[batch_idx].append(microbatch_idx)

        for batch_idx in batch_to_microbatch_indices:
            batch_to_microbatch_indices[batch_idx] = sorted(
                set(batch_to_microbatch_indices[batch_idx])
            )

        optimized_schedule: Schedule = []
        optimized_schedule_without_reorder: Schedule = []
        total_iteration_time = 0.0

        for batch_idx in sorted(batch_to_microbatch_indices):
            batch_microbatch_indices = batch_to_microbatch_indices[batch_idx]
            batch_microbatches = [mlora_schedule[idx] for idx in batch_microbatch_indices]

            if is_return_cappuccino_without_reorder:
                (
                    optimized_batch_without_reorder,
                    optimized_batch,
                    estimation_time,
                ) = self._cappuccino_pipeline_planner(
                    batch_microbatches=batch_microbatches,
                    aggregated_dataset=aggregated_dataset,
                    pp_size=pp_size,
                    is_return_cappuccino_without_reorder=True,
                )
                optimized_schedule_without_reorder.extend(optimized_batch_without_reorder)
                optimized_schedule.extend(optimized_batch)
            else:
                optimized_batch, estimation_time = self._cappuccino_pipeline_planner(
                    batch_microbatches=batch_microbatches,
                    aggregated_dataset=aggregated_dataset,
                    pp_size=pp_size,
                    is_return_cappuccino_without_reorder=False,
                )
                optimized_schedule.extend(optimized_batch)

            total_iteration_time += estimation_time if estimation_time is not None else 0.0

        logger.info(
            "Cappuccino schedule generated | batches={} | microbatches={} | "
            "optimized_microbatches={} | estimated_total_iteration_time={:.6f}",
            len(batch_to_microbatch_indices),
            len(mlora_schedule),
            len(optimized_schedule),
            total_iteration_time,
        )

        if is_return_cappuccino_without_reorder:
            return optimized_schedule_without_reorder, optimized_schedule, total_iteration_time
        return optimized_schedule, total_iteration_time


@click.command()
@click.option("--dataset_path", default="datasets/dataset_distributions.json", type=str)
@click.option("--num_adapters", default=4, type=int)
@click.option("--num_pipeline_stages", default=4, type=int)
@click.option("--adapter_to_dataset_idx", default="0,4,8,12", type=str)
@click.option("--adapter_to_global_batch_size", default="4,4,8,8", type=str)
@click.option("--microbatch_size", default=4, type=int)
@click.option(
    "--adapter_to_microbatch_size",
    default=None,
    type=str,
    help="Optional per-adapter micro-batch sizes, e.g., '1,2,4,1'.",
)
@click.option("--model_name", default="meta-llama/Llama-3.1-8B-Instruct", type=str)
@click.option("--gpu_memory_limit", default=40.0, type=float)
@click.option("--output_name", default="test_cappuccino", type=str)
@click.option("--max_num_batches", default=2, type=int)
def test_cappuccino_optimizer(
    dataset_path: str,
    num_adapters: int,
    num_pipeline_stages: int,
    adapter_to_dataset_idx: str,
    adapter_to_global_batch_size: str,
    microbatch_size: int,
    adapter_to_microbatch_size: Optional[str],
    model_name: str,
    gpu_memory_limit: float,
    output_name: str,
    max_num_batches: int,
) -> None:
    """CLI entry point for validating the Cappuccino pipeline planner."""
    del gpu_memory_limit  # Kept for CLI compatibility; planner uses the default unless overridden.

    adapter_to_dataset_idx_list = list_of_ints(adapter_to_dataset_idx)
    adapter_to_global_batch_size_list = list_of_ints(adapter_to_global_batch_size)

    if len(adapter_to_dataset_idx_list) != num_adapters:
        raise ValueError(
            f"--adapter_to_dataset_idx length must be {num_adapters}, "
            f"but got {len(adapter_to_dataset_idx_list)}."
        )
    if len(adapter_to_global_batch_size_list) != num_adapters:
        raise ValueError(
            f"--adapter_to_global_batch_size length must be {num_adapters}, "
            f"but got {len(adapter_to_global_batch_size_list)}."
        )

    if adapter_to_microbatch_size is None:
        adapter_to_microbatch_size_list = [microbatch_size] * num_adapters
    else:
        adapter_to_microbatch_size_list = list_of_ints(adapter_to_microbatch_size)
        if len(adapter_to_microbatch_size_list) != num_adapters:
            raise ValueError(
                f"--adapter_to_microbatch_size length must be {num_adapters}, "
                f"but got {len(adapter_to_microbatch_size_list)}."
            )

    if any(size <= 0 for size in adapter_to_microbatch_size_list):
        raise ValueError("All micro-batch sizes must be positive.")

    logger.info("=" * 72)
    logger.info("Starting Cappuccino pipeline planner validation")
    logger.info("=" * 72)
    logger.info("Per-adapter micro-batch sizes: {}", adapter_to_microbatch_size_list)

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    dataset_list = load_dataset_list(dataset_path=dataset_file)
    aggregated_dataset: List[List[List[int]]] = []

    for adapter_idx in range(num_adapters):
        dataset_name, seed_idx, permutation_idx = dataset_list[
            adapter_to_dataset_idx_list[adapter_idx]
        ]
        mock_data_args = MockDataArguments(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            seed_idx=seed_idx,
            permutation_idx=permutation_idx,
        )
        mock_dataset = MockDataset.from_dataset_args(mock_data_args)

        global_batch_size = adapter_to_global_batch_size_list[adapter_idx]
        adapter_dataset: List[List[int]] = []
        max_batches_for_adapter = min(len(mock_dataset) // global_batch_size, max_num_batches)

        for batch_idx in range(max_batches_for_adapter):
            batch = mock_dataset[
                batch_idx * global_batch_size : (batch_idx + 1) * global_batch_size
            ]
            adapter_dataset.append(batch)

        aggregated_dataset.append(adapter_dataset)

    constructor = PipelineExecutionConstruction(model_name=model_name)
    _, schedule, total_time = constructor.generate_cappuccino_schedule(
        aggregated_dataset=aggregated_dataset,
        adapter_to_microbatch_size=adapter_to_microbatch_size_list,
        pp_size=num_pipeline_stages,
        is_return_cappuccino_without_reorder=True,
    )

    output_dir = Path("schedules")
    output_dir.mkdir(exist_ok=True)

    new_schedule = MicroBatchInfo.shedules_to_adapter_group_step_infos(
        schedules=schedule,
        aggregated_dataset=aggregated_dataset,
        sequence_batch_layout="ragged",
    )
    save_schedule(
        schedule=new_schedule,
        path=str(output_dir),
        output_name=output_name,
    )

    logger.info(
        "Cappuccino planner validation completed | output_dir={} | "
        "estimated_total_iteration_time={:.6f}",
        output_dir,
        total_time,
    )


if __name__ == "__main__":
    test_cappuccino_optimizer()

"""
Example:
    python -m Cappuccino.PipelineExecutionConstruction \
      --dataset_path examples/dataset_distributions_16all_4096_seqlen_42_seed_1000_samples.json \
      --num_adapters 4 \
      --num_pipeline_stages 2 \
      --adapter_to_dataset_idx "0,4,8,12" \
      --adapter_to_global_batch_size "8,8,8,8" \
      --adapter_to_microbatch_size "1,2,4,1" \
      --model_name Llama-2-7b-hf \
      --gpu_memory_limit 40 \
      --output_name debug_cappuccino \
      --max_num_batches 2
"""