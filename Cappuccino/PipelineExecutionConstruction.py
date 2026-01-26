from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from dataclasses import asdict
import time
import os
import sys
import json
import pickle
import time
import contextlib
import io
from pathlib import Path
import click
from loguru import logger
import time
from .utils import (
    load_dataset_list,
    json_utils_default,
    list_of_floats,
    list_of_ints,
    set_seed,
    update_loguru_level,
    MockDataArguments,
    MockDataset,
    stringify_keys,
    AdapterGroupStepInfo,
    MicroBatchInfo
)
from .profiler.time_cost_model import TimeCostModel
from .profiler.mem_cost_model import MemCostModel

# GPU 显存大小映射
# 按百分之80 使用率计算可用显存
GPU_MEMORY_LIMIT_MAP = {
    'a100-40gb': 40.0,
    'a100-80gb': 80.0,
}

GPU_TYPE = 'a100-40gb'

MEM_PROFILE_DATA_PATH = 'profile_pp_combined.csv'
TIME_PROFILE_DATA_PATH = 'profile_pp_combined.csv'


@dataclass
class SimpleMicroBatchInfo:
    """微批次信息"""
    original_index: int  # 这个微批次在原 batch 的位置
    max_length: int
    total_tokens: int
    microbatch_size: int
    adapter_count: int
    samples: List[Tuple[int, int, int]]  # [(adapter_idx, batch_idx, sample_idx), ...]
    optimizer_steps: List[int]

def save_schedule(schedule: list[AdapterGroupStepInfo], path:str, output_name: str):
    schedule_pickle_path = f"{path}/{output_name}_schedule.pkl"
    schedule_json_path = f"{path}/{output_name}_schedule.json"
    with open(schedule_pickle_path, "wb") as f:
        pickle.dump(schedule, f)
    logger.info(f"Saved schedule pickle to {schedule_pickle_path}")
    with open(schedule_json_path, "w") as f:
        json.dump(
            [stringify_keys(asdict(s)) for s in schedule], f, default=json_utils_default, indent=2
        )
    logger.info(f"Saved schedule json to {schedule_json_path}")


class PipelineExecutionConstruction:

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name

        # 模型实例缓存（如果后面需要按 pp_size 细分）
        self.mem_models: Dict[int, any] = {}

        # 单卡 profile 得到的建模结果
        self.mem_model: Optional[any] = None
        self.time_model: Optional[any] = None

        # 初始化时间模型
        self._initialize_time_model()
        # 初始化内存模型
        self._initailize_mem_model()

        # time.sleep(10)
        self.is_padding = False


        print(f"PipelineExecutionConstruction initialized for model: {model_name}")

    def _initialize_time_model(self):
        """
        初始化时间成本模型
        这里的时间成本模型是在单卡上进行估计的
        """

        try:
            self.time_model = TimeCostModel(
                model_name=self.model_name,
                csv_name=TIME_PROFILE_DATA_PATH
            )
            print(f"Initialized TimeCostModel for {self.model_name} ")
        except Exception as e:
            print(f"[WARN] Failed to initialize TimeCostModel: {e}")

    def _initailize_mem_model(self):
        """
        初始化内存成本模型
        这里的内存成本模型是在单卡上进行估计的
        """

        try:
            self.mem_model = MemCostModel(
                model_name=self.model_name,
                csv_name=MEM_PROFILE_DATA_PATH
            )
            print(f"Initialized MemCostModel for {self.model_name} ")
        except Exception as e:
            print(f"[WARN] Failed to initialize MemCostModel: {e}")
            
    # step-level 检查一个 batch 中的微批次调度是否符合显存限制, 传入具体的微批次调度
    def step_level_check_if_fit_memory(
        self,
        batch_microbatches: List[Tuple[List[Tuple[int, int, int]], List[int]]],
        aggregated_dataset: List[List[List[int]]],
        pp_size: int = 4,
        gpu_memory_limit: float = GPU_MEMORY_LIMIT_MAP.get(GPU_TYPE, 40.0),
    ) -> bool:
        """检查给定的微批次调度是否符合显存限制"""
        if not batch_microbatches:
            return True

        for mb_idx, (microbatch_samples, optimizer_steps) in enumerate(batch_microbatches):
            # 一个微批次中样本的数量
            combined_mbs = len(microbatch_samples)
            combined_max_length = 0
            adapter_set = set()

            for adapter_idx, batch_idx, sample_idx in microbatch_samples:
                sample_length = aggregated_dataset[adapter_idx][batch_idx][sample_idx]
                combined_max_length = max(combined_max_length, sample_length)
                adapter_set.add(adapter_idx)

            combined_adapter_count = len(adapter_set)
            combined_rank = combined_adapter_count * 8  # 假设每个适配器 rank=8

            # 检查每个流水线阶段的显存使用，假设全都是均匀分配的
            mem_usage = self.mem_model.stage_memory_estimate(
                bsz=combined_mbs,
                seqlen=combined_max_length,
                rank=combined_rank,
                tasknum=combined_adapter_count,
                pp_size=pp_size,
            )
            if mem_usage > gpu_memory_limit:
                print(f"微批次 {mb_idx} 显存使用 {mem_usage:.2f}GB 超过限制 {gpu_memory_limit}GB")
                return False

        return True

    def job_leval_check_if_fit_memory(
        self,
        micro_batchsize: int,
        seq_length: int,
        rank: int,
        pp_size: int,
        gpu_memory_limit: float = GPU_MEMORY_LIMIT_MAP.get(GPU_TYPE, 40.0),
    ) -> bool:
        """检查给定的微批次参数是否符合显存限制"""

        # 假设每个阶段均匀分配显存
        mem_usage = self.mem_model.stage_memory_estimate(
            bsz=micro_batchsize,
            seqlen=seq_length,
            rank=rank,
            pp_size=pp_size,
            tasknum=1,
        )
        if mem_usage > gpu_memory_limit:
            print(f"显存使用 {mem_usage:.2f}GB 超过限制 {gpu_memory_limit}GB——micro_batchsize为{micro_batchsize},seqlen为{seq_length},ppsize为{pp_size}")
            return False
        return True

    def _build_sorted_microbatch_infos(
        self,
        batch_microbatches: List[Tuple[List[Tuple[int, int, int]], List[int]]],
        aggregated_dataset: List[List[List[int]]],
    ) -> List[SimpleMicroBatchInfo]:
        microbatch_infos: List[SimpleMicroBatchInfo] = []
        for mb_idx, (microbatch_samples, optimizer_steps) in enumerate(batch_microbatches):
            max_length = 0
            adapter_set = set()
            for adapter_idx, batch_idx, sample_idx in microbatch_samples:
                sample_length = aggregated_dataset[adapter_idx][batch_idx][sample_idx]
                max_length = max(max_length, sample_length)
                adapter_set.add(adapter_idx)
            total_tokens = max_length * len(microbatch_samples)
            microbatch_infos.append(
                SimpleMicroBatchInfo(
                    original_index=mb_idx,
                    max_length=max_length,
                    total_tokens=total_tokens,
                    microbatch_size=len(microbatch_samples),
                    adapter_count=len(adapter_set),
                    samples=microbatch_samples,
                    optimizer_steps=optimizer_steps,
                )
            )
        return sorted(microbatch_infos, key=lambda x: x.max_length)
    
    def _partition_objective(
        self,
        seg_costs: List[float],
        pp_size: int,
    ) -> float:
        """
        Keep identical objective to your Cappuccino code:
          current_iteration_time = (pp-1)*t_max + sum(seg_costs)
          overall = current_iteration_time * (m+p-1)/m
        """
        m = len(seg_costs)
        if m <= 0:
            return float("inf")
        t_max = max(seg_costs)
        current_iteration_time = (pp_size - 1) * t_max + sum(seg_costs)
        overall = current_iteration_time * (m + pp_size - 1) / m
        return overall
    
    def _cappuccino_pipeline_planner(
        self,
        batch_microbatches: List[Tuple[List[Tuple[int, int, int]], List[int]]],
        aggregated_dataset: List[List[List[int]]],
        pp_size: int = 4,
        gpu_memory_limit: float = GPU_MEMORY_LIMIT_MAP.get(GPU_TYPE, 40.0),
        is_return_cappuccino_without_reorder: bool = True,
    ):
        """
        使用 Cappuccino 算法优化单个 batch 的微批次调度

        Args:
            batch_microbatches: 单个 batch 的微批次列表
            aggregated_dataset: 聚合数据集 [适配器][批次][样本]
            pp_size: 流水线阶段数
            gpu_memory_limit: GPU 显存限制 (GB)

        Returns:
            优化后的微批次调度, 以及估计的 iteration time
        """

        if not batch_microbatches:
            return [], None

        print(f"开始优化 batch, 包含 {len(batch_microbatches)} 个微批次")

        start_time = time.time()

        # 1. 提取微批次信息并按样本长度排序
        microbatch_infos: List[SimpleMicroBatchInfo] = []
        for mb_idx, (microbatch_samples, optimizer_steps) in enumerate(batch_microbatches):
            max_length = 0
            adapter_set = set()

            for adapter_idx, batch_idx, sample_idx in microbatch_samples:
                sample_length = aggregated_dataset[adapter_idx][batch_idx][sample_idx]
                max_length = max(max_length, sample_length)
                adapter_set.add(adapter_idx)

            total_tokens = max_length * len(microbatch_samples)

            microbatch_info = SimpleMicroBatchInfo(
                original_index=mb_idx,
                max_length=max_length,
                total_tokens=total_tokens,
                microbatch_size=len(microbatch_samples),
                adapter_count=len(adapter_set),
                samples=microbatch_samples,
                optimizer_steps=optimizer_steps,
            )
            microbatch_infos.append(microbatch_info)

        # 按最大样本长度升序排序
        sorted_microbatch_infos = sorted(
            microbatch_infos,
            key=lambda x: x.max_length,
        )

        # 2. 构建预计算时间矩阵
        num_microbatches = len(sorted_microbatch_infos)
        precomputed_times = self._get_precomputed_times(
            sorted_microbatch_infos, aggregated_dataset, pp_size, gpu_memory_limit
        )

        # 3. 获取 t_max 候选值
        t_max_candidates = self._get_t_max_candidates(precomputed_times)

        if not t_max_candidates:
            print("警告：没有找到有效的时间候选值，使用原始调度")
            if is_return_cappuccino_without_reorder:
                return batch_microbatches, batch_microbatches, None
            return batch_microbatches, None

        # 4. 动态规划求解最优分组
        min_iteration_time = float('inf')
        best_partition = None
        best_t_max = 0.0

        for t_max in t_max_candidates:
            N = num_microbatches
            INF = float('inf')
            # dp2[k][i]: 覆盖前 i 个微批次、恰好分成 k 组的最小组内时间总和
            dp2 = [[INF] * (N + 1) for _ in range(N + 1)]
            prev = [[-1] * (N + 1) for _ in range(N + 1)]  # 回溯起点
            dp2[0][0] = 0.0

            for k in range(1, N + 1):
                for i in range(1, N + 1):
                    best_val = INF
                    best_j = -1
                    j_start = k - 1
                    if j_start > i - 1:
                        continue
                    for j in range(j_start, i):
                        seg_cost = precomputed_times[j][i - 1]  # 组 [j, i) 的代价
                        if seg_cost > t_max:
                            continue
                        prev_cost = dp2[k - 1][j]
                        if prev_cost == INF:
                            continue
                        cand = prev_cost + seg_cost
                        if cand < best_val:
                            best_val = cand
                            best_j = j
                    dp2[k][i] = best_val
                    prev[k][i] = best_j

            for k in range(1, N + 1):
                if dp2[k][N] == INF:
                    continue

                current_iteration_time = (pp_size - 1) * t_max + dp2[k][N]
                m = k  # 组数
                p = pp_size

                # 这里使用一个简单的折算因子 (m + p - 1) / m，保证 m 越大折算后越接近 current_iteration_time
                current_overall_iteration_time = current_iteration_time * (m + p - 1) / m

                if current_overall_iteration_time < min_iteration_time:
                    # 回溯该 k 的最优分段
                    partition = []
                    kk = k
                    i = N
                    feasible = True
                    while kk > 0 and i >= 0:
                        j = prev[kk][i]
                        if j < 0:
                            feasible = False
                            break
                        partition.append((j, i))
                        i = j
                        kk -= 1
                    if not feasible or i != 0 or len(partition) != k:
                        continue
                    partition.reverse()

                    min_iteration_time = current_overall_iteration_time
                    best_partition = partition
                    best_t_max = t_max

        if best_partition is None:
            print("警告：动态规划未找到有效分组，使用原始调度")
            if is_return_cappuccino_without_reorder:
                return batch_microbatches, batch_microbatches, None
            return batch_microbatches, None

        computation_time = time.time() - start_time

        # print(f"最优分组找到: t_max={best_t_max:.4f}, 迭代时间估计={min_iteration_time:.4f}")
        # print(f"分组方案: {best_partition}")
        # print(f"DP计算时间: {computation_time:.4f}秒")

        # 5. 重构优化后的微批次调度
        optimized_schedule_without_reoder: List[Tuple[List[Tuple[int, int, int]], List[int]]] = []
        packed_microbatch_times: List[float] = []

        for group_idx, (group_start, group_end) in enumerate(best_partition):
            combined_samples: List[Tuple[int, int, int]] = []
            combined_optimizer_steps: List[int] = []

            for mb_idx in range(group_start, group_end):
                mb = sorted_microbatch_infos[mb_idx]
                combined_samples.extend(mb.samples)
                combined_optimizer_steps.extend(mb.optimizer_steps)

            combined_optimizer_steps = list(set(combined_optimizer_steps))

            optimized_schedule_without_reoder.append((combined_samples, combined_optimizer_steps))
            packed_microbatch_times.append(precomputed_times[group_start][group_end - 1])

        # print("============================优化后的微批次调度情况============================")
        # for idx, (samples, optimizer_steps) in enumerate(optimized_schedule):
        #     lengths = [
        #         aggregated_dataset[adapter_idx][batch_idx][sample_idx]
        #         for adapter_idx, batch_idx, sample_idx in samples
        #     ]
        #     print(
        #         f"微批次 {idx}: 微批次大小={len(samples)}, 优化器步骤={optimizer_steps}, "
        #         f"预计算时间={packed_microbatch_times[idx]:.6f}, "
        #         f"批次样本长度=[{', '.join(str(l) for l in lengths)}], "
        #         f"批次最长样本长度={max(lengths) if lengths else 0}"
        #     )

        # print(f"优化完成: {len(batch_microbatches)} 个微批次 -> {len(optimized_schedule)} 个组合微批次")

        print("开始对优化后的调度进行重排序...")
        optimized_schedule_with_reorder = self._reorder_optimized_schedule(
            optimized_schedule_without_reoder,
            packed_microbatch_times,
            pp_size=pp_size,
        )

        if is_return_cappuccino_without_reorder:
            return optimized_schedule_without_reoder, optimized_schedule_with_reorder,min_iteration_time
        
        return optimized_schedule_with_reorder, min_iteration_time

    def _reorder_optimized_schedule(
        self,
        optimized_schedule: List[Tuple[List[Tuple[int, int, int]], List[int]]],
        packed_microbatch_times: List[float],
        pp_size: int = 4,  # 保留但本策略不使用
    ) -> List[Tuple[List[Tuple[int, int, int]], List[int]]]:
        """
        对“单个 batch 内”的微批次进行 V 型排序（两边小、中间大），且最小的放在最前面。
        仅改变微批次顺序；不改动每个微批的内容与 optimizer_steps。
        """
        if not optimized_schedule or not packed_microbatch_times:
            return optimized_schedule

        m = len(optimized_schedule)
        if len(packed_microbatch_times) != m:
            print(
                f"警告：packed_microbatch_times 长度 ({len(packed_microbatch_times)}) "
                f"与 optimized_schedule 长度 ({m}) 不匹配，跳过重排。"
            )
            return optimized_schedule

        items = [(idx, t, optimized_schedule[idx]) for idx, t in enumerate(packed_microbatch_times)]
        items.sort(key=lambda x: (x[1], x[0]))  # 小 -> 大；保证相同时间时保序

        left, right = 0, m - 1
        placed: List[Optional[Tuple[float, Tuple[List[Tuple[int, int, int]], List[int]]]]] = [None] * m
        place_left = True
        for _, t, mb in items:  # 从小到大
            if place_left:
                placed[left] = (t, mb)
                left += 1
            else:
                placed[right] = (t, mb)
                right -= 1
            place_left = not place_left

        reordered_schedule = [mb for (t, mb) in placed if mb is not None]
        return reordered_schedule

    def _get_precomputed_times(
        self,
        sorted_microbatch_infos: List[SimpleMicroBatchInfo],
        aggregated_dataset: List[List[List[int]]],
        pp_size: int,
        gpu_memory_limit: float,
    ) -> List[List[float]]:
        """计算预计算时间矩阵"""
        num_microbatches = len(sorted_microbatch_infos)
        precomputed_times = [[float('inf')] * num_microbatches for _ in range(num_microbatches)]

        for i in range(num_microbatches):
            for j in range(i, num_microbatches):
                combined_mbs = 0
                combined_max_length = 0
                adapter_set = set()

                if self.is_padding:
                    for k in range(i, j + 1):
                        mb = sorted_microbatch_infos[k]
                        combined_mbs += mb.microbatch_size
                        combined_max_length = max(combined_max_length, mb.max_length)
                        for adapter_idx, _, _ in mb.samples:
                            adapter_set.add(adapter_idx)
                else:
                    # 如果不采用padding, combined_mbs设为1, combined_max_length设为所有样本实际长度之和
                    combined_mbs = 1
                    total_length = 0
                    for k in range(i, j + 1):
                        mb = sorted_microbatch_infos[k]
                        for adapter_idx, batch_idx, sample_idx in mb.samples:
                            sample_length = aggregated_dataset[adapter_idx][batch_idx][sample_idx]
                            total_length += sample_length
                            adapter_set.add(adapter_idx)
                    combined_max_length = total_length


                combined_adapter_count = len(adapter_set)
                combined_rank = combined_adapter_count * 16  # 假设每个适配器 rank=16

                is_mem_valid = True
                try:
                    mem_usage = self.mem_model.stage_memory_estimate(
                        bsz=combined_mbs,
                        seqlen=combined_max_length,
                        rank=combined_rank,
                        tasknum=combined_adapter_count,
                        pp_size=pp_size,
                    )
                    if mem_usage > gpu_memory_limit:
                        is_mem_valid = False
                except Exception as e:
                    print(f"显存估算失败，跳过组合 [{i},{j}]: {e}")
                    is_mem_valid = False

                if is_mem_valid:
                    try:
                        exec_time = self.time_model.stage_execution_time_estimate(
                            micro_bsz=combined_mbs,
                            seqlen=combined_max_length,
                            rank=combined_rank,
                            tasknum=combined_adapter_count,
                            pp_size=pp_size,
                        )
                        precomputed_times[i][j] = exec_time
                    except Exception as e:
                        print(f"时间估算失败，跳过组合 [{i},{j}]: {e}")
                        precomputed_times[i][j] = float('inf')

        self.precomputed_times = precomputed_times
        return precomputed_times

    def _get_t_max_candidates(self, precomputed_times: List[List[float]]) -> List[float]:
        """获取 t_max 候选值"""
        t_max_candidates = set()
        for i in range(len(precomputed_times)):
            for j in range(i, len(precomputed_times)):
                if precomputed_times[i][j] != float('inf'):
                    t_max_candidates.add(precomputed_times[i][j])
        return sorted(t_max_candidates)

    def _generate_mlora_schedule(
        self,
        aggregated_dataset: List[List[List[int]]],
        adapter_to_microbatch_size: List[int],
        max_num_batches_to_schedule: int = 10000,
        **kwargs,
    ) -> List[Tuple[List[Tuple[int, int, int]], List[int]]]:
        """生成 mLoRA 调度，支持每个适配器不同的 micro-batch size。

        aggregated_dataset[i][j][k]:
            adapter i, batch j, sample k 的 token 长度（这里只用来拿索引结构）
        adapter_to_microbatch_size[i]:
            第 i 个适配器的 micro-batch size
        """
        num_adapters = len(aggregated_dataset)
        if len(adapter_to_microbatch_size) != num_adapters:
            raise ValueError(
                f"adapter_to_microbatch_size length {len(adapter_to_microbatch_size)} "
                f"does not match num_adapters {num_adapters}"
            )

        schedule: List[Tuple[List[Tuple[int, int, int]], List[int]]] = []
        num_batches_in_each_adapter = [len(dataset) for dataset in aggregated_dataset]
        max_num_batches = max(num_batches_in_each_adapter) if num_adapters > 0 else 0

        for j in range(max_num_batches):
            for i in range(num_adapters):
                # 当前适配器没有这么多 batch，跳过
                if j >= num_batches_in_each_adapter[i]:
                    continue

                batch = aggregated_dataset[i][j]
                num_samples_in_batch = len(batch)
                mbsz = adapter_to_microbatch_size[i]
                if mbsz <= 0:
                    raise ValueError(f"adapter {i} has non-positive micro-batch size {mbsz}")

                # 按各自适配器的 mbsz 切 micro-batch
                for microbatch_idx in range(
                    (num_samples_in_batch + mbsz - 1) // mbsz
                ):
                    start_idx = microbatch_idx * mbsz
                    end_idx = min(start_idx + mbsz, num_samples_in_batch)
                    sample_indices = [(i, j, k) for k in range(start_idx, end_idx)]
                    schedule.append((sample_indices, []))

                    if len(schedule) >= max_num_batches_to_schedule:
                        return schedule

                # 每个 batch 最后一个 micro-batch 做一次 optimizer step
                schedule[-1][1].append(i)

        return schedule

    # ============================================================
    # 对外调度接口：作为 PipelineExecutionConstruction 的实例方法
    # ============================================================
    def generate_cappuccino_schedule(
        self,
        aggregated_dataset: List[List[List[int]]],
        adapter_to_microbatch_size: List[int],
        pp_size: int = 4,
        max_num_batches_to_schedule: int = 100000,
        is_return_cappuccino_without_reorder: bool = True,
    ) -> List[Tuple[List[Tuple[int, int, int]], List[int]]]:
        """
        生成 Cappuccino 优化后的调度（类方法版）

        返回：
            优化后的微批次调度列表
        """
        # 1. 生成原始 mLoRA 调度
        print("生成原始 mLoRA 调度...")
        mlora_schedule = self._generate_mlora_schedule(
            aggregated_dataset=aggregated_dataset,
            adapter_to_microbatch_size=adapter_to_microbatch_size,
            max_num_batches_to_schedule=max_num_batches_to_schedule,
        )

        # print(f"生成的 mLoRA 调度包含 {len(mlora_schedule)} 个微批次")
        # for item in mlora_schedule:
        #     print(item)

        # 2. 分析调度中的批次结构
        print("分析批次结构...")
        batches: Dict[int, List[int]] = {}  # {batch_idx: [microbatch_indices]}

        for microbatch_idx, (microbatch_samples, optimizer_steps) in enumerate(mlora_schedule):
            for adapter_idx, batch_idx, sample_idx in microbatch_samples:
                if batch_idx not in batches:
                    batches[batch_idx] = []
                if microbatch_idx not in batches[batch_idx]:
                    batches[batch_idx].append(microbatch_idx)

        for batch_idx in batches:
            batches[batch_idx] = sorted(list(set(batches[batch_idx])))

        num_batches = len(batches)
        print(f"发现 {num_batches} 个批次")

        # 3. 为每个批次应用 Cappuccino 优化
        optimized_schedule: List[Tuple[List[Tuple[int, int, int]], List[int]]] = []
        optimized_schedule_without_reorder: List[Tuple[List[Tuple[int, int, int]], List[int]]] = []

        total_iteration_time = 0.0
        for batch_idx in sorted(batches.keys()):
            print(f"\n处理批次 {batch_idx}...")

            batch_microbatch_indices = batches[batch_idx]
            batch_microbatches = [mlora_schedule[idx] for idx in batch_microbatch_indices]

            print(f"批次 {batch_idx} 包含 {len(batch_microbatches)} 个微批次")

            if is_return_cappuccino_without_reorder:
                optimized_batch_without_reorder, optimized_batch, estimation_time = self._cappuccino_pipeline_planner(
                    batch_microbatches=batch_microbatches,
                    aggregated_dataset=aggregated_dataset,
                    pp_size=pp_size,
                    is_return_cappuccino_without_reorder=is_return_cappuccino_without_reorder,
                )
                optimized_schedule_without_reorder.extend(optimized_batch_without_reorder)
                optimized_schedule.extend(optimized_batch)


            else:
                optimized_batch, estimation_time = self._cappuccino_pipeline_planner(
                    batch_microbatches=batch_microbatches,
                    aggregated_dataset=aggregated_dataset,
                    pp_size=pp_size,
                    is_return_cappuccino_without_reorder=is_return_cappuccino_without_reorder,
                )
                optimized_schedule.extend(optimized_batch)
            

            print(
                f"批次 {batch_idx} 优化完成: "
                f"{len(batch_microbatches)} -> {len(optimized_batch)} 个微批次"
            )

            total_iteration_time += estimation_time if estimation_time is not None else 0.0

        print(f"\n总体优化完成: {len(mlora_schedule)} -> {len(optimized_schedule)} 个微批次")

        if is_return_cappuccino_without_reorder:
            return optimized_schedule_without_reorder,optimized_schedule, total_iteration_time
            
        print
        return optimized_schedule, total_iteration_time


# =============================================
# Click 测试入口
# =============================================
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
    help="Optional per-adapter microbatch sizes, e.g. '1,2,4,1'",
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
    adapter_to_dataset_idx_list = list_of_ints(adapter_to_dataset_idx)
    adapter_to_global_batch_size_list = list_of_ints(adapter_to_global_batch_size)

    if len(adapter_to_dataset_idx_list) != num_adapters:
        raise ValueError(
            f"--adapter_to_dataset_idx length must be {num_adapters}, "
            f"but got {len(adapter_to_dataset_idx_list)}"
        )
    if len(adapter_to_global_batch_size_list) != num_adapters:
        raise ValueError(
            f"--adapter_to_global_batch_size length must be {num_adapters}, "
            f"but got {len(adapter_to_global_batch_size_list)}"
        )

    # per-adapter microbatch_size
    if adapter_to_microbatch_size is None:
        adapter_to_microbatch_size_list = [microbatch_size] * num_adapters
    else:
        adapter_to_microbatch_size_list = list_of_ints(adapter_to_microbatch_size)
        if len(adapter_to_microbatch_size_list) != num_adapters:
            raise ValueError(
                f"--adapter_to_microbatch_size length must be {num_adapters}, "
                f"but got {len(adapter_to_microbatch_size_list)}"
            )

    if any(m <= 0 for m in adapter_to_microbatch_size_list):
        raise ValueError("All microbatch sizes must be positive.")

    print("=" * 60)
    print("Start Cappuccino Pipeline Planner Test")
    print("=" * 60)
    print(f"Per-adapter microbatch sizes: {adapter_to_microbatch_size_list}")

    # load dataset
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset_path not exists: {dataset_path}")

    dataset_list = load_dataset_list(dataset_path=path)

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

        global_bsz = adapter_to_global_batch_size_list[adapter_idx]
        curr_dataset: List[List[int]] = []
        max_batches_here = min(len(mock_dataset) // global_bsz, max_num_batches)

        for batch_idx in range(max_batches_here):
            batch = mock_dataset[
                batch_idx * global_bsz : (batch_idx + 1) * global_bsz
            ]
            curr_dataset.append(batch)

        aggregated_dataset.append(curr_dataset)

    constructor = PipelineExecutionConstruction(model_name=model_name)

    sched_wo, sched, total_time = constructor.generate_cappuccino_schedule(
        aggregated_dataset=aggregated_dataset,
        adapter_to_microbatch_size=adapter_to_microbatch_size_list,
        pp_size=num_pipeline_stages,
        is_return_cappuccino_without_reorder=True,
    )

    output_dir = Path("schedules")
    output_dir.mkdir(exist_ok=True)
    new_schedule = MicroBatchInfo.shedules_to_adapter_group_step_infos(
                schedules = sched, 
                aggregated_dataset=aggregated_dataset, 
                sequence_batch_layout="ragged")

    save_schedule(
        schedule=new_schedule,
        path=str(output_dir),
        output_name=output_name,
    )
    print("Done.")


if __name__ == "__main__":
    test_cappuccino_optimizer()

'''
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
'''