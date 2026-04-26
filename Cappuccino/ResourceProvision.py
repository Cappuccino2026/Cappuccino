from __future__ import annotations

import argparse
import heapq
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for ResourceProvision. "
        "Install it with: pip install scikit-learn"
    ) from exc

try:
    from .PipelineExecutionConstruction import PipelineExecutionConstruction
    from .utils import load_adapter_config, load_dataset_list
except Exception:  # pragma: no cover
    from PipelineExecutionConstruction import PipelineExecutionConstruction
    from utils import load_adapter_config, load_dataset_list

GPU_MEMORY_LIMIT_MAP: Dict[str, float] = {
    "a100-40gb": 40.0,
    "a100-80gb": 80.0,
}

# Number of batches used for pipeline schedule estimation.
DEFAULT_NUM_ESTIMATION_BATCHES = 10


def _split_target_modules(target_modules: str) -> List[str]:
    """Parse target module names from comma- or plus-separated strings."""
    target_modules = (target_modules or "").strip()
    if not target_modules:
        return []
    return [
        module.strip()
        for module in target_modules.replace(",", "+").split("+")
        if module.strip()
    ]


def _chunk_max(sequence: List[int], chunk_size: int) -> List[int]:
    """Split a sequence into fixed-size chunks and return the maximum of each chunk."""
    if chunk_size <= 0:
        raise ValueError("microbatch_size must be at least 1.")
    if not sequence:
        return []

    values = np.asarray(sequence, dtype=np.int64)
    padding = (-len(values)) % chunk_size
    if padding:
        values = np.pad(values, (0, padding), mode="edge")

    return np.max(values.reshape(-1, chunk_size), axis=1).tolist()


def _percentile(sorted_values: np.ndarray, q: float) -> float:
    """Return the q-th percentile for an already sorted array."""
    if sorted_values.size == 0:
        return 0.0

    try:
        return float(np.quantile(sorted_values, q / 100.0, method="linear"))
    except TypeError:  # pragma: no cover
        return float(np.quantile(sorted_values, q / 100.0, interpolation="linear"))


def _histogram_probabilities(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert sequence-length samples into a normalized histogram."""
    if values.size == 0:
        return np.zeros(edges.size - 1, dtype=np.float64)

    clipped_values = np.clip(np.asarray(values, dtype=np.float64), edges[0], edges[-1])
    counts, _ = np.histogram(clipped_values, bins=edges)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / total


def _emd_1d_from_hist_probs(p: np.ndarray, q: np.ndarray, edges: np.ndarray) -> float:
    """Compute exact 1D Wasserstein-1 distance from histogram probabilities."""
    cdf_diff = np.cumsum(p - q)
    bin_widths = np.diff(edges)
    return float(np.sum(np.abs(cdf_diff) * bin_widths))


@dataclass
class TaskFeature:
    """Feature vector used for grouping one adapter training task."""

    adapter_id: int
    dataset_idx: int
    dataset_name: str
    seed_idx: int
    perm_idx: int
    p50: float
    p90: float
    p99: float
    rel_spread: float
    tail_index: float
    hist_prob: np.ndarray
    global_batch_size: int
    microbatch_size: int
    rank: int
    target_modules: List[str]
    dropout: float
    peak_load: float
    lora_complexity: float


@dataclass
class ReplicaState:
    """Current GPU allocation state of one replica."""

    replica_id: int
    adapter_ids: List[int]
    g: int
    T_iter: float

    @property
    def dataset_indices(self) -> List[int]:
        """Backward-compatible alias for older downstream code."""
        return self.adapter_ids


class ResourceProvision:
    """Cluster LoRA tasks into replicas and allocate GPUs greedily across replicas."""

    def __init__(
        self,
        model_name: str,
        gpu_type: str,
        total_gpus: int,
        dataset_path: str,
        adapters: List[Dict[str, Any]],
        num_seq_len_bins: int = 16,
        seq_len_min: float = 1.0,
        seq_len_max: float = 4096.0,
        distance_weights: Tuple[float, float, float, float, float, float, float] = (
            0.45,
            0.20,
            0.15,
            0.10,
            0.10,
            0.0,
            0.0,
        ),
    ) -> None:
        self.model_name = model_name
        self.gpu_type = gpu_type
        self.total_gpus = int(total_gpus)
        self.dataset_path = dataset_path
        self.dataset_list = load_dataset_list(dataset_path=self.dataset_path)

        self.pipeline_constructor = PipelineExecutionConstruction(model_name=self.model_name)

        self.adapters = adapters
        self.num_adapters = len(adapters)

        self.adapter_to_dataset_idx: List[int] = []
        self.adapter_to_global_bsz: List[int] = []
        self.adapter_to_micro_bsz: List[int] = []
        self.adapter_to_rank: List[int] = []
        self.adapter_to_target_modules: List[List[str]] = []
        self.adapter_to_dropout: List[float] = []

        for adapter in adapters:
            self.adapter_to_dataset_idx.append(int(adapter["dataset_idx"]))
            self.adapter_to_global_bsz.append(int(adapter["global_batch_size"]))
            self.adapter_to_micro_bsz.append(int(adapter["microbatch_size"]))
            self.adapter_to_rank.append(int(adapter.get("rank", 16)))
            self.adapter_to_target_modules.append(
                _split_target_modules(adapter.get("target_modules", ""))
            )
            self.adapter_to_dropout.append(float(adapter.get("dropout", 0.0)))

        self.seq_len_min = float(seq_len_min)
        self.seq_len_max = float(seq_len_max)
        self.num_seq_len_bins = int(num_seq_len_bins)
        if self.seq_len_max <= self.seq_len_min:
            raise ValueError(
                "Invalid histogram range: "
                f"seq_len_min={self.seq_len_min}, seq_len_max={self.seq_len_max}."
            )

        if len(distance_weights) != 7:
            raise ValueError("distance_weights must contain exactly 7 elements.")
        (
            self.w_emd,
            self.w_dp90,
            self.w_dp99,
            self.w_tail,
            self.w_spread,
            self.w_peak,
            self.w_lora,
        ) = distance_weights

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            self.dataset_distribution = json.load(file)

        self.feats: List[TaskFeature] = []
        self.feats_by_adapter_id: Dict[int, TaskFeature] = {}

    def _load_lengths_for_dataset_idx(self, dataset_idx: int) -> Tuple[str, int, int, List[int]]:
        """Load token lengths for a dataset-list entry."""
        name, seed_idx, perm_idx = self.dataset_list[dataset_idx]
        dataset_block = self.dataset_distribution[name]
        seed_value = dataset_block["seeds"][seed_idx]
        seed_key = f"seed_{seed_value}"
        permutation_key = f"permutation_{perm_idx + 1}"
        lengths = dataset_block[seed_key][permutation_key]
        return name, seed_idx, perm_idx, lengths

    def extract_all_features(
        self,
        adapter_ids: Optional[List[int]] = None,
    ) -> Tuple[List[TaskFeature], np.ndarray]:
        """Extract sequence-distribution and LoRA-configuration features."""
        if adapter_ids is None:
            adapter_ids = list(range(self.num_adapters))

        edges = np.linspace(
            self.seq_len_min,
            self.seq_len_max,
            self.num_seq_len_bins + 1,
            dtype=np.float64,
        )

        features: List[TaskFeature] = []
        features_by_id: Dict[int, TaskFeature] = {}

        for adapter_id in adapter_ids:
            dataset_idx = self.adapter_to_dataset_idx[adapter_id]
            name, seed_idx, perm_idx, lengths = self._load_lengths_for_dataset_idx(dataset_idx)

            micro_bsz = self.adapter_to_micro_bsz[adapter_id]
            microbatch_max_lengths = _chunk_max(lengths, micro_bsz)
            sorted_microbatch_max_lengths = np.sort(
                np.asarray(microbatch_max_lengths, dtype=np.float64)
            )

            p50 = _percentile(sorted_microbatch_max_lengths, 50)
            p90 = _percentile(sorted_microbatch_max_lengths, 90)
            p99 = _percentile(sorted_microbatch_max_lengths, 99)
            rel_spread = (p90 - p50) / max(p50, 1e-6)
            tail_index = p99 / max(p90, 1e-6)
            hist_prob = _histogram_probabilities(sorted_microbatch_max_lengths, edges)

            global_bsz = self.adapter_to_global_bsz[adapter_id]
            rank = self.adapter_to_rank[adapter_id]
            target_modules = self.adapter_to_target_modules[adapter_id]
            dropout = self.adapter_to_dropout[adapter_id]

            peak_load = micro_bsz * p90
            lora_complexity = rank * max(len(target_modules), 1)

            feature = TaskFeature(
                adapter_id=adapter_id,
                dataset_idx=dataset_idx,
                dataset_name=name,
                seed_idx=seed_idx,
                perm_idx=perm_idx,
                p50=p50,
                p90=p90,
                p99=p99,
                rel_spread=rel_spread,
                tail_index=tail_index,
                hist_prob=hist_prob,
                global_batch_size=global_bsz,
                microbatch_size=micro_bsz,
                rank=rank,
                target_modules=target_modules,
                dropout=dropout,
                peak_load=peak_load,
                lora_complexity=lora_complexity,
            )
            features.append(feature)
            features_by_id[adapter_id] = feature

        self.feats = features
        self.feats_by_adapter_id = features_by_id
        return features, edges

    def pairwise_distance(self, first: TaskFeature, second: TaskFeature, edges: np.ndarray) -> float:
        """Compute the weighted distance between two adapter task features."""
        emd = _emd_1d_from_hist_probs(first.hist_prob, second.hist_prob, edges)
        dp90_rel = abs(first.p90 - second.p90) / max(first.p90, second.p90, 1e-6)
        dp99_rel = abs(first.p99 - second.p99) / max(first.p99, second.p99, 1e-6)
        tail_delta = abs(first.tail_index - second.tail_index)
        spread_delta = abs(first.rel_spread - second.rel_spread)
        peak_delta = abs(first.peak_load - second.peak_load) / max(
            first.peak_load,
            second.peak_load,
            1e-6,
        )
        lora_delta = abs(first.lora_complexity - second.lora_complexity) / max(
            first.lora_complexity,
            second.lora_complexity,
            1e-6,
        )

        return (
            self.w_emd * emd
            + self.w_dp90 * dp90_rel
            + self.w_dp99 * dp99_rel
            + self.w_tail * tail_delta
            + self.w_spread * spread_delta
            + self.w_peak * peak_delta
            + self.w_lora * lora_delta
        )

    def distance_matrix(self, features: List[TaskFeature], edges: np.ndarray) -> np.ndarray:
        """Build a symmetric pairwise-distance matrix for clustering."""
        num_features = len(features)
        distances = np.zeros((num_features, num_features), dtype=np.float64)
        for i in range(num_features):
            for j in range(i + 1, num_features):
                distance = self.pairwise_distance(features[i], features[j], edges)
                distances[i, j] = distances[j, i] = distance
        return distances

    def auto_cluster_with_silhouette(
        self,
        feats: List[TaskFeature],
        edges: np.ndarray,
        kmin: int = 1,
        kmax: Optional[int] = None,
        linkage: str = "average",
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """Select the number of clusters using silhouette score on precomputed distances."""
        num_features = len(feats)
        if num_features == 0:
            return [], {"best_K": 0, "best_score": 0.0, "candidates": []}
        if num_features == 1:
            return [[feats[0].adapter_id]], {
                "best_K": 1,
                "best_score": 1.0,
                "candidates": [(1, 1.0, 0.0)],
            }

        distances = self.distance_matrix(feats, edges)
        if kmax is None:
            kmax = num_features
        else:
            kmax = min(kmax, num_features)
        kmin = max(1, min(kmin, kmax))

        candidates: List[Tuple[int, float, float]] = []
        best: Optional[Tuple[float, int, np.ndarray, float]] = None

        for num_clusters in range(kmin, kmax + 1):
            if num_clusters == 1:
                labels = np.zeros(num_features, dtype=int)
                score = -1.0
                intra_cost = float(distances.sum() / 2.0)
            else:
                try:
                    model = AgglomerativeClustering(
                        n_clusters=num_clusters,
                        linkage=linkage,
                        metric="precomputed",
                    )
                except TypeError:  # pragma: no cover
                    model = AgglomerativeClustering(
                        n_clusters=num_clusters,
                        linkage=linkage,
                        affinity="precomputed",
                    )

                labels = model.fit_predict(distances)
                try:
                    score = float(silhouette_score(distances, labels, metric="precomputed"))
                except Exception as exc:
                    logger.warning(
                        "Silhouette score computation failed | k={} | error={}",
                        num_clusters,
                        exc,
                    )
                    score = -1.0

                intra_cost = 0.0
                for cluster_id in range(num_clusters):
                    members = np.where(labels == cluster_id)[0]
                    if members.size > 1:
                        sub_matrix = distances[np.ix_(members, members)]
                        intra_cost += float(sub_matrix.sum() / 2.0)

            candidates.append((num_clusters, score, intra_cost))
            is_better_score = best is None or score > best[0] + 1e-9
            is_better_cost = (
                best is not None
                and abs(score - best[0]) <= 1e-9
                and intra_cost < best[3]
            )
            if is_better_score or is_better_cost:
                best = (score, num_clusters, labels, intra_cost)

        assert best is not None
        best_score, best_k, labels, best_cost = best

        groups: Dict[int, List[int]] = {}
        for feature_idx, label in enumerate(labels):
            groups.setdefault(int(label), []).append(feats[feature_idx].adapter_id)

        clusters = list(groups.values())
        return clusters, {
            "best_K": int(best_k),
            "best_score": float(best_score),
            "best_cost": float(best_cost),
            "candidates": candidates,
        }

    def _generate_aggregated_dataset(
        self,
        adapter_ids: List[int],
        batch_number: int = DEFAULT_NUM_ESTIMATION_BATCHES,
    ) -> List[List[List[int]]]:
        """Build aggregated_dataset[local_adapter][batch][sample] = token_length."""
        aggregated: List[List[List[int]]] = []

        for adapter_id in adapter_ids:
            feature = self.feats_by_adapter_id[adapter_id]
            _, _, _, lengths = self._load_lengths_for_dataset_idx(feature.dataset_idx)

            global_bsz = max(int(feature.global_batch_size), 1)
            max_batches = min(batch_number, max(len(lengths) // global_bsz, 0))

            current_dataset: List[List[int]] = []
            for batch_idx in range(max_batches):
                start = batch_idx * global_bsz
                end = start + global_bsz
                current_dataset.append([int(length) for length in lengths[start:end]])

            if not current_dataset:
                # Keep scheduler inputs non-empty for tiny or degenerate datasets.
                current_dataset.append([int(length) for length in lengths[:global_bsz]])

            aggregated.append(current_dataset)

        return aggregated

    def min_feasible_gpus_for_cluster(self, replica_id: int, adapter_ids: List[int]) -> ReplicaState:
        """Find the minimum feasible pipeline size for a replica."""
        gpu_memory_limit = GPU_MEMORY_LIMIT_MAP.get(self.gpu_type, 40.0)

        pp_size = 1
        while pp_size <= self.total_gpus:
            is_feasible = True
            for adapter_id in adapter_ids:
                feature = self.feats_by_adapter_id[adapter_id]
                if not self.pipeline_constructor.job_leval_check_if_fit_memory(
                    micro_batchsize=int(feature.microbatch_size),
                    seq_length=int(math.ceil(feature.p90)),
                    rank=int(feature.rank),
                    pp_size=pp_size,
                    gpu_memory_limit=gpu_memory_limit,
                ):
                    is_feasible = False
                    break
            if is_feasible:
                break
            pp_size += 1

        if pp_size > self.total_gpus:
            raise RuntimeError(
                f"Replica {replica_id}: no feasible pp_size in [1, {self.total_gpus}] "
                f"under a {gpu_memory_limit:.2f} GB memory limit."
            )

        aggregated_dataset = self._generate_aggregated_dataset(
            adapter_ids,
            batch_number=DEFAULT_NUM_ESTIMATION_BATCHES,
        )
        adapter_to_microbatch_size = [self.adapter_to_micro_bsz[adapter_id] for adapter_id in adapter_ids]

        schedule, iteration_time = self.pipeline_constructor.generate_cappuccino_schedule(
            aggregated_dataset=aggregated_dataset,
            adapter_to_microbatch_size=adapter_to_microbatch_size,
            pp_size=pp_size,
            is_return_cappuccino_without_reorder=False,
        )
        if not schedule or iteration_time is None:
            raise RuntimeError(
                f"Replica {replica_id}: schedule estimation failed for pp_size={pp_size}."
            )

        return ReplicaState(
            replica_id=replica_id,
            adapter_ids=adapter_ids,
            g=pp_size,
            T_iter=float(iteration_time),
        )

    def evaluate_one_more_gpu(self, replica: ReplicaState) -> Optional[Tuple[float, float]]:
        """Return the GPU-time saving from assigning one more GPU to a replica."""
        old_gpu_count = replica.g
        new_gpu_count = old_gpu_count + 1
        if new_gpu_count > self.total_gpus:
            return None

        gpu_memory_limit = GPU_MEMORY_LIMIT_MAP.get(self.gpu_type, 40.0)
        for adapter_id in replica.adapter_ids:
            feature = self.feats_by_adapter_id[adapter_id]
            if not self.pipeline_constructor.job_leval_check_if_fit_memory(
                micro_batchsize=int(feature.microbatch_size),
                seq_length=int(math.ceil(feature.p90)),
                rank=int(feature.rank),
                pp_size=new_gpu_count,
                gpu_memory_limit=gpu_memory_limit,
            ):
                return None

        aggregated_dataset = self._generate_aggregated_dataset(
            replica.adapter_ids,
            batch_number=DEFAULT_NUM_ESTIMATION_BATCHES,
        )
        adapter_to_microbatch_size = [
            self.adapter_to_micro_bsz[adapter_id] for adapter_id in replica.adapter_ids
        ]

        new_schedule, new_iteration_time = self.pipeline_constructor.generate_cappuccino_schedule(
            aggregated_dataset=aggregated_dataset,
            adapter_to_microbatch_size=adapter_to_microbatch_size,
            pp_size=new_gpu_count,
            is_return_cappuccino_without_reorder=False,
        )
        if not new_schedule or new_iteration_time is None:
            return None

        old_gpu_time = old_gpu_count * replica.T_iter
        new_gpu_time = new_gpu_count * float(new_iteration_time)
        delta = old_gpu_time - new_gpu_time
        if delta <= 0:
            return None

        return float(delta), float(new_iteration_time)

    def greedy_gpu_allocation_over_replicas(
        self,
        clusters: List[List[int]],
    ) -> Tuple[List[ReplicaState], float]:
        """Allocate remaining GPUs by repeatedly choosing the largest GPU-time reduction."""
        replicas = [
            self.min_feasible_gpus_for_cluster(replica_id, adapter_ids)
            for replica_id, adapter_ids in enumerate(clusters)
        ]

        total_gpu_time = sum(replica.g * replica.T_iter for replica in replicas)
        used_gpus = sum(replica.g for replica in replicas)
        if used_gpus > self.total_gpus:
            raise ValueError(
                f"Minimum GPU demand ({used_gpus}) exceeds total_gpus={self.total_gpus}."
            )

        remaining_budget = self.total_gpus - used_gpus
        if remaining_budget <= 0:
            return replicas, float(total_gpu_time)

        heap: List[Tuple[float, int, int, float]] = []
        for replica in replicas:
            candidate = self.evaluate_one_more_gpu(replica)
            if candidate is None:
                continue
            delta, new_iteration_time = candidate
            heapq.heappush(heap, (-delta, replica.replica_id, replica.g + 1, new_iteration_time))

        while remaining_budget > 0 and heap:
            neg_delta, replica_id, new_gpu_count, new_iteration_time = heapq.heappop(heap)
            delta = -neg_delta
            if delta <= 0:
                break

            replica = replicas[replica_id]
            if replica.g != new_gpu_count - 1:
                refreshed_candidate = self.evaluate_one_more_gpu(replica)
                if refreshed_candidate is None:
                    continue
                refreshed_delta, refreshed_iteration_time = refreshed_candidate
                if refreshed_delta > 0:
                    heapq.heappush(
                        heap,
                        (-refreshed_delta, replica.replica_id, replica.g + 1, refreshed_iteration_time),
                    )
                continue

            replica.g = new_gpu_count
            replica.T_iter = new_iteration_time
            remaining_budget -= 1
            total_gpu_time -= delta

            if remaining_budget > 0:
                next_candidate = self.evaluate_one_more_gpu(replica)
                if next_candidate is not None:
                    next_delta, next_iteration_time = next_candidate
                    if next_delta > 0:
                        heapq.heappush(
                            heap,
                            (-next_delta, replica.replica_id, replica.g + 1, next_iteration_time),
                        )

        return replicas, float(total_gpu_time)

    def resource_provisioning(
        self,
        kmin: int = 1,
        kmax: Optional[int] = None,
        linkage: str = "average",
    ) -> Dict[str, Any]:
        """Run the full provisioning pipeline: feature extraction, clustering, and allocation."""
        features, edges = self.extract_all_features()
        clusters, cluster_info = self.auto_cluster_with_silhouette(
            feats=features,
            edges=edges,
            kmin=kmin,
            kmax=kmax,
            linkage=linkage,
        )
        replicas, total_gpu_time = self.greedy_gpu_allocation_over_replicas(clusters)

        return {
            "replicas": replicas,
            "clusters": clusters,
            "cluster_info": cluster_info,
            "total_gpu_time": total_gpu_time,
        }


def _parse_float_list(values: str) -> Tuple[float, ...]:
    """Parse a comma- or whitespace-separated list of floats."""
    return tuple(float(value) for value in values.replace(",", " ").split() if value.strip())


def configure_loguru(log_level: str = "INFO", quiet: bool = False) -> None:
    """Configure Loguru output for all Cappuccino modules.

    Args:
        log_level: Minimum log level to emit. Common values are DEBUG, INFO,
            WARNING, ERROR, and CRITICAL.
        quiet: If true, suppress all logs below ERROR.
    """
    effective_level = "ERROR" if quiet else log_level.upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=effective_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Cappuccino resource provisioning")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--adapter_config", required=True)
    parser.add_argument("--gpu_type", default="a100-40gb", choices=list(GPU_MEMORY_LIMIT_MAP.keys()))
    parser.add_argument("--total_gpus", type=int, required=True)
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--linkage", default="average")
    parser.add_argument("--num_seq_len_bins", type=int, default=16)
    parser.add_argument("--seq_len_min", type=float, default=1.0)
    parser.add_argument("--seq_len_max", type=float, default=4096.0)
    parser.add_argument(
        "--distance_weights",
        default="0.45,0.20,0.15,0.10,0.10,0.0,0.0",
        help="Seven weights for: emd, dp90, dp99, tail, spread, peak, lora.",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--log_level", default="INFO", choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error logs.")
    return parser


def _serialize_replicas(replicas: List[ReplicaState]) -> List[Dict[str, Any]]:
    """Serialize replica states into a JSON-friendly representation."""
    return [asdict(replica) for replica in replicas]


def main() -> None:
    """CLI entry point for resource provisioning."""
    args = build_argparser().parse_args()
    configure_loguru(log_level=args.log_level, quiet=args.quiet)

    logger.info("Starting Cappuccino resource provisioning.")
    logger.info(
        "Configuration | model={} | gpu_type={} | total_gpus={} | k_range=[{}, {}]",
        args.model_name,
        args.gpu_type,
        args.total_gpus,
        args.kmin,
        args.kmax,
    )

    adapters = load_adapter_config(args.adapter_config)
    distance_weights = _parse_float_list(args.distance_weights)

    provisioner = ResourceProvision(
        model_name=args.model_name,
        gpu_type=args.gpu_type,
        total_gpus=args.total_gpus,
        dataset_path=args.dataset_path,
        adapters=adapters,
        num_seq_len_bins=args.num_seq_len_bins,
        seq_len_min=args.seq_len_min,
        seq_len_max=args.seq_len_max,
        distance_weights=distance_weights,  # type: ignore[arg-type]
    )

    result = provisioner.resource_provisioning(
        kmin=args.kmin,
        kmax=args.kmax,
        linkage=args.linkage,
    )

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = output_dir / "resource_provisioning_result.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump(result, file)

    json_path = output_dir / "resource_provisioning_result.json"
    json_obj = {
        "replicas": _serialize_replicas(result["replicas"]),
        "clusters": result["clusters"],
        "cluster_info": result["cluster_info"],
        "total_gpu_time": result["total_gpu_time"],
    }
    json_path.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")

    logger.info("Resource provisioning completed successfully.")
    logger.info("Saved pickle result | path = {}", pickle_path)
    logger.info("Saved JSON summary | path = {}", json_path)


if __name__ == "__main__":
    main()

"""
Example:
    python -m Cappuccino.ResourceProvision \
      --model_name "Qwen2.5-32B-Instruct" \
      --dataset_path "examples/dataset_distributions_16all_4096_seqlen_42_seed_1000_samples.json" \
      --adapter_config "examples/adapter_config.json" \
      --total_gpus 16 \
      --gpu_type "a100-40gb" \
      --output_dir "results"
"""