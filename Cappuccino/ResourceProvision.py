
from __future__ import annotations


import argparse
import json
import math
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for ResourceProvision (pip install scikit-learn)."
    ) from e

# Local import (works for both `python -m` and direct script execution)
try:
    from .PipelineExecutionConstruction import PipelineExecutionConstruction
    from .utils import load_dataset_list, load_adapter_config
except Exception:  # pragma: no cover
    from PipelineExecutionConstruction import PipelineExecutionConstruction
    from utils import load_dataset_list


# -----------------------------
# Constants
# -----------------------------

GPU_MEMORY_LIMIT_MAP = {
    "a100-40gb": 40.0,
    "a100-80gb": 80.0,
}

BATCH_NUMBER = 10  # number of batches used in pipeline schedule estimation

# -----------------------------
# Helpers
# -----------------------------

def _split_target_modules(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.replace(",", "+").split("+") if x.strip()]


def _chunk_max(seq: List[int], chunk: int) -> List[int]:
    """Group `seq` into chunks and take max per chunk."""
    if chunk <= 0:
        raise ValueError("microbatch_size must be >= 1")
    if not seq:
        return []
    arr = np.asarray(seq, dtype=np.int64)
    pad = (-len(arr)) % chunk
    if pad:
        arr = np.pad(arr, (0, pad), mode="edge")
    return np.max(arr.reshape(-1, chunk), axis=1).tolist()


def _percentile(sorted_vals: np.ndarray, q: float) -> float:
    if sorted_vals.size == 0:
        return 0.0
    # numpy>=1.22: prefer method
    try:
        return float(np.quantile(sorted_vals, q / 100.0, method="linear"))
    except TypeError:  # pragma: no cover
        return float(np.quantile(sorted_vals, q / 100.0, interpolation="linear"))


def _hist_prob(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(edges.size - 1, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    vals = np.clip(vals, edges[0], edges[-1])
    h, _ = np.histogram(vals, bins=edges)
    s = h.sum()
    return (h / s) if s > 0 else np.zeros_like(h, dtype=np.float64)


def _emd_1d_from_hist_probs(p: np.ndarray, q: np.ndarray, edges: np.ndarray) -> float:
    """Exact 1D Wasserstein-1 using histogram CDF differences."""
    cdf_diff = np.cumsum(p - q)
    widths = np.diff(edges)
    return float(np.sum(np.abs(cdf_diff) * widths))


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class TaskFeature:
    # identity
    adapter_id: int
    dataset_idx: int
    dataset_name: str
    seed_idx: int
    perm_idx: int

    # distribution
    p50: float
    p90: float
    p99: float
    rel_spread: float
    tail_index: float
    hist_prob: np.ndarray

    # hyper-params
    global_batch_size: int
    microbatch_size: int
    rank: int
    target_modules: List[str]
    dropout: float

    # derived
    peak_load: float
    lora_complexity: float


@dataclass
class ReplicaState:
    replica_id: int
    dataset_indices: List[int]  # adapter_ids inside the replica
    g: int
    T_iter: float


# -----------------------------
# ResourceProvision
# -----------------------------


class ResourceProvision:
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
    ):
        self.model_name = model_name
        self.gpu_type = gpu_type
        self.total_gpus = int(total_gpus)
        self.dataset_path = dataset_path
        self.dataset_list = load_dataset_list(dataset_path=self.dataset_path)

        # pipeline cost estimator
        self.pipeline_constructor = PipelineExecutionConstruction(model_name=self.model_name)

        # adapter hyper-params
        self.adapters = adapters
        self.num_adapters = len(adapters)

        self.adapter_to_dataset_idx: List[int] = []
        self.adapter_to_global_bsz: List[int] = []
        self.adapter_to_micro_bsz: List[int] = []
        self.adapter_to_rank: List[int] = []
        self.adapter_to_target_modules: List[List[str]] = []
        self.adapter_to_dropout: List[float] = []

        for a in adapters:
            self.adapter_to_dataset_idx.append(int(a["dataset_idx"]))
            self.adapter_to_global_bsz.append(int(a["global_batch_size"]))
            self.adapter_to_micro_bsz.append(int(a["microbatch_size"]))
            self.adapter_to_rank.append(int(a.get("rank", 16)))
            self.adapter_to_target_modules.append(_split_target_modules(a.get("target_modules", "")))
            self.adapter_to_dropout.append(float(a.get("dropout", 0.0)))

        # histogram settings
        self.seq_len_min = float(seq_len_min)
        self.seq_len_max = float(seq_len_max)
        self.num_seq_len_bins = int(num_seq_len_bins)
        if self.seq_len_max <= self.seq_len_min:
            raise ValueError(
                f"Invalid histogram range: seq_len_min={self.seq_len_min}, seq_len_max={self.seq_len_max}"
            )

        if len(distance_weights) != 7:
            raise ValueError("distance_weights must have 7 elements")
        (
            self.w_emd,
            self.w_dp90,
            self.w_dp99,
            self.w_tail,
            self.w_spread,
            self.w_peak,
            self.w_lora,
        ) = distance_weights

        # dataset distributions
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.ds = json.load(f)

        self.feats: List[TaskFeature] = []
        self.feats_by_adapter_id: Dict[int, TaskFeature] = {}

    # -------------------------
    # Dataset distribution IO
    # -------------------------

    def _load_lengths_for_dataset_idx(self, dataset_idx: int) -> Tuple[str, int, int, List[int]]:
        """Return (name, seed_idx, perm_idx, lengths) for a dataset_list index."""
        name, seed_i, perm_i = self.dataset_list[dataset_idx]
        block = self.ds[name]
        seed_val = block["seeds"][seed_i]
        seed_key = f"seed_{seed_val}"
        perm_key = f"permutation_{perm_i + 1}"
        lengths = block[seed_key][perm_key]
        return name, seed_i, perm_i, lengths

    # -------------------------
    # Feature extraction
    # -------------------------

    def extract_all_features(self, adapter_ids: Optional[List[int]] = None) -> Tuple[List[TaskFeature], np.ndarray]:
        """Extract features for each adapter_id."""
        if adapter_ids is None:
            adapter_ids = list(range(self.num_adapters))

        edges = np.linspace(
            self.seq_len_min, self.seq_len_max, self.num_seq_len_bins + 1, dtype=np.float64
        )

        feats: List[TaskFeature] = []
        by_id: Dict[int, TaskFeature] = {}

        for adapter_id in adapter_ids:
            ds_idx = self.adapter_to_dataset_idx[adapter_id]
            name, seed_i, perm_i, lengths = self._load_lengths_for_dataset_idx(ds_idx)

            micro_bsz = self.adapter_to_micro_bsz[adapter_id]
            bmax_seq = _chunk_max(lengths, micro_bsz)
            bmax_sorted = np.sort(np.asarray(bmax_seq, dtype=np.float64))

            p50 = _percentile(bmax_sorted, 50)
            p90 = _percentile(bmax_sorted, 90)
            p99 = _percentile(bmax_sorted, 99)
            rel_spread = (p90 - p50) / max(p50, 1e-6)
            tail_index = p99 / max(p90, 1e-6)
            hprob = _hist_prob(bmax_sorted, edges)

            global_bsz = self.adapter_to_global_bsz[adapter_id]
            rank = self.adapter_to_rank[adapter_id]
            tgt_mods = self.adapter_to_target_modules[adapter_id]
            dropout = self.adapter_to_dropout[adapter_id]

            peak_load = micro_bsz * p90
            lora_complexity = rank * max(len(tgt_mods), 1)

            tf = TaskFeature(
                adapter_id=adapter_id,
                dataset_idx=ds_idx,
                dataset_name=name,
                seed_idx=seed_i,
                perm_idx=perm_i,
                p50=p50,
                p90=p90,
                p99=p99,
                rel_spread=rel_spread,
                tail_index=tail_index,
                hist_prob=hprob,
                global_batch_size=global_bsz,
                microbatch_size=micro_bsz,
                rank=rank,
                target_modules=tgt_mods,
                dropout=dropout,
                peak_load=peak_load,
                lora_complexity=lora_complexity,
            )
            feats.append(tf)
            by_id[adapter_id] = tf

        self.feats = feats
        self.feats_by_adapter_id = by_id
        return feats, edges

    # -------------------------
    # Distance & clustering
    # -------------------------

    def pairwise_distance(self, A: TaskFeature, B: TaskFeature, edges: np.ndarray) -> float:
        emd = _emd_1d_from_hist_probs(A.hist_prob, B.hist_prob, edges)

        dp90_rel = abs(A.p90 - B.p90) / max(A.p90, B.p90, 1e-6)
        dp99_rel = abs(A.p99 - B.p99) / max(A.p99, B.p99, 1e-6)
        dtail = abs(A.tail_index - B.tail_index)
        dspread = abs(A.rel_spread - B.rel_spread)

        dpeak = abs(A.peak_load - B.peak_load) / max(A.peak_load, B.peak_load, 1e-6)
        dlora = abs(A.lora_complexity - B.lora_complexity) / max(
            A.lora_complexity, B.lora_complexity, 1e-6
        )

        return (
            self.w_emd * emd
            + self.w_dp90 * dp90_rel
            + self.w_dp99 * dp99_rel
            + self.w_tail * dtail
            + self.w_spread * dspread
            + self.w_peak * dpeak
            + self.w_lora * dlora
        )

    def distance_matrix(self, feats: List[TaskFeature], edges: np.ndarray) -> np.ndarray:
        n = len(feats)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.pairwise_distance(feats[i], feats[j], edges)
                D[i, j] = D[j, i] = d
        return D

    def auto_cluster_with_silhouette(
        self,
        feats: List[TaskFeature],
        edges: np.ndarray,
        Kmin: int = 1,
        Kmax: Optional[int] = None,
        linkage: str = "average",
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """Choose K with best silhouette score on precomputed distances."""
        n = len(feats)
        if n == 0:
            return [], {"best_K": 0, "best_score": 0.0, "candidates": []}
        if n == 1:
            return [[feats[0].adapter_id]], {"best_K": 1, "best_score": 1.0, "candidates": [(1, 1.0, 0.0)]}

        D = self.distance_matrix(feats, edges)

        if Kmax is None:
            Kmax = n
        else:
            Kmax = min(Kmax, n)
        Kmin = max(1, min(Kmin, Kmax))

        candidates: List[Tuple[int, float, float]] = []  # (K, score, intra_cost)
        best: Optional[Tuple[float, int, np.ndarray, float]] = None  # (score,K,labels,cost)

        for K in range(Kmin, Kmax + 1):
            if K == 1:
                labels = np.zeros(n, dtype=int)
                score = -1.0
                cost = float(D.sum() / 2.0)
            else:
                # sklearn API changed: metric vs affinity
                try:
                    model = AgglomerativeClustering(
                        n_clusters=K,
                        linkage=linkage,
                        metric="precomputed",
                    )
                except TypeError:  # pragma: no cover
                    model = AgglomerativeClustering(
                        n_clusters=K,
                        linkage=linkage,
                        affinity="precomputed",
                    )
                labels = model.fit_predict(D)
                try:
                    score = float(silhouette_score(D, labels, metric="precomputed"))
                except Exception:
                    score = -1.0

                cost = 0.0
                for k in range(K):
                    members = np.where(labels == k)[0]
                    if members.size > 1:
                        sub = D[np.ix_(members, members)]
                        cost += float(sub.sum() / 2.0)

            candidates.append((K, score, cost))
            if best is None or score > best[0] + 1e-9 or (abs(score - best[0]) <= 1e-9 and cost < best[3]):
                best = (score, K, labels, cost)

        assert best is not None
        best_score, best_K, labels, best_cost = best

        groups: Dict[int, List[int]] = {}
        for i, k in enumerate(labels):
            groups.setdefault(int(k), []).append(feats[i].adapter_id)
        clusters = list(groups.values())

        return clusters, {
            "best_K": int(best_K),
            "best_score": float(best_score),
            "best_cost": float(best_cost),
            "candidates": candidates,
        }

    # -------------------------
    # Scheduling helpers
    # -------------------------

    def _generate_aggregated_dataset(self, adapter_ids: List[int], batch_number: int = BATCH_NUMBER) -> List[List[List[int]]]:
        """Build aggregated_dataset[local_adapter][batch][sample] = length."""
        aggregated: List[List[List[int]]] = []

        for adapter_id in adapter_ids:
            tf = self.feats_by_adapter_id[adapter_id]
            _, _, _, lengths = self._load_lengths_for_dataset_idx(tf.dataset_idx)

            gb = max(int(tf.global_batch_size), 1)
            max_batches = min(batch_number, max(len(lengths) // gb, 0))

            curr: List[List[int]] = []
            for b_idx in range(max_batches):
                start = b_idx * gb
                end = start + gb
                curr.append([int(x) for x in lengths[start:end]])

            if not curr:
                # ensure non-empty to avoid corner cases in scheduler
                curr.append([int(x) for x in lengths[:gb]])

            aggregated.append(curr)

        return aggregated

    # -------------------------
    # GPU allocation
    # -------------------------

    def min_feasible_gpus_for_cluster(self, replica_id: int, adapter_ids: List[int]) -> ReplicaState:
        gpu_memory_limit = GPU_MEMORY_LIMIT_MAP.get(self.gpu_type, 40.0)

        # try pp_size from 1..total_gpus
        pp_size = 1
        while pp_size <= self.total_gpus:
            ok = True
            for adapter_id in adapter_ids:
                tf = self.feats_by_adapter_id[adapter_id]
                if not self.pipeline_constructor.job_leval_check_if_fit_memory(
                    micro_batchsize=int(tf.microbatch_size),
                    seq_length=int(math.ceil(tf.p90)),
                    rank=int(tf.rank),
                    pp_size=pp_size,
                    gpu_memory_limit=gpu_memory_limit,
                ):
                    ok = False
                    break
            if ok:
                break
            pp_size += 1

        if pp_size > self.total_gpus:
            raise RuntimeError(
                f"[Replica {replica_id}] No feasible pp_size in [1,{self.total_gpus}] under {gpu_memory_limit}GB."
            )

        # schedule estimation
        aggregated_dataset = self._generate_aggregated_dataset(adapter_ids, batch_number=BATCH_NUMBER)
        adapter_to_microbatch_size = [self.adapter_to_micro_bsz[i] for i in adapter_ids]
        adapter_to_rank = [self.adapter_to_rank[i] for i in adapter_ids]

        schedule, T_iter = self.pipeline_constructor.generate_cappuccino_schedule(
            aggregated_dataset=aggregated_dataset,
            adapter_to_microbatch_size=adapter_to_microbatch_size,
            pp_size=pp_size,
            is_return_cappuccino_without_reorder=False,
        )
        if not schedule or T_iter is None:
            raise RuntimeError(
                f"[Replica {replica_id}] schedule estimation failed for pp_size={pp_size}."
            )

        return ReplicaState(replica_id=replica_id, dataset_indices=adapter_ids, g=pp_size, T_iter=float(T_iter))

    def evaluate_one_more_gpu(self, replica: ReplicaState) -> Optional[Tuple[float, float]]:
        g_old = replica.g
        g_new = g_old + 1
        if g_new > self.total_gpus:
            return None

        gpu_memory_limit = GPU_MEMORY_LIMIT_MAP.get(self.gpu_type, 40.0)

        # memory feasibility
        for adapter_id in replica.dataset_indices:
            tf = self.feats_by_adapter_id[adapter_id]
            if not self.pipeline_constructor.job_leval_check_if_fit_memory(
                micro_batchsize=int(tf.microbatch_size),
                seq_length=int(math.ceil(tf.p90)),
                rank=int(tf.rank),
                pp_size=g_new,
                gpu_memory_limit=gpu_memory_limit,
            ):
                return None

        aggregated_dataset = self._generate_aggregated_dataset(replica.dataset_indices, batch_number=BATCH_NUMBER)
        adapter_to_microbatch_size = [self.adapter_to_micro_bsz[i] for i in replica.dataset_indices]
        adapter_to_rank = [self.adapter_to_rank[i] for i in replica.dataset_indices]

        schedule_new, T_iter_new = self.pipeline_constructor.generate_cappuccino_schedule(
            aggregated_dataset=aggregated_dataset,
            adapter_to_microbatch_size=adapter_to_microbatch_size,
            pp_size=g_new,
            is_return_cappuccino_without_reorder=False,
        )
        if not schedule_new or T_iter_new is None:
            return None

        gpu_time_old = g_old * replica.T_iter
        gpu_time_new = g_new * float(T_iter_new)
        delta = gpu_time_old - gpu_time_new
        if delta <= 0:
            return None

        return float(delta), float(T_iter_new)

    def greedy_gpu_allocation_over_replicas(self, clusters: List[List[int]]) -> Tuple[List[ReplicaState], float]:
        # Step 1: minimal feasible per replica
        replicas: List[ReplicaState] = []
        for rid, adapter_ids in enumerate(clusters):
            replicas.append(self.min_feasible_gpus_for_cluster(rid, adapter_ids))

        total_gpu_time = sum(r.g * r.T_iter for r in replicas)
        used = sum(r.g for r in replicas)
        if used > self.total_gpus:
            raise ValueError(f"Minimal demand {used} exceeds total_gpus={self.total_gpus}")

        budget = self.total_gpus - used
        if budget <= 0:
            return replicas, float(total_gpu_time)

        # Step 2: init max heap by delta
        heap: List[Tuple[float, int, int, float]] = []  # (-delta, rid, g_new, T_new)
        for r in replicas:
            cand = self.evaluate_one_more_gpu(r)
            if cand is None:
                continue
            delta, T_new = cand
            heapq.heappush(heap, (-delta, r.replica_id, r.g + 1, T_new))

        # Step 3: greedy allocate
        while budget > 0 and heap:
            neg_delta, rid, g_new, T_new = heapq.heappop(heap)
            delta = -neg_delta
            if delta <= 0:
                break

            r = replicas[rid]

            # lazy check: state drift
            if r.g != g_new - 1:
                cand2 = self.evaluate_one_more_gpu(r)
                if cand2 is None:
                    continue
                delta2, T_new2 = cand2
                if delta2 <= 0:
                    continue
                heapq.heappush(heap, (-delta2, r.replica_id, r.g + 1, T_new2))
                continue

            # apply allocation
            r.g = g_new
            r.T_iter = T_new
            budget -= 1
            total_gpu_time -= delta

            # push next candidate for same replica
            if budget > 0:
                cand_next = self.evaluate_one_more_gpu(r)
                if cand_next is not None:
                    delta_next, T_next = cand_next
                    if delta_next > 0:
                        heapq.heappush(heap, (-delta_next, r.replica_id, r.g + 1, T_next))

        return replicas, float(total_gpu_time)

    # -------------------------
    # Public API
    # -------------------------

    def resource_provisioning(
        self,
        kmin: int = 1,
        kmax: Optional[int] = None,
        linkage: str = "average",
    ) -> Dict[str, Any]:
        """One-call interface: feature -> cluster -> greedy allocate."""

        feats, edges = self.extract_all_features()

        clusters, cluster_info = self.auto_cluster_with_silhouette(
            feats=feats,
            edges=edges,
            Kmin=kmin,
            Kmax=kmax,
            linkage=linkage,
        )

        replicas, total_gpu_time = self.greedy_gpu_allocation_over_replicas(clusters)

        return {
            "replicas": replicas,
            "clusters": clusters,
            "cluster_info": cluster_info,
            "total_gpu_time": total_gpu_time,
        }


# -----------------------------
# CLI
# -----------------------------


def _parse_float_list(s: str) -> Tuple[float, ...]:
    vals = [float(x) for x in s.replace(",", " ").split() if x.strip()]
    return tuple(vals)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cappuccino Resource Provisioning")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--adapter_config", required=True)

    p.add_argument("--gpu_type", default="a100-40gb", choices=list(GPU_MEMORY_LIMIT_MAP.keys()))
    p.add_argument("--total_gpus", type=int, required=True)

    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=8)
    p.add_argument("--linkage", default="average")

    p.add_argument("--num_seq_len_bins", type=int, default=16)
    p.add_argument("--seq_len_min", type=float, default=1.0)
    p.add_argument("--seq_len_max", type=float, default=4096.0)

    p.add_argument(
        "--distance_weights",
        default="0.45,0.20,0.15,0.10,0.10,0.0,0.0",
        help="7 floats: emd, dp90, dp99, tail, spread, peak, lora",
    )

    p.add_argument("--output_dir", default="results")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    adapters = load_adapter_config(args.adapter_config)
    distance_weights = _parse_float_list(args.distance_weights)

    rp = ResourceProvision(
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

    result = rp.resource_provisioning(kmin=args.kmin, kmax=args.kmax, linkage=args.linkage)

    out_dir = Path(args.output_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # pickle replicas
    pkl_path = out_dir / "resource_provisioning_result.pkl"
    with pkl_path.open("wb") as f:
        import pickle

        pickle.dump(result, f)

    # json for readability
    json_path = out_dir / "resource_provisioning_result.json"
    json_obj = [
            {
                "replica_id": r.replica_id,
                "adapter_ids": r.dataset_indices,
                "g": r.g,
                "T_iter": r.T_iter,
            }
            for r in result["replicas"]
    ]
    json_path.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")

    print(f"Saved resource provisioning result to:\n  {pkl_path}\n  {json_path}")


if __name__ == "__main__":
    main()

'''

python -m Cappuccino.ResourceProvision \
  --model_name "Qwen2.5-32B-Instruct" \
  --dataset_path "examples/dataset_distributions_16all_4096_seqlen_42_seed_1000_samples.json" \
  --adapter_config "examples/adapter_config.json" \
  --total_gpus 16 \
  --gpu_type "a100-40gb" \
  --output_dir "results"
'''