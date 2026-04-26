from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.optimize import curve_fit

BASE_DIR = Path(__file__).resolve().parent

MODEL_LAYERS_MAP: Dict[str, int] = {
    "TinyLlama-1.1B-Chat-v1.0": 25,
    "Llama-2-7b-hf": 35,
    "Llama-2-13b-hf": 40,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "Qwen/Qwen2.5-32B-Instruct": 64,
}

REQUIRED_PROFILE_COLUMNS = (
    "param_name",
    "pp_size",
    "micro_bsz",
    "seqlen",
    "rank",
    "peak_memory_mb",
    "status",
)


class MemCostModel:
    """Profile-based per-stage GPU memory cost model.

    The model fits peak per-GPU memory usage in GB from a combined pipeline-parallel
    profiling CSV. The expected CSV contains columns such as model name, pipeline
    size, micro-batch size, sequence length, LoRA rank, peak memory, and status.
    """

    def __init__(
        self,
        model_name: str = "Llama-2-7b-hf",
        csv_name: str = "profile_pp_combined.csv",
        profile_dir: Optional[Path | str] = None,
    ) -> None:
        self.popt: Optional[np.ndarray] = None
        self.X_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self.y_data: Optional[np.ndarray] = None
        self.header: Optional[List[str]] = None

        self.model_name = model_name
        self.model_layers = MODEL_LAYERS_MAP.get(model_name)
        self.profile_dir = Path(profile_dir) if profile_dir is not None else BASE_DIR
        self.csv_name = csv_name

        logger.info(
            "Building memory cost model: model={}, profile_csv={}",
            self.model_name,
            self.profile_dir / self.csv_name,
        )
        self._read_from_profile()

    def _read_from_profile(self) -> None:
        """Load profiling data, aggregate duplicate configurations, and fit the model."""
        csv_path = self.profile_dir / self.csv_name
        if not csv_path.exists():
            logger.warning("Profile CSV not found: {}", csv_path)
            return

        config_agg: Dict[Tuple[int, int, int, int], List[float]] = {}

        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            self.header = reader.fieldnames

            if self.header is None:
                raise ValueError(f"Profile CSV is empty or invalid: {csv_path}")

            missing_columns = [column for column in REQUIRED_PROFILE_COLUMNS if column not in self.header]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {csv_path}: {', '.join(missing_columns)}"
                )

            for row_idx, row in enumerate(reader, start=2):
                if row["param_name"].strip() != self.model_name:
                    continue

                if row["status"].strip() != "OK":
                    continue

                parsed = self._parse_profile_row(row=row, row_idx=row_idx)
                if parsed is None:
                    continue

                pp_size, micro_bsz, seqlen, rank, peak_mem_gb = parsed
                key = (pp_size, micro_bsz, seqlen, rank)
                if key not in config_agg:
                    config_agg[key] = [peak_mem_gb, 1.0]
                else:
                    config_agg[key][0] += peak_mem_gb
                    config_agg[key][1] += 1.0

        self._build_training_arrays(config_agg=config_agg)

        if self.y_data is None or self.y_data.size == 0:
            logger.warning("No valid memory profile data found for model={}", self.model_name)
            return

        self._fit_curve()

    @staticmethod
    def _parse_profile_row(
        row: Dict[str, str],
        row_idx: int,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """Parse one valid profiling row into model features and memory target."""
        try:
            pp_size = int(row["pp_size"])
            micro_bsz = int(row["micro_bsz"])
            seqlen = int(row["seqlen"])
            rank = int(row["rank"])
            peak_mem_mb = float(row["peak_memory_mb"])
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed profile row {}: {}", row_idx, exc)
            return None

        if pp_size <= 0 or micro_bsz <= 0 or seqlen <= 0 or rank <= 0 or peak_mem_mb <= 0:
            logger.debug("Skipping non-positive profile row {}: {}", row_idx, row)
            return None

        return pp_size, micro_bsz, seqlen, rank, peak_mem_mb / 1024.0

    def _build_training_arrays(
        self,
        config_agg: Dict[Tuple[int, int, int, int], List[float]],
    ) -> None:
        """Convert aggregated profiling statistics into fitting arrays."""
        pp_list: List[int] = []
        micro_bsz_list: List[int] = []
        seqlen_list: List[int] = []
        rank_list: List[int] = []
        memory_list: List[float] = []

        for (pp_size, micro_bsz, seqlen, rank), (sum_mem_gb, count) in config_agg.items():
            pp_list.append(pp_size)
            micro_bsz_list.append(micro_bsz)
            seqlen_list.append(seqlen)
            rank_list.append(rank)
            memory_list.append(sum_mem_gb / count)

        self.X_data = (
            np.asarray(pp_list, dtype=np.float64),
            np.asarray(micro_bsz_list, dtype=np.float64),
            np.asarray(seqlen_list, dtype=np.float64),
            np.asarray(rank_list, dtype=np.float64),
        )
        self.y_data = np.asarray(memory_list, dtype=np.float64)

        logger.info(
            "Loaded {} valid memory-profile configurations for model={}",
            self.y_data.size,
            self.model_name,
        )

    def _fit_curve(self) -> None:
        """Fit the memory model coefficients using bounded nonlinear least squares."""
        assert self.X_data is not None
        assert self.y_data is not None

        lower_bounds = [0.0, 0.0, 0.0, 0.0]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]

        try:
            self.popt, _ = curve_fit(
                self._curve_fit_func,
                self.X_data,
                self.y_data,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
        except Exception as exc:
            logger.exception("Curve fitting failed for memory cost model: {}", exc)
            return

        self._log_fitting_metrics()

    @staticmethod
    def _curve_fit_func(
        X: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        c0: float,
        c1: float,
        c2: float,
        c3: float,
    ) -> np.ndarray:
        """Memory curve: mem_stage = c0 + (c1 + c2 * tokens + c3 * rank) / pp_size."""
        pp_size, micro_bsz, seqlen, rank = map(np.asarray, X)
        tokens = micro_bsz * seqlen
        return c0 + (c1 + c2 * tokens + c3 * rank) / pp_size

    def _log_fitting_metrics(self) -> None:
        """Log fitting metrics for the trained memory model."""
        if self.popt is None or self.X_data is None or self.y_data is None:
            logger.warning("Memory cost model is not fitted; skipping evaluation metrics.")
            return

        y_pred = self._curve_fit_func(self.X_data, *self.popt)
        mae = float(np.mean(np.abs(self.y_data - y_pred)))
        mape = float(np.mean(np.abs((self.y_data - y_pred) / self.y_data)) * 100.0)
        ss_res = float(np.sum((self.y_data - y_pred) ** 2))
        ss_tot = float(np.sum((self.y_data - np.mean(self.y_data)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        logger.info(
            "Memory model fit metrics: MAE={:.4f} GB, R2={:.4f}, MAPE={:.2f}%",
            mae,
            r2,
            mape,
        )

    def stage_memory_estimate(
        self,
        bsz: int,
        seqlen: int,
        rank: int,
        tasknum: int,
        pp_size: int,
    ) -> float:
        """Estimate peak per-stage GPU memory in GB.

        The LoRA term is extrapolated linearly with ``tasknum`` because the profile
        data is typically collected from single-task configurations.
        """
        if bsz <= 0 or seqlen <= 0 or rank <= 0 or tasknum <= 0 or pp_size <= 0:
            raise ValueError("bsz, seqlen, rank, tasknum, and pp_size must be positive.")
        if self.popt is None:
            raise ValueError("Memory cost model is not fitted yet.")

        c0, c1, c2, c3 = self.popt
        tokens = float(bsz) * float(seqlen)
        memory_gb = c0 + (
            c1 + c2 * tokens + c3 * float(rank) * float(tasknum)
        ) / float(pp_size)
        return round(float(memory_gb), 3)


def run_smoke_test() -> None:
    """Run a small estimation sweep for manual validation."""
    model_name = "Llama-2-13b-hf"
    mem_cost_model = MemCostModel(model_name=model_name, csv_name="profile_pp_combined.csv")

    if mem_cost_model.popt is None:
        logger.warning("Smoke test skipped because the memory model was not fitted.")
        return

    logger.info("Running memory-estimation smoke test.")
    for pp_size in [1, 2, 3, 4]:
        for bsz in [1, 2, 4]:
            for seqlen in [512, 1024, 2048]:
                for rank in [8, 16, 32]:
                    for tasknum in [1, 2, 4]:
                        estimate = mem_cost_model.stage_memory_estimate(
                            bsz=bsz,
                            seqlen=seqlen,
                            rank=rank,
                            tasknum=tasknum,
                            pp_size=pp_size,
                        )
                        logger.info(
                            "Memory estimate | pp_size={} | bsz={} | seqlen={} | rank={} | tasknum={} | peak={:.3f} GB/stage",
                            pp_size,
                            bsz,
                            seqlen,
                            rank,
                            tasknum,
                            estimate,
                        )


if __name__ == "__main__":
    run_smoke_test()
