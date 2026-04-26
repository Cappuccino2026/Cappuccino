from __future__ import annotations

import argparse
import csv
import sys
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


class TimeCostModel:
    """Profile-based execution-time estimator for pipeline-parallel LoRA training.

    The model fits optimizer-step wall time from a combined profiling CSV. The
    current formulation assumes pipeline parallelism only, a uniform layer split,
    and one data-parallel replica in the profiling data.
    """

    REQUIRED_COLUMNS: Tuple[str, ...] = (
        "param_name",
        "pp_size",
        "global_bsz",
        "micro_bsz",
        "grad_acc_steps",
        "seqlen",
        "rank",
        "time_per_step_s",
        "status",
    )

    def __init__(
        self,
        model_name: str = "Llama-2-7b-hf",
        csv_name: str = "profile_pp_combined.csv",
        profile_path: Optional[Path] = None,
    ) -> None:
        self.popt: Optional[np.ndarray] = None
        self.X_data: Optional[Tuple[np.ndarray, ...]] = None
        self.y_data: Optional[np.ndarray] = None
        self.header: Optional[List[str]] = None

        self.model_name = model_name
        self.model_layers = MODEL_LAYERS_MAP.get(model_name)
        self.profile_path = profile_path or BASE_DIR
        self.csv_name = csv_name

        logger.info("Building time cost model | model={}", self.model_name)
        if self.profile_path.exists():
            self._read_from_profile()
        else:
            logger.warning("Profile directory not found; model fitting skipped | path={}", self.profile_path)

    def _read_from_profile(self) -> None:
        """Load profiling records, aggregate duplicate configs, and fit the model."""
        csv_path = self.profile_path / self.csv_name
        if not csv_path.exists():
            logger.warning("Profile CSV not found; model fitting skipped | path={}", csv_path)
            return

        config_agg: Dict[Tuple[int, int, int, int, int], List[float]] = {}

        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            self.header = reader.fieldnames
            if self.header is None:
                raise ValueError(f"Profile CSV is empty or malformed: {csv_path}")

            missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in self.header]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {csv_path}: {', '.join(missing_columns)}"
                )

            for row_idx, row in enumerate(reader, start=2):
                if row["param_name"].strip() != self.model_name:
                    continue
                if row["status"].strip() != "OK":
                    continue

                try:
                    pp_size = int(row["pp_size"])
                    global_bsz = int(row["global_bsz"])
                    micro_bsz = int(row["micro_bsz"])
                    grad_acc = int(row["grad_acc_steps"])
                    seqlen = int(row["seqlen"])
                    rank = int(row["rank"])
                    step_time = float(row["time_per_step_s"])
                except (TypeError, ValueError) as exc:
                    logger.debug("Skipping malformed profiling row | row={} | error={}", row_idx, exc)
                    continue

                if any(value <= 0 for value in (pp_size, global_bsz, micro_bsz, grad_acc, seqlen, rank, step_time)):
                    logger.debug("Skipping non-positive profiling row | row={}", row_idx)
                    continue

                # The profiler usually runs with DP=1, so grad_acc should equal
                # global_bsz / micro_bsz. We keep valid rows even when this sanity
                # condition is not met, because some profiles may use custom settings.
                key = (pp_size, micro_bsz, seqlen, rank, grad_acc)
                if key not in config_agg:
                    config_agg[key] = [step_time, 1.0]
                else:
                    config_agg[key][0] += step_time
                    config_agg[key][1] += 1.0

        pp_values: List[int] = []
        micro_bsz_values: List[int] = []
        seqlen_values: List[int] = []
        rank_values: List[int] = []
        grad_acc_values: List[int] = []
        step_time_values: List[float] = []

        for (pp_size, micro_bsz, seqlen, rank, grad_acc), (sum_time, count) in config_agg.items():
            pp_values.append(pp_size)
            micro_bsz_values.append(micro_bsz)
            seqlen_values.append(seqlen)
            rank_values.append(rank)
            grad_acc_values.append(grad_acc)
            step_time_values.append(sum_time / count)

        self.X_data = (
            np.asarray(pp_values, dtype=np.float64),
            np.asarray(micro_bsz_values, dtype=np.float64),
            np.asarray(seqlen_values, dtype=np.float64),
            np.asarray(rank_values, dtype=np.float64),
            np.asarray(grad_acc_values, dtype=np.float64),
        )
        self.y_data = np.asarray(step_time_values, dtype=np.float64)

        if self.y_data.size == 0:
            logger.warning("No valid profiling data found | model={}", self.model_name)
            return

        self._fit()

    def _fit(self) -> None:
        """Fit the parametric time model from aggregated profiling samples."""
        assert self.X_data is not None
        assert self.y_data is not None

        lower_bounds = [0.0] * 8
        upper_bounds = [np.inf] * 8

        try:
            self.popt, _ = curve_fit(
                self.curve_fit_func,
                self.X_data,
                self.y_data,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
        except Exception as exc:
            logger.exception("Curve fitting failed | model={} | error={}", self.model_name, exc)
            return

        self._log_fitting_error()

    @staticmethod
    def _sat(work: np.ndarray, vmax: float, w0: float) -> np.ndarray:
        """Saturating work-to-time transform used by the fitted model."""
        eps = 1e-9
        utilization = 1.0 - np.exp(-work / (w0 + eps))
        return work / (vmax * np.maximum(utilization, eps))

    def curve_fit_func(
        self,
        X: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        t0: float,
        vmax_base: float,
        w0_base: float,
        vmax_lora: float,
        w0_lora: float,
        a1: float,
        a2: float,
        a3: float,
    ) -> np.ndarray:
        """Compute predicted optimizer-step time for curve fitting.

        Args:
            X: Tuple of ``(pp_size, micro_bsz, seqlen, rank_eff, grad_acc)``.
            t0: Constant per-microbatch stage overhead.
            vmax_base: Saturation throughput for base-model work.
            w0_base: Saturation scale for base-model work.
            vmax_lora: Saturation throughput for LoRA work.
            w0_lora: Saturation scale for LoRA work.
            a1: Attention-work coefficient.
            a2: MLP-work coefficient.
            a3: LoRA-work coefficient.

        Returns:
            Predicted optimizer-step wall time in seconds.
        """
        pp_size, micro_bsz, seqlen, rank_eff, grad_acc = map(np.asarray, X)

        tokens = micro_bsz * seqlen
        attention_work = tokens * seqlen
        mlp_work = tokens
        lora_work = tokens * rank_eff

        base_work_stage = (a1 * attention_work + a2 * mlp_work) / pp_size
        lora_work_stage = (a3 * lora_work) / pp_size

        stage_time = (
            t0
            + self._sat(base_work_stage, vmax_base, w0_base)
            + self._sat(lora_work_stage, vmax_lora, w0_lora)
        )

        # Pipeline bubble model: one optimizer step takes grad_acc + pp_size - 1
        # stage-time units under standard 1F1B-style pipeline execution.
        return (grad_acc + pp_size - 1.0) * stage_time

    def _log_fitting_error(self) -> None:
        """Log basic regression quality metrics for the fitted model."""
        if self.popt is None or self.X_data is None or self.y_data is None:
            logger.warning("Model is not fitted; fitting metrics are unavailable.")
            return

        y_pred = self.curve_fit_func(self.X_data, *self.popt)
        mae = float(np.mean(np.abs(self.y_data - y_pred)))
        mape = float(np.mean(np.abs((self.y_data - y_pred) / self.y_data)) * 100.0)
        ss_res = float(np.sum((self.y_data - y_pred) ** 2))
        ss_tot = float(np.sum((self.y_data - np.mean(self.y_data)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

        logger.info(
            "Time model fit completed | model={} | samples={} | MAE={:.4f}s | R2={:.4f} | MAPE={:.2f}%",
            self.model_name,
            self.y_data.size,
            mae,
            r2,
            mape,
        )

    def step_execution_time_estimate(
        self,
        global_bsz: int,
        micro_bsz: int,
        seqlen: int,
        rank: int,
        tasknum: int,
        pp_size: int,
    ) -> float:
        """Predict optimizer-step wall time in seconds.

        ``tasknum`` is extrapolated by replacing the fitted LoRA rank with
        ``rank_eff = rank * tasknum``.
        """
        self._validate_positive_inputs(
            global_bsz=global_bsz,
            micro_bsz=micro_bsz,
            seqlen=seqlen,
            rank=rank,
            tasknum=tasknum,
            pp_size=pp_size,
        )
        if self.popt is None:
            raise ValueError("Time cost model is not fitted yet.")
        if global_bsz % micro_bsz != 0:
            raise ValueError("global_bsz must be divisible by micro_bsz under the DP=1 assumption.")

        grad_acc = global_bsz // micro_bsz
        rank_eff = rank * tasknum
        estimate = self.curve_fit_func(
            (pp_size, micro_bsz, seqlen, rank_eff, grad_acc),
            *self.popt,
        )
        return round(float(estimate), 4)

    def stage_execution_time_estimate(
        self,
        micro_bsz: int,
        seqlen: int,
        rank: int,
        tasknum: int,
        pp_size: int,
    ) -> float:
        """Predict per-stage execution time for one microbatch in seconds."""
        self._validate_positive_inputs(
            micro_bsz=micro_bsz,
            seqlen=seqlen,
            rank=rank,
            tasknum=tasknum,
            pp_size=pp_size,
        )
        if self.popt is None:
            raise ValueError("Time cost model is not fitted yet.")

        t0, vmax_base, w0_base, vmax_lora, w0_lora, a1, a2, a3 = self.popt
        rank_eff = rank * tasknum

        tokens = float(micro_bsz) * float(seqlen)
        attention_work = tokens * float(seqlen)
        mlp_work = tokens
        lora_work = tokens * float(rank_eff)

        base_work_stage = (a1 * attention_work + a2 * mlp_work) / float(pp_size)
        lora_work_stage = (a3 * lora_work) / float(pp_size)
        stage_time = (
            t0
            + self._sat(base_work_stage, vmax_base, w0_base)
            + self._sat(lora_work_stage, vmax_lora, w0_lora)
        )
        return round(float(stage_time), 6)

    def layer_execution_time_estimate(
        self,
        micro_bsz: int,
        seqlen: int,
        rank: int,
        tasknum: int,
        pp_size: int,
    ) -> float:
        """Predict per-layer execution time under a uniform pipeline split."""
        if self.model_layers is None:
            raise ValueError(f"Unknown layer count for model: {self.model_name}")

        stage_time = self.stage_execution_time_estimate(
            micro_bsz=micro_bsz,
            seqlen=seqlen,
            rank=rank,
            tasknum=tasknum,
            pp_size=pp_size,
        )
        layers_per_stage = self.model_layers / pp_size
        return round(stage_time / layers_per_stage, 8)

    @staticmethod
    def _validate_positive_inputs(**values: int) -> None:
        """Validate that all input dimensions are positive."""
        invalid = {name: value for name, value in values.items() if value <= 0}
        if invalid:
            raise ValueError(f"All inputs must be positive, but got: {invalid}")


def configure_logging(log_level: str = "INFO") -> None:
    """Configure loguru for consistent CLI and library logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for smoke testing."""
    parser = argparse.ArgumentParser(description="Fit and test a profile-based time cost model.")
    parser.add_argument("--model_name", default="Llama-2-7b-hf")
    parser.add_argument("--csv_name", default="profile_pp_combined.csv")
    parser.add_argument("--profile_path", default=None)
    parser.add_argument("--log_level", default="INFO", choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    """Run a small smoke test for the fitted time cost model."""
    args = build_argparser().parse_args()
    configure_logging(args.log_level)

    profile_path = Path(args.profile_path).resolve() if args.profile_path else None
    time_cost_model = TimeCostModel(
        model_name=args.model_name,
        csv_name=args.csv_name,
        profile_path=profile_path,
    )

    if time_cost_model.popt is None:
        logger.error("Time cost model is not fitted; skip estimation smoke test.")
        return

    logger.info("Running estimation smoke test.")
    micro_bsz_values = [1, 2, 4]
    seqlen_values = [512, 1024, 2048]
    rank_values = [16]
    tasknum_values = [1, 2, 3]
    pp_values = [1, 2, 3, 4]

    for pp_size in pp_values:
        for micro_bsz in micro_bsz_values:
            for seqlen in seqlen_values:
                for rank in rank_values:
                    for tasknum in tasknum_values:
                        step_time = time_cost_model.step_execution_time_estimate(
                            global_bsz=8,
                            micro_bsz=micro_bsz,
                            seqlen=seqlen,
                            rank=rank,
                            tasknum=tasknum,
                            pp_size=pp_size,
                        )
                        stage_time = time_cost_model.stage_execution_time_estimate(
                            micro_bsz=micro_bsz,
                            seqlen=seqlen,
                            rank=rank,
                            tasknum=tasknum,
                            pp_size=pp_size,
                        )
                        logger.info(
                            "Estimate | pp={} | global_bsz=8 | micro_bsz={} | seqlen={} | rank={} | tasknum={} | step_time={:.4f}s | stage_time={:.6f}s",
                            pp_size,
                            micro_bsz,
                            seqlen,
                            rank,
                            tasknum,
                            step_time,
                            stage_time,
                        )


if __name__ == "__main__":
    main()
