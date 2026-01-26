import os
import csv
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

base_dir = Path(__file__).resolve().parent

MODEL_LAYERS_MAP = {
    'TinyLlama-1.1B-Chat-v1.0': 25,
    'Llama-2-7b-hf': 35,
    'Llama-2-13b-hf': 40,
    'meta-llama/Llama-3.1-8B-Instruct': 32,
    'Qwen/Qwen2.5-32B-Instruct': 64,
}


class MemCostModel:
    """
    Profile-based memory model with pp_size as an input variable.
    CSV (combined) columns (from your run_profile_pp_combined.sh):
      param_name,tasknum,model_path,pp_size,world_size,global_bsz,micro_bsz,grad_acc_steps,seqlen,rank,...,peak_memory_mb,...,status
    We fit peak memory per-GPU (i.e., max stage) in GB.

    """
    def __init__(self, model_name: str = "Llama-2-7b-hf", csv_name: str = "profile_pp_combined.csv"):
        self.popt = None
        self.X_data = None
        self.y_data = None
        self.header = None

        self.model_name = model_name
        self.model_layers = MODEL_LAYERS_MAP.get(model_name)

        self.profile_path = base_dir
        self.csv_name = csv_name

        print(f"Building mem cost model for model_name={self.model_name} ...")
        if os.path.exists(self.profile_path):
            self._read_from_profile()
        else:
            print(f"[WARN] Cannot find profile_path: {self.profile_path}. Build failed.")

    def _read_from_profile(self):
        csv_path = self.profile_path / self.csv_name
        if not csv_path.exists():
            print(f"[WARN] Profile CSV not found: {csv_path}")
            return

        # key -> [sum_mem_gb, count]
        # key uses the real modeling variables to merge duplicates
        config_agg: dict[tuple, list] = {}

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            self.header = reader.fieldnames
            print(f"CSV header: {self.header}")

            required = ["param_name", "pp_size", "micro_bsz", "seqlen", "rank", "peak_memory_mb", "status"]
            for k in required:
                if k not in reader.fieldnames:
                    raise ValueError(f"Missing required column '{k}' in {csv_path}")

            for row in reader:
                # 1) filter by model
                if row["param_name"].strip() != self.model_name:
                    continue

                # 2) filter status
                status = row["status"].strip()
                if status != "OK":
                    # skip OOM / invalid / no_metrics
                    continue

                # 3) parse fields
                try:
                    pp = int(row["pp_size"])
                    micro_bsz = int(row["micro_bsz"])
                    seqlen = int(row["seqlen"])
                    rank = int(row["rank"])
                    peak_mem_mb = float(row["peak_memory_mb"])
                except Exception as e:
                    print(f"[WARN] Bad row: {row} err={e}")
                    continue

                if pp <= 0 or micro_bsz <= 0 or seqlen <= 0 or rank <= 0 or peak_mem_mb <= 0:
                    continue

                peak_mem_gb = peak_mem_mb / 1024.0

                # (optional) tasknum column might exist but often is just an id in your CSV;
                # here we fit single-task behavior, so we keep tasknum as an input at inference time only.
                key = (pp, micro_bsz, seqlen, rank)

                if key not in config_agg:
                    config_agg[key] = [peak_mem_gb, 1]
                else:
                    config_agg[key][0] += peak_mem_gb
                    config_agg[key][1] += 1

        pp_list, micro_list, seqlen_list, rank_list, mem_list = [], [], [], [], []
        for (pp, micro_bsz, seqlen, rank), (sum_mem, cnt) in config_agg.items():
            pp_list.append(pp)
            micro_list.append(micro_bsz)
            seqlen_list.append(seqlen)
            rank_list.append(rank)
            mem_list.append(sum_mem / cnt)

        self.X_data = (
            np.array(pp_list, dtype=np.float64),
            np.array(micro_list, dtype=np.float64),
            np.array(seqlen_list, dtype=np.float64),
            np.array(rank_list, dtype=np.float64),
        )
        self.y_data = np.array(mem_list, dtype=np.float64)

        print(f"Loaded {len(self.y_data)} unique configs for model_name={self.model_name}")

        if len(self.y_data) == 0:
            print(f"[WARN] No valid data found for model_name={self.model_name}")
            return

        # Fit: mem_stage = c0 + (c1 + c2*(micro*seqlen) + c3*(rank)) / pp
        # (rank term learned with tasknum=1; later inference can scale by tasknum if desired)
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
            print("\nCurve fitting successful!")
            print("Fitted parameters (popt):", self.popt)
            self._calculate_fitting_error()
            print("Memory cost model built successfully!")
        except Exception as e:
            print(f"Curve fitting failed: {e}")

    @staticmethod
    def _curve_fit_func(X, c0, c1, c2, c3):
        """
        X = (pp, micro_bsz, seqlen, rank)
        mem_stage(GB) = c0 + (c1 + c2*(micro_bsz*seqlen) + c3*rank) / pp
        """
        pp, micro_bsz, seqlen, rank = X
        pp, micro_bsz, seqlen, rank = map(np.asarray, (pp, micro_bsz, seqlen, rank))

        tokens = micro_bsz * seqlen
        return c0 + (c1 + c2 * tokens + c3 * rank) / pp

    def _calculate_fitting_error(self):
        if self.popt is None or self.X_data is None or self.y_data is None:
            print("Model is not fitted yet. Cannot calculate error.")
            return

        y_pred = self._curve_fit_func(self.X_data, *self.popt)

        mae = np.mean(np.abs(self.y_data - y_pred))
        mape = np.mean(np.abs((self.y_data - y_pred) / self.y_data)) * 100
        ss_res = np.sum((self.y_data - y_pred) ** 2)
        ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print("\n--- Memory Model Fit Evaluation ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f} GB")
        print(f"R-squared (R²) Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("--------------------------\n")

    def stage_memory_estimate(
        self,
        bsz: int,
        seqlen: int,
        rank: int,
        tasknum: int,
        pp_size: int,
    ) -> float:
        """
        Estimate peak memory per-stage (per GPU), in GB.
        Note: tasknum is extrapolated linearly on LoRA term (rank*tasknum),
              since training data is typically tasknum=1.
        """
        if bsz <= 0 or seqlen <= 0 or rank <= 0 or tasknum <= 0 or pp_size <= 0:
            raise ValueError("bsz, seqlen, rank, tasknum, pp_size must be positive.")
        if self.popt is None:
            raise ValueError("Memory cost model is not fitted yet.")

        c0, c1, c2, c3 = self.popt
        tokens = float(bsz) * float(seqlen)

        # Replace rank with rank*tasknum for multi-task extrapolation
        mem = c0 + (c1 + c2 * tokens + c3 * float(rank) * float(tasknum)) / float(pp_size)
        return round(float(mem), 3)


if __name__ == "__main__":
    model_name = "Llama-2-13b-hf"
    mem_cost_model = MemCostModel(model_name=model_name, csv_name="profile_pp_combined.csv")

    if mem_cost_model.popt is not None:
        print("\n--- Estimating memory for new configurations ---")
        bsz_test = [1, 2, 4]
        seqlen_test = [512, 1024, 2048]
        rank_test = [8, 16, 32]
        tasknum_test = [1, 2, 4]
        pp_test = [1, 2, 3, 4]

        for pp in pp_test:
            for bsz in bsz_test:
                for seqlen in seqlen_test:
                    for rank in rank_test:
                        for tasknum in tasknum_test:
                            est = mem_cost_model.stage_memory_estimate(
                                bsz=bsz, seqlen=seqlen, rank=rank, tasknum=tasknum, pp_size=pp
                            )
                            print(f"pp={pp}, bsz={bsz}, seqlen={seqlen}, rank={rank}, tasknum={tasknum} -> {est:.3f} GB/stage")
