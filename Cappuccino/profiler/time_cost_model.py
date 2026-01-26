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


class TimeCostModel:
    """
    Profile-based time cost model with pp_size as an input variable.
    Fit target: time_per_step_s from profile_pp_combined.csv (optimizer step wall time).
    Assumption: PP-only, uniform partition.
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

        print(f"Building time cost model for model_name={self.model_name} ...")
        if os.path.exists(self.profile_path):
            self._read_from_profile()
        else:
            print(f"[WARN] Cannot find profile_path: {self.profile_path}. Build failed.")

    def _read_from_profile(self):
        csv_path = self.profile_path / self.csv_name
        if not csv_path.exists():
            print(f"[WARN] Profile CSV not found: {csv_path}")
            return

        # Merge duplicates: key -> [sum_time, count]
        # key uses modeling variables
        config_agg: dict[tuple, list] = {}

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            self.header = reader.fieldnames
            print(f"CSV header: {self.header}")

            required = ["param_name", "pp_size", "global_bsz", "micro_bsz", "grad_acc_steps", "seqlen", "rank",
                        "time_per_step_s", "status"]
            for k in required:
                if k not in reader.fieldnames:
                    raise ValueError(f"Missing required column '{k}' in {csv_path}")

            for row in reader:
                if row["param_name"].strip() != self.model_name:
                    continue
                if row["status"].strip() != "OK":
                    continue

                try:
                    pp = int(row["pp_size"])
                    global_bsz = int(row["global_bsz"])
                    micro_bsz = int(row["micro_bsz"])
                    grad_acc = int(row["grad_acc_steps"])
                    seqlen = int(row["seqlen"])
                    rank = int(row["rank"])
                    t_step = float(row["time_per_step_s"])
                except Exception:
                    continue

                if pp <= 0 or global_bsz <= 0 or micro_bsz <= 0 or grad_acc <= 0 or seqlen <= 0 or rank <= 0:
                    continue
                if t_step <= 0:
                    continue

                # sanity: DP=1 in your profiler, so grad_acc should equal global/micro
                # if not, still keep row but you can uncomment to enforce:
                # if global_bsz != micro_bsz * grad_acc:
                #     continue

                key = (pp, micro_bsz, seqlen, rank, grad_acc)
                if key not in config_agg:
                    config_agg[key] = [t_step, 1]
                else:
                    config_agg[key][0] += t_step
                    config_agg[key][1] += 1

        pp_list, micro_list, seqlen_list, rank_list, gradacc_list, time_list = [], [], [], [], [], []
        for (pp, micro_bsz, seqlen, rank, grad_acc), (sum_t, cnt) in config_agg.items():
            pp_list.append(pp)
            micro_list.append(micro_bsz)
            seqlen_list.append(seqlen)
            rank_list.append(rank)          # tasknum=1 during fitting; inference can use rank_eff=rank*tasknum
            gradacc_list.append(grad_acc)
            time_list.append(sum_t / cnt)

        self.X_data = (
            np.array(pp_list, dtype=np.float64),
            np.array(micro_list, dtype=np.float64),
            np.array(seqlen_list, dtype=np.float64),
            np.array(rank_list, dtype=np.float64),
            np.array(gradacc_list, dtype=np.float64),
        )
        self.y_data = np.array(time_list, dtype=np.float64)

        print(f"Loaded {len(self.y_data)} unique configs for model_name={self.model_name}")

        if len(self.y_data) == 0:
            print(f"[WARN] No valid data found for model_name={self.model_name}")
            return

        # Fit params: (t0, vmax_base, w0_base, vmax_lora, w0_lora, a1, a2, a3)
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
            print("\nCurve fitting successful!")
            print("Fitted parameters (popt):", self.popt)
            self._calculate_fitting_error()
        except Exception as e:
            print(f"Curve fitting failed: {e}")

    @staticmethod
    def _sat(work, vmax, w0):
        eps = 1e-9
        util = 1.0 - np.exp(-work / (w0 + eps))
        return work / (vmax * np.maximum(util, eps))

    def curve_fit_func(self, X,
                       t0,
                       vmax_base, w0_base,
                       vmax_lora, w0_lora,
                       a1, a2, a3):
        """
        X = (pp, micro_bsz, seqlen, rank_eff, grad_acc)

        time_per_step = (m + pp - 1) * t_stage_micro

        t_stage_micro = t0 + sat(base_work_stage; vmax_base, w0_base)
                          + sat(lora_work_stage; vmax_lora, w0_lora)

        base_work_stage = (a1*attn_work + a2*mlp_work) / pp
        lora_work_stage = (a3*lora_work) / pp
        """
        pp, micro_bsz, seqlen, rank_eff, grad_acc = X
        pp, micro_bsz, seqlen, rank_eff, grad_acc = map(np.asarray, (pp, micro_bsz, seqlen, rank_eff, grad_acc))

        tokens = micro_bsz * seqlen
        attn_work = tokens * seqlen          # ~ O(micro_bsz * seqlen^2)
        mlp_work  = tokens                   # ~ O(micro_bsz * seqlen)
        lora_work = tokens * rank_eff        # ~ O(micro_bsz * seqlen * rank_eff)

        base_work_stage = (a1 * attn_work + a2 * mlp_work) / pp
        lora_work_stage = (a3 * lora_work) / pp

        t_stage_micro = t0 + self._sat(base_work_stage, vmax_base, w0_base) + self._sat(lora_work_stage, vmax_lora, w0_lora)

        # bubble: (m + pp - 1) microbatch-times
        return (grad_acc + pp - 1.0) * t_stage_micro

    def _calculate_fitting_error(self):
        if self.popt is None or self.X_data is None or self.y_data is None:
            print("Model is not fitted yet. Cannot calculate error.")
            return

        y_pred = self.curve_fit_func(self.X_data, *self.popt)

        mae = np.mean(np.abs(self.y_data - y_pred))
        mape = np.mean(np.abs((self.y_data - y_pred) / self.y_data)) * 100
        ss_res = np.sum((self.y_data - y_pred) ** 2)
        ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print("\n--- Time Model Fit Evaluation ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f} s")
        print(f"R-squared (R²) Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("--------------------------\n")

    # ---------- public APIs ----------
    def step_execution_time_estimate(self, global_bsz: int, micro_bsz: int, seqlen: int, rank: int, tasknum: int, pp_size: int) -> float:
        """
        Predict optimizer-step wall time (seconds), aligned with CSV time_per_step_s.
        tasknum is NOT from CSV; it's your "number of concurrent LoRA tasks".
        We extrapolate LoRA part by rank_eff = rank * tasknum.
        """
        if self.popt is None:
            raise ValueError("Model parameters are not fitted yet.")
        if any(v <= 0 for v in [global_bsz, micro_bsz, seqlen, rank, tasknum, pp_size]):
            raise ValueError("all inputs must be positive.")

        if global_bsz % micro_bsz != 0:
            raise ValueError("global_bsz must be divisible by micro_bsz (DP=1 assumption).")

        grad_acc = global_bsz // micro_bsz
        rank_eff = rank * tasknum

        t = self.curve_fit_func((pp_size, micro_bsz, seqlen, rank_eff, grad_acc), *self.popt)
        return round(float(t), 4)

    def stage_execution_time_estimate(self, micro_bsz: int, seqlen: int, rank: int, tasknum: int, pp_size: int) -> float:
        """
        Predict per-stage time for ONE microbatch (seconds).
        Useful if you want to compose pipeline makespan yourself.
        """
        if self.popt is None:
            raise ValueError("Model parameters are not fitted yet.")
        if any(v <= 0 for v in [micro_bsz, seqlen, rank, tasknum, pp_size]):
            raise ValueError("all inputs must be positive.")

        t0, vmax_base, w0_base, vmax_lora, w0_lora, a1, a2, a3 = self.popt
        rank_eff = rank * tasknum

        tokens = float(micro_bsz) * float(seqlen)
        attn_work = tokens * float(seqlen)
        mlp_work  = tokens
        lora_work = tokens * float(rank_eff)

        base_work_stage = (a1 * attn_work + a2 * mlp_work) / float(pp_size)
        lora_work_stage = (a3 * lora_work) / float(pp_size)

        t_stage = t0 + self._sat(base_work_stage, vmax_base, w0_base) + self._sat(lora_work_stage, vmax_lora, w0_lora)
        return round(float(t_stage), 6)

    def layer_execution_time_estimate(self, micro_bsz: int, seqlen: int, rank: int, tasknum: int, pp_size: int) -> float:
        """
        Predict per-layer time (seconds) under uniform PP split.
        layers_per_stage = total_layers / pp_size
        """
        if self.model_layers is None:
            raise ValueError(f"Unknown layer count for model {self.model_name}.")
        t_stage = self.stage_execution_time_estimate(micro_bsz, seqlen, rank, tasknum, pp_size)
        layers_per_stage = self.model_layers / pp_size
        return round(t_stage / layers_per_stage, 8)


if __name__ == "__main__":
    # model_name = "Llama-2-13b-hf"
    model_name = "Llama-2-7b-hf"

    
    time_cost_model = TimeCostModel(model_name=model_name, csv_name="profile_pp_combined.csv")

    if time_cost_model.popt is not None:
        microbsz_test = [1, 2, 4]
        seqlen_test = [512, 1024, 2048]
        rank_test = [16]
        tasknum_test = [1,2,3]
        pp_test = [1, 2, 3, 4]

        for pp in pp_test:
            for microbsz in microbsz_test:
                for seqlen in seqlen_test:
                    for rank in rank_test:
                        for tasknum in tasknum_test:
                            est_step = time_cost_model.step_execution_time_estimate(
                                global_bsz=8, micro_bsz=microbsz, seqlen=seqlen, rank=rank, tasknum=tasknum, pp_size=pp
                            )
                            print(f"pp={pp}, global_bsz=8, micro_bsz={microbsz}, seqlen={seqlen}, rank={rank}, tasknum={tasknum} -> step_time = {est_step:.3f} s")

                            est_micro_batch_stage = time_cost_model.stage_execution_time_estimate(
                                micro_bsz=microbsz, seqlen=seqlen, rank=rank, tasknum=tasknum, pp_size=pp
                            )
                            print(f"pp={pp}, micro_bsz={microbsz}, seqlen={seqlen}, rank={rank}, tasknum={tasknum} -> step_time = {est_micro_batch_stage:.3f} s")


                            