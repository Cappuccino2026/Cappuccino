# Cappuccino

Cappuccino is a replica-level orchestration framework for heterogeneous multi-LoRA fine-tuning, which jointly optimizes GPU provisioning and pipeline execution to minimize overall GPU time.

## Overview

Cappuccino integrates three main modules: Cappuccino Profiler, GPU Allocator, and Pipeline Scheduler. Cappuccino performs inter-replica and intra-replica optimizations based on a performance model for heterogeneous multi-LoRA workloads. It first profiles the target backbone model to obtain latency and GPU memory statistics under different configurations, and uses these measurements to construct a performance model for multi-LoRA fine-tuning. Guided by this model, a workload-aware GPU allocator clusters LoRA adapters according to their resource demands and allocates a cost-efficient number of GPUs to each replica, minimizing the total GPU time spent across all replicas. On top of this replica configuration, a hetero-LoRA pipeline scheduler then generates per-replica pipeline execution plans that pack and reorder micro-batches to reduce pipeline bubbles and padding overhead, further shortening the iteration time for heterogeneous multi-LoRA workloads.

![Cappuccino Overview](images/overview.pdf)

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python >= 3.8
- torch 2.6.0+cu124
- transformers 4.51.1
- peft 0.15.0
- megatron-core 0.11.0
- flash-attn 2.7.4.post1
- datasets 3.4.1

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Cappuccino2026/Cappuccino.git
cd Cappuccino
pip install -r requirements.txt
```

## Running the Prototype System

Cappuccino follows an end-to-end workflow:

1. Profile
2. Fine-tuning configuration
3. Resource provisioning
4. Pipeline execution scheduling

### Profile

The profiling results are stored in `Cappuccino/profiler/profile_pp_combined.csv`, which is used to construct the GPU memory and latency performance model for the subsequent orchestration.

If `profile_pp_combined.csv` does not contain profiling results for your target backbone, run the lightweight profiling script to obtain the target backbone profiling results:

```bash
bash Cappuccino/scripts/run_profile.sh
```

### Fine-tuning Configuration

Prepare the multi-LoRA fine-tuning workload by submitting (i) the datasets to train on and (ii) the LoRA adapter configurations (i.e., the set of LoRA fine-tuning jobs). Cappuccino uses this workload specification as input for subsequent replica provisioning and pipeline schedule construction.

**Datasets**

Prepare the dataset distribution file, e.g. `dataset_example.json`. You can load required datasets by specifying `DATASET_INFO` in `Cappuccino/generate_workloads.py`. Generate an example dataset distribution file:

```bash
python -m Cappuccino.generate_datasets \
  --num-samples 1000 \
  --output-filename dataset_example.json \
  --datasets gsm8k,wikisum,cnn_dailymail
```

**Adapter configurations**

Prepare an adapter configuration file (e.g., `adapter_config.json`) that defines the LoRA fine-tuning jobs to be co-trained, including the dataset assignment (`dataset_idx`) and per-adapter training hyperparameters such as global batch size, micro-batch size, rank, target modules, dropout, and alpha.

Example `adapter_config.json`:

```json
{
  "adapters": [
    {
      "dataset_idx": 0,
      "global_batch_size": 8,
      "rank": 16,
      "microbatch_size": 2,
      "target_modules": "q_proj+k_proj+v_proj+o_proj",
      "dropout": 0.05,
      "alpha": 32
    },
    {
      "dataset_idx": 1,
      "global_batch_size": 8,
      "rank": 16,
      "microbatch_size": 2,
      "target_modules": "q_proj+k_proj+v_proj+o_proj",
      "dropout": 0.05,
      "alpha": 32
    }
  ]
}
```

### Resource Provisioning

Run the resource provisioning module to compute a replica-level allocation plan:

```bash
python -m Cappuccino.ResourceProvision \
  --model_name "Qwen2.5-32B-Instruct" \
  --dataset_path "examples/dataset_example.json" \
  --adapter_config "examples/adapter_config.json" \
  --total_gpus 16 \
  --gpu_type "a100-40gb" \
  --output_dir "results"
```

After execution, the resource provisioning plan will be saved under `Cappuccino/results/`.

Example output:

```json
[
  {
    "replica_id": 0,
    "adapter_ids": [0],
    "g": 3
  },
  {
    "replica_id": 1,
    "adapter_ids": [1, 2, 3],
    "g": 2
  }
]
```

### Pipeline Execution Scheduling

For a given replica, run the pipeline execution scheduling:

```bash
python -m Cappuccino.PipelineExecutionConstruction \
  --dataset_path examples/dataset_example.json \
  --num_adapters 4 \
  --num_pipeline_stages 4 \
  --adapter_to_dataset_idx "0,4,8,12" \
  --adapter_to_global_batch_size "8,8,8,8" \
  --adapter_to_microbatch_size "1,2,4,1" \
  --model_name Qwen2.5-32B-Instruct \
  --output_name cappuccino_schedule
```

After execution, the generated pipeline schedules will be saved under `Cappuccino/schedules/`.

### Example

We provide a Python script for generating an orchestration plan for multi-LoRA fine-tuning based on `Qwen2.5-32B-Instruct`. Execute the following commands:

```bash
cd examples/Qwen2.5-32B-Instruct
python Qwen2.5_32B_Instruct_example.py
```