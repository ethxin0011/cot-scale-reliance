"""Central configuration for the CoT-reliance scaling-law project.

Every other script imports from here. Edit this file only.
Secrets are NEVER stored here: HF_TOKEN is read from environment variables.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------- Azure ML
SUBSCRIPTION_ID = ""
RESOURCE_GROUP = ""
WORKSPACE = ""
CPU_INSTANCE = "ralir-v3d-control"          # Standard_E48s_v3 (48 vCPU, 384 GB RAM)
GPU_CLUSTER = "t4v3"                        # Standard_NC4as_T4_v3 (1x T4 16 GB, 4 vCPU, 28 GB RAM)
DATASTORE = "workspaceblobstore"
BLOB_PREFIX = "cot_scaling"                 # all blobs live under this folder
EXPERIMENT_NAME = "cot-reliance-scaling"
GPU_ENV_NAME = "cot-scaling-gpu"
GPU_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"

# ----------------------------------------------------------------------------- Paths
NOTEBOOK_ROOT = Path("/home/azureuser")
PROJECT_DIR = NOTEBOOK_ROOT / "cot_scaling"
SRC_DIR = PROJECT_DIR / "src"
ENV_DIR = PROJECT_DIR / "env"
DATA_ROOT = Path(os.environ.get("COT_DATA_ROOT", str(PROJECT_DIR / "data")))  # small artifacts (persistent)
# Large temporary files (model downloads) go to the compute-instance local temp disk.
LOCAL_SCRATCH = Path(os.environ.get("COT_SCRATCH", "/mnt/cot_scratch"))


def data_dir(tag: str) -> Path:
    p = DATA_ROOT / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def blob_models(key: str) -> str:
    return f"{BLOB_PREFIX}/models/{key}"


def blob_prompts(tag: str, key: str) -> str:
    return f"{BLOB_PREFIX}/{tag}/prompts/{key}.parquet"


def blob_results(tag: str, key: str) -> str:
    return f"{BLOB_PREFIX}/{tag}/results/{key}"


# ----------------------------------------------------------------------------- Experiment
SEED = 20261018
K_GRID = [2, 4, 8, 12, 16, 24]              # number of swap operations (difficulty)
N_PER_K = 300                               # main run: 1,800 problems x 4 conditions = 7,200 generations / model
PILOT_N_PER_K = 30                          # pilot run
CONDITIONS = ["cot", "direct", "edit", "fresh"]

# ----------------------------------------------------------------------------- GPU job robustness
CHECKPOINT_MINUTES = 15                     # results flushed to blob at least every 15 min
MAX_BATCH = 64                              # hard cap on sequences per batch
GPU_MEM_FRACTION = 0.70                     # fraction of free memory used for the first token budget
MAX_JOB_ATTEMPTS = 12                       # watcher resubmits a preempted/failed job up to this many times
MAX_CONCURRENT_GPU_JOBS = 1                 # 1 T4
WATCH_POLL_SECONDS = 300

# OLMo 2 intermediate pretraining checkpoints: picked at these quantiles of the
# available stage-1 checkpoints, plus the final "main" model.
OLMO_CKPT_QUANTILES = [0.10, 0.25, 0.50, 0.75]


# ----------------------------------------------------------------------------- Model registry
def _m(key, hf, family, kind, quant, dtype, D, split, rev="main"):
    return dict(key=key, hf=hf, family=family, kind=kind, quant=quant,
                dtype=dtype, D=D, split=split, rev=rev)


# D = pretraining tokens. Values are from the public technical reports / model cards;
# verify before submission and edit here if a report gives a different figure.
#   Qwen3: ~36T for all dense sizes; OLMo 2: 1B 4T, 7B ~4T, 13B ~5T (intermediate
#   checkpoints are parsed from the revision name); Gemma 3: 1B 2T, 4B 4T, 12B 12T;
#   Llama 3.2 1B/3B: up to 9T; Llama 3.1 8B: ~15T.
# split: "fit" = used to fit the law; "heldout" = predicted out-of-sample;
#        "control" = quantization control only.
# quant: "none" or "nf4" (bitsandbytes 4-bit). dtype: compute dtype on the T4 (no bf16).
MODELS = [
    # --- Qwen3 post-trained (run in NON-thinking mode with explicit step format)
    _m("qwen3-0.6b", "Qwen/Qwen3-0.6B", "qwen3", "chat", "none", "float16", 36e12, "fit"),
    _m("qwen3-1.7b", "Qwen/Qwen3-1.7B", "qwen3", "chat", "none", "float16", 36e12, "fit"),
    _m("qwen3-4b", "Qwen/Qwen3-4B", "qwen3", "chat", "none", "float16", 36e12, "fit"),
    _m("qwen3-8b", "Qwen/Qwen3-8B", "qwen3", "chat", "nf4", "float16", 36e12, "fit"),
    _m("qwen3-14b", "Qwen/Qwen3-14B", "qwen3", "chat", "nf4", "float16", 36e12, "fit"),
    _m("qwen3-4b-nf4", "Qwen/Qwen3-4B", "qwen3", "chat", "nf4", "float16", 36e12, "control"),
    # --- Qwen3 base
    _m("qwen3-0.6b-base", "Qwen/Qwen3-0.6B-Base", "qwen3", "base", "none", "float16", 36e12, "fit"),
    _m("qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base", "qwen3", "base", "none", "float16", 36e12, "fit"),
    _m("qwen3-4b-base", "Qwen/Qwen3-4B-Base", "qwen3", "base", "none", "float16", 36e12, "fit"),
    _m("qwen3-8b-base", "Qwen/Qwen3-8B-Base", "qwen3", "base", "nf4", "float16", 36e12, "fit"),
    _m("qwen3-14b-base", "Qwen/Qwen3-14B-Base", "qwen3", "base", "nf4", "float16", 36e12, "fit"),
    # --- OLMo 2 base: rev="AUTO" expands into intermediate checkpoints + final (training-data axis)
    _m("olmo2-1b", "allenai/OLMo-2-0425-1B", "olmo2", "base", "none", "float16", 4e12, "fit", rev="AUTO"),
    _m("olmo2-7b", "allenai/OLMo-2-1124-7B", "olmo2", "base", "nf4", "float16", 4e12, "fit", rev="AUTO"),
    _m("olmo2-13b", "allenai/OLMo-2-1124-13B", "olmo2", "base", "nf4", "float16", 5e12, "fit", rev="AUTO"),
    # --- Held-out families (never used to fit the law)
    _m("gemma3-1b", "google/gemma-3-1b-it", "gemma3", "chat", "none", "float32", 2e12, "heldout"),
    _m("gemma3-4b", "google/gemma-3-4b-it", "gemma3", "chat", "nf4", "float32", 4e12, "heldout"),
    _m("gemma3-12b", "google/gemma-3-12b-it", "gemma3", "chat", "nf4", "float32", 12e12, "heldout"),
    _m("llama3.2-1b", "unsloth/Llama-3.2-1B-Instruct", "llama3", "chat", "none", "float16", 9e12, "heldout"),
    _m("llama3.2-3b", "unsloth/Llama-3.2-3B-Instruct", "llama3", "chat", "none", "float16", 9e12, "heldout"),
    _m("llama3.1-8b", "unsloth/Llama-3.1-8B-Instruct", "llama3", "chat", "nf4", "float16", 15e12, "heldout"),
]

PILOT_MODELS = ["qwen3-1.7b", "qwen3-1.7b-base", "olmo2-1b-main", "llama3.2-1b", "gemma3-1b"]

# Pilot go/no-go thresholds
GATE_MIN_FORMAT_COMPLIANCE = 0.90
GATE_MIN_EDIT_ELIGIBLE = 0.20
