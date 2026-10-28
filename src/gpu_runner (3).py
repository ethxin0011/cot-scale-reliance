"""Self-healing GPU runner: ONE job on t4v3 processes every model in the run list, and the
watcher lives inside the GPU job itself (no CPU cluster, no compute instance needed).

How preemption is survived without an external watcher
  * When a runner starts, it submits ONE "backup" runner job to the same cluster. Because t4v3
    has max_instances=1, the backup waits in the queue and costs nothing.
  * If the T4 is preempted, the current runner dies, and the queued backup gets the next node.
    It resumes every model from its saved part files, because gpu_generate.py skips finished
    items, and then queues its own backup.
  * When everything is complete, the runner cancels the queued backup and spawns nothing.
    The cluster scales to 0.
  * Safety: chain length capped by --max_attempts; a duplicate runner (if another runner is
    already Running) exits immediately.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ACTIVE = {"NotStarted", "Queued", "Preparing", "Provisioning", "Starting", "Running", "Finalizing"}
CHILD = None


def log(*a):
    print(time.strftime("%H:%M:%S"), "[runner]", *a, flush=True)


def is_done(out_root, key):
    return (Path(out_root) / key / "_DONE").exists()


def ml_client():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "azure-ai-ml>=1.20", "azure-identity"],
                   check=False)
    import config
    from azure.ai.ml import MLClient
    try:
        from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
        cred = AzureMLOnBehalfOfCredential()
        c = MLClient(cred, config.SUBSCRIPTION_ID, config.RESOURCE_GROUP, config.WORKSPACE)
        c.compute.get(config.GPU_CLUSTER)
        return c
    except Exception as e:
        log("OBO credential failed, trying DefaultAzureCredential:", repr(e)[:200])
        from azure.identity import DefaultAzureCredential
        c = MLClient(DefaultAzureCredential(), config.SUBSCRIPTION_ID, config.RESOURCE_GROUP, config.WORKSPACE)
        c.compute.get(config.GPU_CLUSTER)
        return c


def build_runner_job(tag, attempt, code_dir):
    import config
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.entities import UserIdentityConfiguration
    ds = f"azureml://datastores/{config.DATASTORE}/paths/{config.BLOB_PREFIX}"
    return command(
        code=str(code_dir),
        command=("python gpu_runner.py --tag " + tag + f" --attempt {attempt} "
                 "--models_root ${{inputs.models}} --prompts_root ${{inputs.prompts}} "
                 "--out_root ${{outputs.out}}"),
        environment=f"{config.GPU_ENV_NAME}@latest",
        compute=config.GPU_CLUSTER,
        experiment_name=f"{config.EXPERIMENT_NAME}-{tag}",
        display_name=f"runner-{tag}-a{attempt}",
        tags={"runner_tag": tag, "attempt": str(attempt)},
        identity=UserIdentityConfiguration(),
        inputs=dict(
            models=Input(type=AssetTypes.URI_FOLDER, path=f"{ds}/models/", mode=InputOutputModes.RO_MOUNT),
            prompts=Input(type=AssetTypes.URI_FOLDER, path=f"{ds}/{tag}/prompts/",
                          mode=InputOutputModes.DOWNLOAD)),
        outputs=dict(out=Output(type=AssetTypes.URI_FOLDER, path=f"{ds}/{tag}/results/",
                                mode=InputOutputModes.RW_MOUNT)),
        environment_variables={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                               "TOKENIZERS_PARALLELISM": "false",
                               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )


def other_runners(ml, tag, me):
    out = []
    for n, j in enumerate(ml.jobs.list()):          # newest first
        if n >= 300:
            break
        if (j.tags or {}).get("runner_tag") == tag and j.name != me and j.status in ACTIVE:
            out.append(j)
    return out


def on_term(signum, frame):
    log(f"signal {signum}: forwarding to generator and exiting")
    if CHILD and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)
        try:
            CHILD.wait(timeout=60)
        except Exception:
            pass
    sys.exit(143)


def main():
    global CHILD
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--prompts_root", required=True)
    ap.add_argument("--out_root", required=True)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_term)

    import config
    run = json.loads(Path(f"run_{a.tag}.json").read_text())
    entries = run["models"]
    todo = [e for e in entries if not is_done(a.out_root, e["key"])]
    log(f"attempt {a.attempt}: {len(entries) - len(todo)}/{len(entries)} models already done")
    me = os.environ.get("AZUREML_RUN_ID", "")

    # ---------------------------------------------------------------- watcher duties (in-job)
    ml = None
    try:
        ml = ml_client()
        others = other_runners(ml, a.tag, me)
        if any(j.status == "Running" for j in others):
            log("another runner is already Running -> exiting as duplicate"); return
        if not todo:
            for j in others:
                ml.jobs.begin_cancel(j.name)
            log("all models complete; cancelled leftover backups"); return
        if not others and a.attempt < run["max_attempts"]:
            j = ml.jobs.create_or_update(build_runner_job(a.tag, a.attempt + 1, Path.cwd()))
            log(f"backup runner queued: {j.name}")
        elif others:
            log(f"backup already queued: {[j.name for j in others]}")
        else:
            log("max_attempts reached: no further backup")
    except Exception as e:
        log("WARNING: cannot reach workspace, no backup queued (preemption would stop the chain):",
            repr(e)[:300])
    if not todo:
        log("nothing to do"); return

    # ---------------------------------------------------------------- generation, model by model
    local = Path(os.environ.get("TMPDIR", "/tmp")) / "models"
    for e in todo:
        key = e["key"]
        if is_done(a.out_root, key):
            continue
        src = Path(a.models_root) / key
        dst = local / key
        log(f"=== {key}: copying weights to local disk")
        t0 = time.time()
        shutil.rmtree(local, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("_UPLOAD_OK", ".cache"))
        log(f"copied in {time.time() - t0:.0f}s")
        out_dir = Path(a.out_root) / key
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "gpu_generate.py", "--model_dir", str(dst),
               "--prompts", str(Path(a.prompts_root) / f"{key}.parquet"),
               "--out_dir", str(out_dir), "--model_key", key,
               "--quant", e["quant"], "--dtype", e["dtype"],
               "--max_bs", str(config.MAX_BATCH), "--mem_fraction", str(config.GPU_MEM_FRACTION),
               "--ckpt_minutes", str(config.CHECKPOINT_MINUTES)]
        CHILD = subprocess.Popen(cmd)
        rc = CHILD.wait()
        CHILD = None
        log(f"=== {key}: exit code {rc} ({'complete' if is_done(a.out_root, key) else 'INCOMPLETE'})")
        shutil.rmtree(dst, ignore_errors=True)

    remaining = [e["key"] for e in entries if not is_done(a.out_root, e["key"])]
    if not remaining and ml is not None:
        for j in other_runners(ml, a.tag, me):
            if j.status != "Running":
                ml.jobs.begin_cancel(j.name)
        log("ALL MODELS COMPLETE; backups cancelled")
    elif remaining:
        log(f"remaining incomplete: {remaining} (queued backup will retry)")
        sys.exit(1)


if __name__ == "__main__":
    main()
