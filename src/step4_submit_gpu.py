"""STEP 4 (run on the CPU compute instance): submit + babysit one GPU job per model on t4v3.

The watcher loop:
  * submits jobs (smallest model first), at most MAX_CONCURRENT_GPU_JOBS at a time;
  * polls every WATCH_POLL_SECONDS; when a job ends Failed/Canceled (e.g. T4 preemption)
    or Completed without the blob `_DONE` marker, it resubmits it. The new attempt mounts
    the same output folder and resumes from the saved part files;
  * persists its own state in data/<tag>/job_state.json, so the cell (or a nohup process)
    can be stopped and restarted at any time without duplicate submissions.

Usage:
  python step4_submit_gpu.py --tag pilot --models pilot            # submit + watch until done
  python step4_submit_gpu.py --tag main  --models all --once       # one poll/submit round
"""
import argparse
import json
import time

from config import (DATA_ROOT, EXPERIMENT_NAME, ENV_DIR, GPU_BASE_IMAGE, GPU_CLUSTER, GPU_ENV_NAME,
                    MAX_BATCH, MAX_CONCURRENT_GPU_JOBS, MAX_JOB_ATTEMPTS, PILOT_MODELS,
                    CHECKPOINT_MINUTES, GPU_MEM_FRACTION, SRC_DIR, WATCH_POLL_SECONDS,
                    DATASTORE, blob_models, blob_prompts, blob_results, data_dir)

ACTIVE = {"NotStarted", "Queued", "Preparing", "Provisioning", "Starting", "Running",
          "Finalizing", "CancelRequested"}


def az_path(blob, folder=True):
    return f"azureml://datastores/{DATASTORE}/paths/{blob}" + ("/" if folder else "")


def get_env(ml):
    from azure.ai.ml.entities import Environment
    env = Environment(name=GPU_ENV_NAME, image=GPU_BASE_IMAGE,
                      conda_file=str(ENV_DIR / "conda_gpu.yml"),
                      description="torch 2.5.1 cu121 + transformers + bitsandbytes for T4")
    return ml.environments.create_or_update(env)


def build_job(entry, tag, env):
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    key = entry["key"]
    cmd = ("python gpu_generate.py --model_dir ${{inputs.model}} --prompts ${{inputs.prompts}} "
           "--out_dir ${{outputs.out}} "
           f"--model_key {key} --quant {entry['quant']} --dtype {entry['dtype']} "
           f"--max_bs {MAX_BATCH} --mem_fraction {GPU_MEM_FRACTION} "
           f"--ckpt_minutes {CHECKPOINT_MINUTES}")
    return command(
        code=str(SRC_DIR),
        command=cmd,
        environment=env,
        compute=GPU_CLUSTER,
        experiment_name=f"{EXPERIMENT_NAME}-{tag}",
        display_name=f"gen-{tag}-{key}",
        inputs=dict(
            model=Input(type=AssetTypes.URI_FOLDER, path=az_path(blob_models(key)),
                        mode=InputOutputModes.DOWNLOAD),
            prompts=Input(type=AssetTypes.URI_FILE, path=az_path(blob_prompts(tag, key), False),
                          mode=InputOutputModes.DOWNLOAD)),
        outputs=dict(out=Output(type=AssetTypes.URI_FOLDER, path=az_path(blob_results(tag, key)),
                                mode=InputOutputModes.RW_MOUNT)),
        environment_variables={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1"},
    )


def main(tag, which, once):
    import storage
    ml = storage.ml_client()
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    keys = list(manifest)
    if which == "pilot":
        keys = [k for k in keys if k in PILOT_MODELS]
    elif which != "all":
        keys = which.split(",")
    keys.sort(key=lambda k: (manifest[k]["N"] * (0.5 if manifest[k]["quant"] == "nf4" else 1), k))
    state_path = data_dir(tag) / "job_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    env = None

    def save():
        state_path.write_text(json.dumps(state, indent=1))

    while True:
        active, n_done, n_dead = 0, 0, 0
        for k in keys:
            s = state.setdefault(k, dict(attempts=0, job=None, status="pending"))
            if s["status"] == "done":
                n_done += 1; continue
            if storage.exists(f"{blob_results(tag, k)}/_DONE"):
                s["status"] = "done"; n_done += 1; save(); continue
            if s["job"]:
                try:
                    js = ml.jobs.get(s["job"]).status
                except Exception as e:
                    js = "Unknown"; print(f"[warn] {k}: {e!r}")
                if js in ACTIVE or js == "Unknown":
                    active += 1; s["status"] = js; continue
                s["status"] = js            # Completed w/o _DONE, Failed, Canceled
            if s["attempts"] >= MAX_JOB_ATTEMPTS:
                s["status"] = "gave_up"; n_dead += 1; continue
        # submit within concurrency limit
        for k in keys:
            s = state[k]
            if active >= MAX_CONCURRENT_GPU_JOBS:
                break
            if s["status"] in ("done", "gave_up") or s["status"] in ACTIVE or s["status"] == "Unknown":
                continue
            if env is None:
                env = get_env(ml)
            job = ml.jobs.create_or_update(build_job(manifest[k], tag, env))
            s.update(job=job.name, attempts=s["attempts"] + 1, status="Queued",
                     submitted=time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"[submit] {k} attempt {s['attempts']} -> {job.name}  {job.studio_url}")
            active += 1
        save()
        summary = {k: f"{state[k]['status']}({state[k]['attempts']})" for k in keys}
        print(time.strftime("%H:%M"), f"done {n_done}/{len(keys)}  gave_up {n_dead}  active {active}")
        print("   ", summary)
        if once or n_done + n_dead == len(keys):
            break
        time.sleep(WATCH_POLL_SECONDS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", choices=["pilot", "main"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    main(a.tag, a.models, a.once)
