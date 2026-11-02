"""Blob helpers for the workspace datastore (used on the CPU compute instance).

Uses the datastore's own credential (account key / SAS) when available; otherwise
falls back to Azure AD (DefaultAzureCredential -> needs 'Storage Blob Data Contributor').
"""
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from config import DATASTORE, RESOURCE_GROUP, SUBSCRIPTION_ID, WORKSPACE


@lru_cache(maxsize=1)
def ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    return MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE)


@lru_cache(maxsize=1)
def container():
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient
    try:
        ds = ml_client().datastores.get(DATASTORE, include_secrets=True)
    except TypeError:
        ds = ml_client().datastores.get(DATASTORE)
    endpoint = getattr(ds, "endpoint", None) or "core.windows.net"
    url = f"https://{ds.account_name}.blob.{endpoint}"
    cred = None
    c = getattr(ds, "credentials", None)
    if c is not None:
        cred = getattr(c, "account_key", None) or getattr(c, "sas_token", None)
    return ContainerClient(url, ds.container_name, credential=cred or DefaultAzureCredential())


def exists(blob):
    return container().get_blob_client(blob).exists()


def upload_bytes(data, blob):
    container().upload_blob(blob, data, overwrite=True)


def upload_file(local, blob):
    with open(local, "rb") as f:
        container().upload_blob(blob, f, overwrite=True, max_concurrency=8)


def upload_dir(local_dir, blob_prefix, workers=16):
    local_dir = Path(local_dir)
    files = [p for p in local_dir.rglob("*") if p.is_file() and ".cache" not in p.parts]

    def _up(p):
        upload_file(p, f"{blob_prefix}/{p.relative_to(local_dir).as_posix()}")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_up, files))
    return len(files)


def list_blobs(prefix):
    return [b.name for b in container().list_blobs(name_starts_with=prefix)]


def download_prefix(prefix, local_dir, suffix=None, workers=16):
    local_dir = Path(local_dir)
    names = [n for n in list_blobs(prefix) if suffix is None or n.endswith(suffix)]

    def _down(n):
        dst = local_dir / n[len(prefix):].lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            f.write(container().download_blob(n).readall())
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_down, names))
    return len(names)


def read_text(blob):
    return container().download_blob(blob).readall().decode()
