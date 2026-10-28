"""GPT-6.1-Sol (Azure AI Foundry) helpers, CPU side only.

Used ONLY where it improves measurement accuracy, never to solve the task:
  (a) parse fallback: when the regex parser finds no valid 'Answer', GPT extracts the
      final 5-letter state the model WROTE (it is told not to compute anything);
  (b) parser audit: on a random sample GPT extracts answer + step-k state independently,
      and we report regex-vs-GPT disagreement (a measurement-validity number for the paper).
Key is read from env AZURE_OPENAI_API_KEY. Responses are cached on disk (jsonl).
"""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from config import AOAI_API_VERSION, AOAI_DEPLOYMENT, AOAI_ENDPOINT, AOAI_MAX_WORKERS, DATA_ROOT

SYSTEM = ("You are a strict text extractor. You never solve or correct problems. "
          "You only copy what is literally written in the given model output.")
TEMPLATE = (
    "Below is the raw output of a model asked to track letters in 5 boxes.\n"
    "Extract, exactly as written by the model (do NOT compute or fix anything):\n"
    "1) final_answer: the 5 letters the model gives as its final answer, in box order 1-5;\n"
    "2) step_{k}: the 5 letters written for 'Step {k}', if present.\n"
    "Use null if not present or ambiguous. Return JSON: "
    '{{"final_answer": "ABCDE" or null, "step_{k}": "ABCDE" or null}}\n\n'
    "MODEL OUTPUT:\n<<<\n{text}\n>>>")

_lock = threading.Lock()
_CACHE_PATH = DATA_ROOT / "gpt_cache.jsonl"
_cache = None


def _client():
    from openai import AzureOpenAI
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set AZURE_OPENAI_API_KEY in the environment")
    return AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=key, api_version=AOAI_API_VERSION)


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if _CACHE_PATH.exists():
            for line in open(_CACHE_PATH):
                try:
                    r = json.loads(line); _cache[r["h"]] = r["v"]
                except Exception:
                    pass
    return _cache


def _norm(s):
    if not s:
        return None
    s = "".join(c for c in str(s).upper() if c in "ABCDE")
    return s if len(s) == 5 and len(set(s)) == 5 else None


def extract(text, k, client=None, retries=6):
    prompt = TEMPLATE.format(text=(text or "")[-6000:], k=k)
    h = hashlib.sha256((AOAI_DEPLOYMENT + prompt).encode()).hexdigest()
    cache = _load_cache()
    if h in cache:
        return cache[h]
    client = client or _client()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    out, use_json = None, True
    for att in range(retries):
        try:
            kw = dict(model=AOAI_DEPLOYMENT, messages=msgs, max_completion_tokens=400)
            if use_json:
                kw["response_format"] = {"type": "json_object"}
            r = client.chat.completions.create(**kw)
            raw = r.choices[0].message.content or "{}"
            raw = raw[raw.find("{"): raw.rfind("}") + 1] or "{}"
            d = json.loads(raw)
            out = dict(final_answer=_norm(d.get("final_answer")), step_k=_norm(d.get(f"step_{k}")))
            break
        except Exception as e:
            msg = repr(e)
            if "response_format" in msg:
                use_json = False
            time.sleep(min(60, 2 ** att + (5 if "429" in msg else 0)))
    if out is None:
        out = dict(final_answer=None, step_k=None, error=True)
    else:
        with _lock:
            cache[h] = out
            with open(_CACHE_PATH, "a") as f:
                f.write(json.dumps(dict(h=h, v=out)) + "\n")
    return out


def extract_many(texts, ks):
    client = _client()
    with ThreadPoolExecutor(AOAI_MAX_WORKERS) as ex:
        return list(ex.map(lambda tk: extract(tk[0], tk[1], client), zip(texts, ks)))


def ping():
    c = _client()
    r = c.chat.completions.create(model=AOAI_DEPLOYMENT, max_completion_tokens=20,
                                  messages=[{"role": "user", "content": "Reply with OK."}])
    return r.choices[0].message.content
