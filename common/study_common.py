"""
study_common.py - shared infrastructure for the LLM vulnerability-scanner comparison study (study v1).

Every scanner in the study talks to the models ONLY through the logging proxies defined here:

  * Target proxy (default port 11500)
      Sits in front of the model under test. It forces identical decoding settings on every request
      (temperature 0, seed 42, num_ctx 4096, max_tokens 1024), refuses requests for any other model, and logs
      every prompt and response to proxy_log.jsonl. Two back ends:
        - Ollama (gemma3:4b, qwen2.5:7b), Ollama native and OpenAI-compatible endpoints;
        - AWS Bedrock Converse (Llama 3 70B). Seed and num_ctx cannot be set there, so they are logged as null.

  * Brain proxies (judge on port 11600, attacker on port 11601)
      OpenAI-compatible proxies in front of the two "brain" LLMs on AWS Bedrock:
        - judge    = GPT-OSS-120B    (scoring / grading / evaluation)
        - attacker = Qwen3-Next-80B  (attack generation / simulation)
      They inject the real Bedrock API key (scanners only ever see a dummy key), reject unexpected model names,
      strip GPT-OSS's inline <reasoning>...</reasoning> text from the returned answer (the raw text is logged),
      and log every call to brain_log_judge.jsonl / brain_log_attacker.jsonl.

  * Stub target (port 11999, smoke tests only)
      Fixed-reply fake Ollama server. Used where a scanner needs a target endpoint but no real model call
      (e.g. Promptfoo's attack generation step). Never used for study results.

  * run_behind_proxies(...)
      Starts the proxies, runs one scanner command, stops the proxies and writes wrapper_result.json with the
      request / error counts read back from the logs.

The file uses only the Python standard library, so it can be imported from every scanner's own virtual
environment. The proxies run as separate processes:

    python study_common.py target-proxy --scanner garak --model gemma3:4b --port 11500 --log-dir DIR
    python study_common.py brain-proxy  --scanner garak --role judge --port 11600 --log-dir DIR
    python study_common.py stub         --port 11999 --mode refuse

The Bedrock API key is read from the environment variable BEDROCK_API_KEY, or from a line
`bedrock_api_key=...` in a .env / .env.txt file in the project root. It is never printed or logged.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# =====================================================================================================================
# 1. Study settings (frozen for study v1)
# =====================================================================================================================

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("STUDY_PROJECT_ROOT", CODE_DIR.parent))
# All scan output goes below RUNS_DIR/<scanner>/<model>/ (override with STUDY_RUNS_DIR).
RUNS_DIR = Path(os.environ.get("STUDY_RUNS_DIR", PROJECT_ROOT / "runs" / "study_v1"))
SCANNERS_DIR = Path(os.environ.get("STUDY_SCANNERS_DIR", PROJECT_ROOT / "Scanners"))

SEED = 42
NEUTRAL_PURPOSE = "A general-purpose helpful assistant."   # same target description for every scanner that asks

# --- Target models ---------------------------------------------------------------------------------------------------
OLLAMA_MODELS = ["gemma3:4b", "qwen2.5:7b"]
BEDROCK_MODELS = ["meta.llama3-70b-instruct-v1:0"]
STUDY_MODELS = ["gemma3:4b", "qwen2.5:7b", "meta.llama3-70b-instruct-v1:0"]

OLLAMA_URL = os.environ.get("STUDY_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_DECODING = {"temperature": 0.0, "seed": SEED, "num_ctx": 4096, "max_tokens": 1024}
BEDROCK_REGION = "us-east-1"
BEDROCK_RUNTIME_URL = f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com"
BEDROCK_DECODING = {"temperature": 0.0, "seed": None, "num_ctx": None, "max_tokens": 1024}
BEDROCK_RETRY = {"max_retries": 8, "base_delay_s": 2, "max_delay_s": 60, "retry_on_status": [429, 500, 502, 503, 504]}

# --- Brain LLMs (AWS Bedrock, OpenAI-compatible endpoint) -------------------------------------------------------------
BRAIN_BASE_URL = f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com/openai/v1"
JUDGE_MODEL = "openai.gpt-oss-120b-1:0"          # GPT-OSS-120B
ATTACKER_MODEL = "qwen.qwen3-next-80b-a3b"       # Qwen3-Next-80B
JUDGE_ALIASES = [JUDGE_MODEL, "gpt-oss-120b"]
JUDGE_TEMPERATURE, ATTACKER_TEMPERATURE = 0.0, 0.7
DUMMY_KEY = "dummy-brain-proxy-injects-real-key"  # what scanners are given; the proxy swaps in the real key

# --- Ports -------------------------------------------------------------------------------------------------------------
MAIN_PORTS = (11500, 11600, 11601)        # (target, judge, attacker)
PARALLEL_PORTS = (11510, 11610, 11611)    # second set, to scan Llama 3 70B (Bedrock) in parallel with a local model
STUB_PORT = 11999

# --- Smoke tests only (both unset for every study run) -----------------------------------------------------------------
# STUDY_STUB_MODE=refuse|neutral : the target proxy forwards to the stub instead of Ollama (no real target model)
# STUDY_SMOKE_LIMIT=N            : each scanner runs only N attacks / conversations per part of its plan
STUB_MODE = os.environ.get("STUDY_STUB_MODE") or None
SMOKE_LIMIT = int(os.environ.get("STUDY_SMOKE_LIMIT") or 0)

# --- Scanner installations (override with the environment variables named here) --------------------------------------
def _exe(env_name: str, default: Path) -> str:
    return os.environ.get(env_name, str(default))


GARAK_PYTHON = _exe("GARAK_PYTHON", SCANNERS_DIR / "garak/garak_env/Scripts/python.exe")
DEEPTEAM_PYTHON = _exe("DEEPTEAM_PYTHON", SCANNERS_DIR / "deepteam/deppteam_venv/Scripts/python.exe")
PYRIT_PYTHON = _exe("PYRIT_PYTHON", SCANNERS_DIR / "PyRIT/pyrit_env/Scripts/python.exe")
GISKARD_PYTHON = _exe("GISKARD_PYTHON", SCANNERS_DIR / "giskard-oss/.venv/Scripts/python.exe")
PROMPTFOO_ENTRY = _exe("PROMPTFOO_ENTRY", SCANNERS_DIR / "promptfoo/dist/src/entrypoint.js")
AUGUSTUS_BIN = _exe("AUGUSTUS_BIN", SCANNERS_DIR / "augustus/bin/augustus.exe")
CYBERSECEVAL_REPO = _exe("CYBERSECEVAL_REPO", SCANNERS_DIR / "PurpleLlama")


# =====================================================================================================================
# 2. Small helpers
# =====================================================================================================================

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def model_dir(model: str) -> str:
    """Folder name for a model: 'gemma3:4b' -> 'gemma3-4b'."""
    return model.replace(":", "-")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def port_is_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_port(port: int, timeout_s: float = 30) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not port_is_free(port):
            return True
        time.sleep(0.3)
    return False


def parse_ports(text: str | None) -> tuple[int, int, int]:
    """'11510,11610,11611' -> (11510, 11610, 11611); None -> the main port set."""
    if not text:
        return MAIN_PORTS
    target, judge, attacker = (int(x) for x in text.split(","))
    return target, judge, attacker


def archive_incomplete(folder: Path, label: str = "incomplete") -> None:
    """Keep the leftovers of an interrupted run under a new name, so they are never mixed with a fresh run."""
    if folder.exists() and any(folder.iterdir()):
        folder.rename(folder.with_name(f"{folder.name}__{label}_{int(time.time())}"))


def load_bedrock_key() -> str:
    """Bedrock API key from BEDROCK_API_KEY, or from `bedrock_api_key=...` in PROJECT_ROOT/.env(.txt). Never logged."""
    if os.environ.get("BEDROCK_API_KEY"):
        return os.environ["BEDROCK_API_KEY"]
    for name in (".env", ".env.txt"):
        path = PROJECT_ROOT / name
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if "=" in line and line.split("=", 1)[0].strip().lower() == "bedrock_api_key":
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("Bedrock API key not found: set BEDROCK_API_KEY or put bedrock_api_key=... in the project .env file")


class JsonlLog:
    """Thread-safe JSONL log file used by all proxies."""

    def __init__(self, path: Path):
        self.path, self.lock, self.count = Path(path), threading.Lock(), 0

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            self.count += 1
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class QuietHandler(BaseHTTPRequestHandler):
    """Base request handler: no console logging (everything goes to the JSONL logs) and small send helpers."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send_bytes(self, code: int, data: bytes, content_type: str = "application/json", headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, code: int, obj) -> None:
        self.send_bytes(code, json.dumps(obj).encode())

    def read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""


# =====================================================================================================================
# 3. Target proxy - Ollama back end
# =====================================================================================================================

NATIVE_PATHS = {"/api/chat", "/api/generate"}                      # Ollama native API
OPENAI_PATHS = {"/v1/chat/completions", "/v1/completions"}         # Ollama's OpenAI-compatible API
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
               "proxy-authorization", "proxy-authenticate", "content-length", "host"}
# Ollama stops a generation that loops on the same tokens with HTTP 500 "token repeat limit reached". At temperature 0
# this is deterministic, so a retry can never succeed. The proxy returns this marker text as a normal reply instead,
# so the scanner scores it and moves on; the original error is kept in the log (decision D41).
REPEAT_ABORT_MARK = "token repeat limit reached"
REPEAT_ABORT_TEXT = "GENERATION_ABORTED_REPETITION"
SAMPLING_FIELDS = ("model", "messages", "prompt", "options", "stream", "temperature", "seed", "max_tokens",
                   "max_completion_tokens", "top_p", "n")


def force_ollama_decoding(path: str, body: dict) -> dict:
    """Overwrite the scanner's decoding settings with the study settings (in place); return the original values."""
    d = OLLAMA_DECODING
    if path in NATIVE_PATHS:
        options = body.get("options") or {}
        original = {k: options.get(k) for k in ("temperature", "seed", "num_ctx", "num_predict", "top_p", "top_k")}
        options.update({"temperature": d["temperature"], "seed": d["seed"], "num_ctx": d["num_ctx"],
                        "num_predict": d["max_tokens"]})
        body["options"] = options
    else:
        original = {k: body.get(k) for k in ("temperature", "seed", "max_tokens", "max_completion_tokens", "top_p", "n")}
        body["temperature"], body["seed"], body["max_tokens"] = d["temperature"], d["seed"], d["max_tokens"]
        body.pop("max_completion_tokens", None)
        if body.get("n", 1) not in (None, 1):
            body["n"] = 1
        # num_ctx cannot be sent over the OpenAI API; the Ollama server is started with OLLAMA_CONTEXT_LENGTH=4096.
    return original


def response_text(path: str, obj: dict) -> str:
    """Reply text from one Ollama JSON object (full response or one streamed chunk)."""
    if path == "/api/chat":
        return (obj.get("message") or {}).get("content", "") or ""
    if path == "/api/generate":
        return obj.get("response", "") or ""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    if path == "/v1/chat/completions":
        return ((choices[0].get("message") or choices[0].get("delta") or {}).get("content")) or ""
    return choices[0].get("text", "") or ""


def response_usage(path: str, obj: dict) -> dict:
    if path in NATIVE_PATHS:
        return {"prompt_tokens": obj.get("prompt_eval_count"), "completion_tokens": obj.get("eval_count"),
                "done_reason": obj.get("done_reason")}
    usage = obj.get("usage") or {}
    finish = ((obj.get("choices") or [{}])[0] or {}).get("finish_reason")
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "done_reason": finish}


def parse_ollama_response(path: str, streamed: bool, data: bytes) -> dict:
    """Collect reply text and token usage from a (possibly streamed) Ollama response body, for the log."""
    text = data.decode("utf-8", errors="replace")
    out = {"response": "", "usage": {}}
    try:
        if not streamed:
            obj = json.loads(text)
            if "error" in obj:
                out["error"] = obj["error"]
            out["response"], out["usage"] = response_text(path, obj), response_usage(path, obj)
            return out
        parts, last = [], {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):                    # OpenAI-style server-sent events
                line = line[5:].strip()
                if line == "[DONE]":
                    continue
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                out["error"] = obj["error"]
            parts.append(response_text(path, obj))
            if obj.get("done") or obj.get("usage") or (obj.get("choices") or [{}])[0].get("finish_reason"):
                last = obj
        out["response"] = "".join(parts)
        out["usage"] = response_usage(path, last) if last else {}
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as e:
        out["parse_error"], out["raw_response"] = repr(e), text[:20000]
    return out


class OllamaTargetProxy(QuietHandler):
    scanner: str
    model: str
    run_id: str
    upstream: urllib.parse.ParseResult
    log: JsonlLog

    def do_GET(self):
        self.forward(None)

    def do_HEAD(self):
        self.forward(None)

    def do_DELETE(self):
        self.forward(None)

    def do_POST(self):
        self.forward(self.read_body())

    def forward(self, raw: bytes | None) -> None:
        path = urllib.parse.urlparse(self.path).path
        record = None
        if raw and (path in NATIVE_PATHS or path in OPENAI_PATHS):
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return self.send_json(400, {"error": "proxy: request body is not JSON"})
            if body.get("model") != self.model:                      # this run is locked to one model
                self.log.write({"ts": utc_now(), "run_id": self.run_id, "scanner": self.scanner, "path": path,
                                "event": "rejected_model", "requested_model": body.get("model")})
                return self.send_json(400, {"error": f"proxy: model {body.get('model')!r} rejected; "
                                                     f"this run is locked to {self.model!r}"})
            original_body = copy.deepcopy(body)
            original_params = force_ollama_decoding(path, body)
            raw = json.dumps(body).encode()
            streamed = bool(body.get("stream", path in NATIVE_PATHS))   # Ollama native streams by default
            record = {"id": str(uuid.uuid4()), "ts": utc_now(), "run_id": self.run_id, "scanner": self.scanner,
                      "model": self.model, "path": path, "stream": streamed,
                      "original_params": original_params, "forced_params": OLLAMA_DECODING,
                      "prompt": original_body.get("messages") if "chat" in path else original_body.get("prompt"),
                      "extra_request_fields": {k: v for k, v in original_body.items() if k not in SAMPLING_FIELDS}}

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        if raw is not None:
            headers["Content-Length"] = str(len(raw))
        t0 = time.perf_counter()
        conn = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port or 80, timeout=3600)
        try:
            conn.request(self.command, self.path, body=raw, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            if record:
                record.update({"status": 502, "error": repr(e), "latency_s": time.perf_counter() - t0})
                self.log.write(record)
            return self.send_json(502, {"error": f"proxy: upstream unreachable: {e!r}"})

        if record is not None and resp.status == 500:
            err_body = resp.read()
            conn.close()
            if REPEAT_ABORT_MARK in err_body.decode("utf-8", "replace"):
                return self.reply_repeat_abort(path, record, err_body, t0)
            record.update({"status": 500, "latency_s": round(time.perf_counter() - t0, 3),
                           **parse_ollama_response(path, False, err_body)})
            self.log.write(record)
            return self.send_bytes(500, err_body, resp.getheader("Content-Type", "application/json"))

        # Relay the upstream response unchanged (streamed responses chunk by chunk) and keep a copy for the log.
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_HEADERS:
                self.send_header(k, v)
        chunked = resp.getheader("Transfer-Encoding", "").lower() == "chunked" or resp.length is None
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(resp.length))
        self.end_headers()
        if self.command == "HEAD":
            conn.close()
            return
        collected = bytearray()
        try:
            if chunked:
                while chunk := (resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)):
                    collected += chunk
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            else:
                data = resp.read()
                collected += data
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError) as e:
            if record:
                record["client_disconnected"] = repr(e)
        finally:
            conn.close()
        if record is not None:
            record.update({"status": resp.status, "latency_s": round(time.perf_counter() - t0, 3),
                           **parse_ollama_response(path, record["stream"], bytes(collected))})
            self.log.write(record)

    def reply_repeat_abort(self, path: str, record: dict, err_body: bytes, t0: float) -> None:
        """Answer an Ollama 'token repeat limit' abort with a normal, clearly marked reply (decision D41)."""
        text, now, created = REPEAT_ABORT_TEXT, utc_now(), int(time.time())
        if path == "/api/chat":
            final = {"model": self.model, "created_at": now, "message": {"role": "assistant", "content": text},
                     "done": True, "done_reason": "repeat_abort", "eval_count": 0}
            chunks = [final]
        elif path == "/api/generate":
            final = {"model": self.model, "created_at": now, "response": text, "done": True,
                     "done_reason": "repeat_abort", "eval_count": 0}
            chunks = [final]
        elif path == "/v1/chat/completions":
            final = {"id": "chatcmpl-repeat-abort", "object": "chat.completion", "created": created, "model": self.model,
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            chunks = [{"id": final["id"], "object": "chat.completion.chunk", "created": created, "model": self.model,
                       "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}]
        else:   # /v1/completions
            final = {"id": "cmpl-repeat-abort", "object": "text_completion", "created": created, "model": self.model,
                     "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            chunks = [final]
        if not record["stream"]:
            data, content_type = json.dumps(final).encode(), "application/json"
        elif path in NATIVE_PATHS:
            data, content_type = ("\n".join(json.dumps(c) for c in chunks) + "\n").encode(), "application/x-ndjson"
        else:
            data = ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
            content_type = "text/event-stream"
        record.update({"status": 200, "latency_s": round(time.perf_counter() - t0, 3), "response": text,
                       "usage": {"prompt_tokens": None, "completion_tokens": 0, "done_reason": "repeat_abort"},
                       "flags": {"ollama_repeat_abort": True},
                       "upstream_status": 500, "upstream_error": err_body.decode("utf-8", "replace")[:2000]})
        self.log.write(record)
        self.send_bytes(200, data, content_type)


# =====================================================================================================================
# 4. Target proxy - AWS Bedrock back end (Llama 3 70B)
# =====================================================================================================================

BEDROCK_STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


def to_converse_messages(messages):
    """Chat messages -> (system blocks, Converse messages, notes). Converse needs alternating roles starting with
    'user', so consecutive same-role turns are merged; every such change is reported in `notes` (and logged)."""
    system, conv, notes = [], [], []
    for m in messages or []:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):                                 # OpenAI content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = content if isinstance(content, str) else ("" if content is None else str(content))
        if role == "system":
            if content:
                system.append({"text": content})
            continue
        role = "assistant" if role == "assistant" else "user"
        if not content.strip():
            content = " "
            notes.append("empty_content_replaced")
        if conv and conv[-1]["role"] == role:
            conv[-1]["content"][0]["text"] += "\n\n" + content
            notes.append("merged_consecutive_" + role)
        else:
            conv.append({"role": role, "content": [{"text": content}]})
    if conv and conv[0]["role"] != "user":
        conv.insert(0, {"role": "user", "content": [{"text": " "}]})
        notes.append("leading_user_inserted")
    return system, conv, notes


def call_bedrock_converse(model: str, key: str, system: list, messages: list):
    """One Converse call with retries on throttling / 5xx. Returns (status, body_or_error, retries)."""
    body = {"messages": messages, "inferenceConfig": {"temperature": BEDROCK_DECODING["temperature"],
                                                      "maxTokens": BEDROCK_DECODING["max_tokens"]}}
    if system:
        body["system"] = system
    url = f"{BEDROCK_RUNTIME_URL}/model/{urllib.parse.quote(model, safe='')}/converse"
    retries, r = [], BEDROCK_RETRY
    for attempt in range(r["max_retries"] + 1):
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        delay = min(r["max_delay_s"], r["base_delay_s"] * 2 ** attempt)
        try:
            with urllib.request.urlopen(request, timeout=600) as resp:
                return 200, json.loads(resp.read()), retries
        except urllib.error.HTTPError as e:
            message = e.read().decode("utf-8", "replace")[:2000]
            if e.code in r["retry_on_status"] and attempt < r["max_retries"]:
                retries.append({"status": e.code, "error": message[:300], "slept_s": delay})
                time.sleep(delay)
                continue
            return e.code, message, retries
        except OSError as e:
            if attempt < r["max_retries"]:
                retries.append({"status": None, "error": repr(e)[:300], "slept_s": delay})
                time.sleep(delay)
                continue
            return 502, repr(e), retries


class BedrockTargetProxy(QuietHandler):
    """Speaks the same /api/chat and /v1/chat/completions formats as Ollama, so scanners need no change."""
    scanner: str
    model: str
    run_id: str
    key: str
    log: JsonlLog

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/tags":
            return self.send_json(200, {"models": [{"name": self.model, "model": self.model}]})
        if path == "/v1/models":
            return self.send_json(200, {"object": "list", "data": [{"id": self.model, "object": "model"}]})
        if path in ("/", "/api/version"):
            return self.send_json(200, {"version": "bedrock-target-proxy"})
        self.send_json(404, {"error": f"bedrock target proxy: GET {path} not supported"})

    def do_POST(self):
        path, raw = urllib.parse.urlparse(self.path).path, self.read_body()
        if path == "/api/show":
            return self.send_json(200, {"modelfile": "", "parameters": "", "details": {"family": "bedrock"}})
        if path not in ("/api/chat", "/v1/chat/completions"):
            return self.send_json(400, {"error": f"bedrock target proxy: {path} not supported (chat endpoints only)"})
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return self.send_json(400, {"error": "proxy: request body is not JSON"})
        if body.get("model") != self.model:
            self.log.write({"ts": utc_now(), "run_id": self.run_id, "scanner": self.scanner, "path": path,
                            "event": "rejected_model", "requested_model": body.get("model")})
            return self.send_json(400, {"error": f"proxy: model {body.get('model')!r} rejected; locked to {self.model!r}"})

        native = path == "/api/chat"
        if native:
            options = body.get("options") or {}
            original = {k: options.get(k) for k in ("temperature", "seed", "num_ctx", "num_predict", "top_p", "top_k")}
        else:
            original = {k: body.get(k) for k in ("temperature", "seed", "max_tokens", "max_completion_tokens", "top_p", "n")}
        streamed = bool(body.get("stream", native))
        system, conv, notes = to_converse_messages(body.get("messages"))
        record = {"id": str(uuid.uuid4()), "ts": utc_now(), "run_id": self.run_id, "scanner": self.scanner,
                  "model": self.model, "path": path, "stream": streamed, "backend": "bedrock-converse",
                  "original_params": original, "forced_params": BEDROCK_DECODING, "prompt": body.get("messages"),
                  "extra_request_fields": {k: v for k, v in body.items() if k not in SAMPLING_FIELDS}}
        if notes:
            record["message_transform"] = notes

        t0 = time.perf_counter()
        status, result, retries = call_bedrock_converse(self.model, self.key, system, conv)
        record["latency_s"] = round(time.perf_counter() - t0, 3)
        if retries:
            record["retries"] = retries
        if status != 200:
            record.update({"status": status, "error": result, "response": "", "usage": {}})
            self.log.write(record)
            return self.send_json(status, {"error": f"bedrock: {result}"})

        blocks = ((result.get("output") or {}).get("message") or {}).get("content") or []
        text = "".join(b.get("text", "") for b in blocks)
        stop = result.get("stopReason")
        u = result.get("usage") or {}
        usage = {"prompt_tokens": u.get("inputTokens"), "completion_tokens": u.get("outputTokens"),
                 "done_reason": BEDROCK_STOP_REASONS.get(stop, stop)}
        record.update({"status": 200, "response": text, "usage": usage, "bedrock_stop_reason": stop,
                       "bedrock_latency_ms": (result.get("metrics") or {}).get("latencyMs")})
        if stop not in BEDROCK_STOP_REASONS:
            record["flags"] = {"bedrock_unusual_stop": stop}
        self.log.write(record)
        self.reply_in_client_format(native, streamed, record, text, usage)

    def reply_in_client_format(self, native, streamed, record, text, usage):
        done = usage["done_reason"]
        if native:                                                  # Ollama /api/chat format
            final = {"model": self.model, "created_at": utc_now(), "message": {"role": "assistant", "content": text},
                     "done": True, "done_reason": done, "prompt_eval_count": usage["prompt_tokens"],
                     "eval_count": usage["completion_tokens"], "total_duration": int(record["latency_s"] * 1e9)}
            if streamed:
                return self.send_bytes(200, (json.dumps(final) + "\n").encode(), "application/x-ndjson")
            return self.send_json(200, final)
        chat_id, created = "chatcmpl-" + record["id"], int(time.time())   # OpenAI chat-completions format
        oai_usage = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                     "total_tokens": (usage["prompt_tokens"] or 0) + (usage["completion_tokens"] or 0)}
        if streamed:
            events = [{"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": self.model,
                       "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                      {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": self.model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": done}], "usage": oai_usage}]
            payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return self.send_bytes(200, payload.encode(), "text/event-stream")
        self.send_json(200, {"id": chat_id, "object": "chat.completion", "created": created, "model": self.model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": done}], "usage": oai_usage})


# =====================================================================================================================
# 5. Brain proxy - judge (GPT-OSS-120B) and attacker (Qwen3-Next-80B) on AWS Bedrock
# =====================================================================================================================

REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
LEADING_BRACE_RE = re.compile(r"^\{\s*(?=\{)")
BRAIN_HOP_HEADERS = HOP_HEADERS | {"authorization", "x-api-key", "api-key", "accept-encoding"}
BRAIN_LOGGED_PARAMS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "n", "response_format",
                       "tools", "tool_choice", "reasoning_effort")


def split_reasoning(content):
    """GPT-OSS on Bedrock returns '<reasoning>...</reasoning>answer'. Return (answer, reasoning, flags) (decision D27)."""
    if not isinstance(content, str):
        return content, None, {}
    parts = REASONING_RE.findall(content)
    clean, flags = REASONING_RE.sub("", content), {}
    lower = clean.lower()
    if "<reasoning>" in lower:                          # opened but never closed (cut off by max_tokens)
        i = lower.index("<reasoning>")
        parts.append(clean[i + len("<reasoning>"):])
        clean = clean[:i]
        flags["reasoning_unclosed"] = True
    reasoning = "\n".join(p.strip() for p in parts) if parts else None
    return clean.strip(), reasoning, flags


def repair_duplicated_brace(text):
    """With response_format set, Bedrock sometimes returns GPT-OSS JSON starting with a duplicated '{' (decision D31).
    Repair ONLY if the text is invalid JSON and removing exactly one leading '{' makes it valid. -> (text, repaired)."""
    if not isinstance(text, str):
        return text, False
    try:
        json.loads(text)
        return text, False
    except json.JSONDecodeError:
        pass
    fixed = LEADING_BRACE_RE.sub("", text, count=1)
    if fixed == text:
        return text, False
    try:
        json.loads(fixed)
        return fixed, True
    except json.JSONDecodeError:
        return text, False


class BrainProxy(QuietHandler):
    scanner: str
    run_id: str
    role: str                 # "judge" or "attacker"
    upstream_model: str
    accepted_names: list
    key: str
    upstream_host: str
    upstream_prefix: str      # e.g. /openai/v1
    log: JsonlLog

    def upstream_path(self) -> str:
        p = urllib.parse.urlparse(self.path)
        sub = p.path[len("/v1"):] if p.path.startswith("/v1") else p.path
        return self.upstream_prefix + sub + (("?" + p.query) if p.query else "")

    def call_upstream(self, method: str, body: bytes | None):
        headers = {k: v for k, v in self.headers.items() if k.lower() not in BRAIN_HOP_HEADERS}
        headers.update({"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPSConnection(self.upstream_host, timeout=600)
        try:
            conn.request(method, self.upstream_path(), body=body, headers=headers)
            r = conn.getresponse()
            return r.status, r.getheaders(), r.read()
        finally:
            conn.close()

    def send_error_json(self, code: int, message: str):
        self.send_json(code, {"error": {"message": f"brain_proxy: {message}", "type": "proxy_error"}})

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path.rstrip("/") == "/v1/models":
            return self.send_json(200, {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "bedrock"}
                                                                   for n in self.accepted_names]})
        try:
            status, headers, data = self.call_upstream("GET", None)
        except OSError as e:
            return self.send_error_json(502, f"upstream unreachable: {e!r}")
        self.send_bytes(status, data, dict(headers).get("Content-Type", "application/json"))

    def do_POST(self):
        path, raw = urllib.parse.urlparse(self.path).path, self.read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self.send_error_json(400, "request body is not JSON")
        if body.get("model") not in self.accepted_names:
            self.log.write({"ts": utc_now(), "run_id": self.run_id, "scanner": self.scanner, "path": path,
                            "event": "rejected_model", "requested_model": body.get("model")})
            return self.send_error_json(400, f"model {body.get('model')!r} rejected; "
                                             f"this proxy forwards to {self.upstream_model!r}")

        original = copy.deepcopy(body)
        wants_stream = bool(body.get("stream"))
        body["model"] = self.upstream_model
        body.pop("stream", None)                  # Bedrock is always called non-streaming (see send_as_sse)
        body.pop("stream_options", None)
        record = {"id": str(uuid.uuid4()), "ts": utc_now(), "run_id": self.run_id, "scanner": self.scanner,
                  "path": path, "role": self.role, "system_prompt_added": False,
                  "requested_model": original.get("model"), "upstream_model": self.upstream_model,
                  "client_stream": wants_stream, "params": {k: original.get(k) for k in BRAIN_LOGGED_PARAMS},
                  "messages": original.get("messages"), "prompt": original.get("prompt")}
        t0 = time.perf_counter()
        try:
            status, _, data = self.call_upstream("POST", json.dumps(body).encode())
        except OSError as e:
            record.update({"status": 502, "error": repr(e), "latency_s": round(time.perf_counter() - t0, 3)})
            self.log.write(record)
            return self.send_error_json(502, f"upstream unreachable: {e!r}")
        record["status"], record["latency_s"] = status, round(time.perf_counter() - t0, 3)

        text = data.decode("utf-8", "replace")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            record["raw_response_text"] = text[:20000]
            self.log.write(record)
            return self.send_bytes(status, data)
        record["raw_response"] = copy.deepcopy(obj)
        if status == 200 and isinstance(obj.get("choices"), list):
            self.clean_choices(obj, original, record)
        else:
            record["error"] = obj.get("error") or obj.get("message") or obj
        record["usage"] = obj.get("usage")
        self.log.write(record)
        if wants_stream and status == 200:
            return self.send_as_sse(obj)
        self.send_bytes(status, json.dumps(obj).encode())

    @staticmethod
    def clean_choices(obj: dict, original: dict, record: dict) -> None:
        """Remove inline reasoning from every choice (in place) and log answer + reasoning separately."""
        answers, reasonings = [], []
        for choice in obj["choices"]:
            msg = choice.get("message") or {}
            clean, reasoning, flags = split_reasoning(msg.get("content"))
            for k in ("reasoning", "reasoning_content"):          # in case Bedrock returns a separate field
                if k in msg:
                    reasoning = (reasoning + "\n" if reasoning else "") + str(msg.pop(k))
            if original.get("response_format") and isinstance(clean, str):
                before = clean
                clean, repaired = repair_duplicated_brace(clean)
                if repaired:
                    flags["brace_repaired"] = True
                    record["answer_before_brace_repair"] = before
            if "content" in msg:
                msg["content"] = clean
            if "text" in choice:                                   # legacy completions endpoint
                choice["text"], r2, f2 = split_reasoning(choice["text"])
                reasoning, flags, clean = reasoning or r2, {**flags, **f2}, choice["text"]
            answers.append(clean)
            reasonings.append(reasoning)
            if flags:
                record.setdefault("flags", {}).update(flags)
        record["answer"] = answers[0] if len(answers) == 1 else answers
        record["reasoning"] = reasonings[0] if len(reasonings) == 1 else reasonings
        record["finish_reason"] = obj["choices"][0].get("finish_reason") if obj["choices"] else None

    def send_as_sse(self, obj: dict) -> None:
        """Return a non-streamed answer as a valid server-sent-event stream (one content chunk + finish + [DONE])."""
        chat_id, created, model = obj.get("id", "chatcmpl-proxy"), obj.get("created", int(time.time())), obj.get("model")
        events = []
        for i, choice in enumerate(obj.get("choices") or []):
            msg = choice.get("message") or {}
            delta = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                delta["tool_calls"] = [{**tc, "index": j} for j, tc in enumerate(msg["tool_calls"])]
            events.append({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [{"index": i, "delta": delta, "finish_reason": None}]})
            events.append({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [{"index": i, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]})
        if obj.get("usage"):
            events.append({"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [], "usage": obj["usage"]})
        payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        self.send_bytes(200, payload.encode(), "text/event-stream", {"Cache-Control": "no-cache"})


# =====================================================================================================================
# 6. Stub target (smoke tests only - never used for study results)
# =====================================================================================================================

STUB_REPLIES = {"refuse": "I'm sorry, but I can't help with that.",
                "neutral": "Thanks for the question. Could you tell me a bit more about what you are looking for?"}


class StubTarget(QuietHandler):
    reply: str

    def do_GET(self):
        self.send_json(200, {"models": []})

    def do_POST(self):
        body = json.loads(self.read_body() or b"{}")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        if self.path in ("/v1/chat/completions", "/v1/completions"):
            usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            if body.get("stream"):
                events = [{"id": "stub", "object": "chat.completion.chunk", "created": 0, "model": body.get("model"),
                           "choices": [{"index": 0, "delta": {"role": "assistant", "content": self.reply},
                                        "finish_reason": None}]},
                          {"id": "stub", "object": "chat.completion.chunk", "created": 0, "model": body.get("model"),
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}]
                payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
                return self.send_bytes(200, payload.encode(), "text/event-stream")
            return self.send_json(200, {"id": "stub", "object": "chat.completion", "created": 0, "model": body.get("model"),
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": self.reply},
                                                     "finish_reason": "stop"}], "usage": usage})
        if self.path == "/api/generate":
            return self.send_json(200, {"model": body.get("model"), "created_at": now, "response": self.reply,
                                        "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 10})
        if self.path != "/api/chat":
            return self.send_json(404, {"error": f"stub: {self.path} not supported"})
        self.send_json(200, {"model": body.get("model"), "created_at": now,
                             "message": {"role": "assistant", "content": self.reply}, "done": True,
                             "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 10})


# =====================================================================================================================
# 7. Starting the proxies and running a scanner behind them
# =====================================================================================================================

QUIET = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ProxySet:
    """Context manager that starts the study proxies as background processes and stops them on exit.

        with ProxySet("deepteam", log_dir, model="gemma3:4b", ports=PARALLEL_PORTS):
            ...   # target on ports[0], judge on ports[1], attacker on ports[2]

    model=None    -> no target proxy (e.g. attack generation only)
    brains=False  -> no judge/attacker proxies (scanners that use no LLM besides the target)
    stub_mode     -> 'refuse' / 'neutral': the target proxy forwards to the stub instead of Ollama (smoke tests)
    """

    def __init__(self, scanner: str, log_dir: Path, model: str | None = None, ports=MAIN_PORTS,
                 brains: bool = True, stub_mode: str | None = None):
        self.scanner, self.log_dir, self.model = scanner, Path(log_dir), model
        self.ports, self.brains, self.stub_mode = tuple(ports), brains, stub_mode or STUB_MODE
        self.run_id = f"{scanner}_{self.log_dir.name}"
        self.processes: list[subprocess.Popen] = []

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        target_port, judge_port, attacker_port = self.ports
        wanted = ([target_port] if self.model else []) + ([judge_port, attacker_port] if self.brains else [])
        wanted += [STUB_PORT] if self.stub_mode else []
        for port in wanted:
            if not port_is_free(port):
                raise SystemExit(f"port {port} is busy (another scan running?)")
        me = [sys.executable, str(Path(__file__).resolve())]
        common = ["--scanner", self.scanner, "--run-id", self.run_id, "--log-dir", str(self.log_dir)]
        if self.stub_mode:
            self.processes.append(subprocess.Popen(me + ["stub", "--port", str(STUB_PORT), "--mode", self.stub_mode], **QUIET))
        if self.model:
            cmd = me + ["target-proxy", "--model", self.model, "--port", str(target_port)] + common
            if self.stub_mode and self.model not in BEDROCK_MODELS:
                cmd += ["--upstream", f"http://127.0.0.1:{STUB_PORT}"]
            self.processes.append(subprocess.Popen(cmd, **QUIET))
        if self.brains:
            self.processes.append(subprocess.Popen(me + ["brain-proxy", "--role", "judge", "--port", str(judge_port)]
                                                   + common, **QUIET))
            self.processes.append(subprocess.Popen(me + ["brain-proxy", "--role", "attacker", "--port", str(attacker_port)]
                                                   + common, **QUIET))
        for port in wanted:
            if not wait_for_port(port):
                self.__exit__()
                raise SystemExit(f"proxy on port {port} did not start")
        return self

    def __exit__(self, *exc):
        for p in self.processes:
            p.terminate()
        for p in self.processes:
            p.wait(15)


def log_counts(log_dir: Path) -> dict:
    """Request / error counts read back from the three proxy logs in a run folder."""
    d = Path(log_dir)
    target = [r for r in read_jsonl(d / "proxy_log.jsonl") if "status" in r]
    attacker = [r for r in read_jsonl(d / "brain_log_attacker.jsonl") if "status" in r or "response" in r]
    judge = [r for r in read_jsonl(d / "brain_log_judge.jsonl") if "status" in r or "response" in r]
    rejected = sum(r.get("event") == "rejected_model" for name in ("proxy_log.jsonl", "brain_log_attacker.jsonl",
                                                                   "brain_log_judge.jsonl") for r in read_jsonl(d / name))
    return {"target_requests": len(target), "target_errors": sum(r.get("status") != 200 for r in target),
            "attacker_calls": len(attacker), "attacker_errors": sum(r.get("status", 200) != 200 for r in attacker),
            "judge_calls": len(judge), "judge_errors": sum(r.get("status", 200) != 200 for r in judge),
            "rejected_model_events": rejected}


def scanner_environment(ports=MAIN_PORTS, update: dict | None = None, remove: tuple = ()) -> dict:
    """Environment for a scanner process: telemetry off, offline Hugging Face, target URL = target proxy, and no
    OpenAI credentials (so no scanner can ever reach api.openai.com by accident)."""
    target = f"http://127.0.0.1:{ports[0]}"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "DO_NOT_TRACK": "1",
           "GISKARD_TELEMETRY_DISABLED": "1", "DEEPEVAL_TELEMETRY_OPT_OUT": "YES", "DEEPTEAM_TELEMETRY_OPT_OUT": "YES",
           "PROMPTFOO_DISABLE_TELEMETRY": "1", "PROMPTFOO_DISABLE_UPDATE": "1",
           "PROMPTFOO_DISABLE_REMOTE_GENERATION": "true", "PROMPTFOO_DISABLE_SHARING": "1",
           "OLLAMA_HOST": target, "OLLAMA_BASE_URL": target, "STUDY_PROXY_URL": target, "HF_HUB_OFFLINE": "1"}
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL") + tuple(remove):
        env.pop(name, None)
    env.update(update or {})
    return env


def run_behind_proxies(scanner: str, model: str, log_dir: Path, command: list[str], ports=MAIN_PORTS,
                       brains: bool = True, stub_mode: str | None = None, env_update: dict | None = None,
                       env_remove: tuple = (), cwd: Path | None = None, timeout_s: int = 3 * 3600) -> dict:
    """Start the proxies, run `command`, stop the proxies, write scanner_stdout/stderr.txt and wrapper_result.json
    into log_dir and return the wrapper result (exit code, duration and the counts from log_counts)."""
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    started, t0, exit_code = utc_now(), time.time(), None
    with ProxySet(scanner, log_dir, model=model, ports=ports, brains=brains, stub_mode=stub_mode) as proxies:
        stub_mode = proxies.stub_mode
        done = subprocess.run(command, cwd=str(cwd or log_dir), env=scanner_environment(ports, env_update, env_remove),
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        exit_code = done.returncode
        (log_dir / "scanner_stdout.txt").write_text(done.stdout, encoding="utf-8")
        (log_dir / "scanner_stderr.txt").write_text(done.stderr, encoding="utf-8")
    result = {"scanner": scanner, "model": model, "stub_target": bool(stub_mode), "stub_mode": stub_mode,
              "ports": ",".join(map(str, ports)), "command": command, "started_utc": started,
              "duration_s": round(time.time() - t0, 1), "exit_code": exit_code, **log_counts(log_dir)}
    write_json(log_dir / "wrapper_result.json", result)
    return result


# =====================================================================================================================
# 8. Command line (used by ProxySet to start each proxy as its own process)
# =====================================================================================================================

def serve(handler_class, host: str, port: int, banner: str) -> None:
    server = ThreadingHTTPServer((host, port), handler_class)
    print(banner, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Study proxies (normally started by ProxySet / run_behind_proxies).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("target-proxy", "brain-proxy"):
        p = sub.add_parser(name)
        p.add_argument("--scanner", required=True)
        p.add_argument("--run-id", required=True)
        p.add_argument("--port", type=int, required=True)
        p.add_argument("--log-dir", required=True)
        p.add_argument("--host", default="127.0.0.1")
        if name == "target-proxy":
            p.add_argument("--model", required=True)
            p.add_argument("--upstream", default=None, help="smoke tests only: forward to the stub instead of Ollama")
        else:
            p.add_argument("--role", required=True, choices=["judge", "attacker"])
    p = sub.add_parser("stub")
    p.add_argument("--port", type=int, default=STUB_PORT)
    p.add_argument("--mode", default="refuse", choices=sorted(STUB_REPLIES))
    a = parser.parse_args()

    if a.command == "stub":
        StubTarget.reply = STUB_REPLIES[a.mode]
        return serve(StubTarget, "127.0.0.1", a.port, f"[stub] 127.0.0.1:{a.port} mode={a.mode}")

    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    session = {"started_utc": utc_now(), "scanner": a.scanner, "run_id": a.run_id, "listen": f"{a.host}:{a.port}"}

    if a.command == "target-proxy":
        if a.model in BEDROCK_MODELS:
            handler = BedrockTargetProxy
            handler.key = load_bedrock_key()
            session.update({"model": a.model, "target_backend": "bedrock-converse", "region": BEDROCK_REGION,
                            "forced": BEDROCK_DECODING})
        elif a.model in OLLAMA_MODELS:
            handler = OllamaTargetProxy
            handler.upstream = urllib.parse.urlparse(a.upstream or OLLAMA_URL)
            session.update({"model": a.model, "upstream": a.upstream or OLLAMA_URL, "forced": OLLAMA_DECODING})
        else:
            sys.exit(f"--model {a.model!r} is not a study model")
        handler.scanner, handler.model, handler.run_id = a.scanner, a.model, a.run_id
        handler.log = JsonlLog(log_dir / "proxy_log.jsonl")
        write_json(log_dir / "proxy_session.json", session)
        return serve(handler, a.host, a.port, f"[target-proxy] {a.host}:{a.port} model={a.model} log={handler.log.path}")

    # brain-proxy
    BrainProxy.scanner, BrainProxy.run_id, BrainProxy.role = a.scanner, a.run_id, a.role
    BrainProxy.upstream_model = JUDGE_MODEL if a.role == "judge" else ATTACKER_MODEL
    BrainProxy.accepted_names = JUDGE_ALIASES + ([ATTACKER_MODEL] if a.role == "attacker" else [])
    BrainProxy.key = load_bedrock_key()
    upstream = urllib.parse.urlparse(BRAIN_BASE_URL)
    BrainProxy.upstream_host, BrainProxy.upstream_prefix = upstream.hostname, upstream.path.rstrip("/")
    BrainProxy.log = JsonlLog(log_dir / f"brain_log_{a.role}.jsonl")
    write_json(log_dir / f"brain_session_{a.role}.json",
               {**session, "role": a.role, "upstream": BRAIN_BASE_URL, "model": BrainProxy.upstream_model,
                "accepted_model_names": BrainProxy.accepted_names,
                "reasoning_handling": "inline <reasoning> removed from returned content; full raw content logged"})
    serve(BrainProxy, a.host, a.port, f"[brain-proxy] {a.host}:{a.port}/v1 role={a.role} model={BrainProxy.upstream_model}")


if __name__ == "__main__":
    main()
