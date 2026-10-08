#!/usr/bin/env python3
"""Run a bounded, read-only administrative prompt through Codex Spark.

Administrative calls summarize evidence; they never build artifacts or choose a strategy.
The primary route is read-only ephemeral Codex Spark; infrastructure failures fall back
to the proven GPT-5.5 baseline.
"""
import argparse
import os
import subprocess
import sys


def codex_command(model: str, effort: str) -> list[str]:
    return [
        "codex", "exec", "-m", model,
        "-c", f'model_reasoning_effort="{effort}"',
        "--sandbox", "read-only", "--ephemeral", "--ignore-rules",
        "--skip-git-repo-check", "-",
    ]


# Backward-compatible name used by the unit test and any local callers.
command = codex_command


def run(prompt: str, timeout: int, effort: str):
    spark = os.environ.get("ADMIN_SPARK_MODEL",
                           os.environ.get("ADMIN_MODEL", "gpt-5.3-codex-spark"))
    fallback = os.environ.get("ADMIN_FALLBACK_MODEL", "gpt-5.5")
    attempts = [(spark, effort)]
    if fallback and fallback != spark:
        attempts.append((fallback, os.environ.get("ADMIN_FALLBACK_EFFORT", "low")))
    last = None
    last_model = spark
    for model, model_effort in attempts:
        last_model = model
        try:
            last = subprocess.run(codex_command(model, model_effort), input=prompt,
                                  capture_output=True, text=True, timeout=timeout,
                                  cwd="/tmp")
        except (OSError, subprocess.TimeoutExpired) as exc:
            last = exc
            continue
        body = (last.stdout or "").strip()
        if last.returncode == 0 and body:
            return 0, body, model
    if isinstance(last, subprocess.CompletedProcess):
        detail = (last.stderr or last.stdout or "administrative model failed").strip()[-500:]
        return last.returncode or 1, detail, last_model
    return 1, str(last or "administrative model unavailable"), last_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="prompt text (default: read stdin)")
    ap.add_argument("--effort", default=os.environ.get("ADMIN_EFFORT", "low"))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("ADMIN_TIMEOUT", 300)))
    a = ap.parse_args()
    prompt = a.prompt if a.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        sys.exit("admin_model: empty prompt")
    rc, body, _ = run(prompt, a.timeout, a.effort)
    print(body)
    return rc


if __name__ == "__main__":
    sys.exit(main())
