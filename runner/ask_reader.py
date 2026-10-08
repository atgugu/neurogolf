#!/usr/bin/env python3
"""ask_reader — cheap delegation for worker agents: token-heavy READING, never building.

Purpose: keep YOUR context window clean. Instead of pulling a 100 KB dump / long dossier /
foreign build script into your own context, hand it to the fast model with a precise
extraction question and read back ~1 KB of facts.

Usage (from anywhere):
  python3 ask_reader.py "list every charged tensor >500B and what produces it" graph_dump.txt
  python3 ask_reader.py "which failing draws share a pattern? group them" diff_out.txt more.txt
  some_command | python3 ask_reader.py "summarize the errors by root cause"

Guardrails (hard-coded): single response, NO tools, low effort, 300 s timeout, input
capped at ~20 KB (tail-biased — recent content survives). The cap is LOAD-BEARING:
over-long prompts can be silently OFFLOADED to a file the tool-less call cannot read —
the reply degrades to a useless "reading the offloaded prompt" stub. On an offload-stub
reply we halve and retry once.

RULES OF USE (you, the caller):
- Treat the answer as HINTS from a fast model — verify anything load-bearing yourself.
- NEVER delegate ONNX writing, shape math, or rule derivation (fleet history: 0/32
  from-scratch builds; hypo.py is the rule tool). Reading and extraction ONLY.
- Don't call it for things you can answer from your own context in seconds.
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

CAP = 20_000  # measured inline-safe budget (CLIs offload ≳4-5k-word prompts)
OFFLOAD_RE = re.compile(r"offload|reading the (full )?prompt", re.I)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    question = sys.argv[1]
    chunks = []
    for f in sys.argv[2:]:
        try:
            chunks.append(f"=== FILE {f} ===\n" + open(f, errors="replace").read())
        except OSError as e:
            chunks.append(f"=== FILE {f} === (unreadable: {e})")
    # read stdin ONLY when no files were given — under worker CLIs stdin is often an
    # open-but-silent pipe, and an unconditional read() would hang forever
    if len(sys.argv) < 3 and not sys.stdin.isatty():
        chunks.append("=== STDIN ===\n" + sys.stdin.read())
    full = "\n".join(chunks)

    def call(cap):
        body = full
        if len(body) > cap:
            body = body[:cap // 4] + "\n[...middle elided...]\n" + body[-3 * cap // 4:]
        prompt = (f"SINGLE-RESPONSE TASK. You have NO tools; your FIRST message IS the final "
                  f"answer. You are an extraction assistant for a build agent. Answer the "
                  f"question from the material below — concise, factual, structured; quote "
                  f"exact numbers/lines where relevant; write 'not present in the material' "
                  f"rather than guessing.\nQUESTION: {question}\n\n{body}")
        cmd = [sys.executable, os.path.join(HERE, "admin_model.py"), "--prompt", prompt,
               "--effort", os.environ.get("ADMIN_CONTEXT_EFFORT", "low"),
               "--timeout", "240"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                              stdin=subprocess.DEVNULL)

    try:
        p = call(CAP)
        out = p.stdout.strip()
        if len(out) < 200 and OFFLOAD_RE.search(out):  # CLI offloaded the prompt anyway
            p = call(CAP // 2)
            out = p.stdout.strip()
    except subprocess.TimeoutExpired:
        sys.exit("ask_reader: timeout (300s) — do the reading yourself or narrow the question")
    print(out or p.stderr.strip()[-300:])
    print("\n[ask_reader: hints from a fast model — verify anything load-bearing]")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
