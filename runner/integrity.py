#!/usr/bin/env python3
"""Integrity and policy checks shared by preflight and the wave heartbeat."""
import hashlib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import CLEAN
SHA_FILE = os.path.join(ROOT, "data", "SCORER_SHAS.txt")

PRICING_FILES = {
    "neurogolf_utils.py": os.path.join(CLEAN, "extracted", "neurogolf_utils", "neurogolf_utils.py"),
    "scoring_v2.py": os.path.join(CLEAN, "neurogolf", "scoring_v2.py"),
    "runner/ngolf.py": os.path.join(ROOT, "runner", "ngolf.py"),
    "runner/fast_verify.py": os.path.join(ROOT, "runner", "fast_verify.py"),
}

# Positive action only. A prompt may repeat the prohibition; that must not trip the lint.
_UNSAFE_LINES = (
    re.compile(r"(?i)(?<!never )(?:edit|patch|modify|change|append|add).{0,60}(?:scorer|pricer|scoring_v2)"),
    re.compile(r"(?i)(?:scorer|pricer).{0,60}(?:matcher|hook|tag|skip)"),
    re.compile(r"(?i)_ngolf_[a-z0-9_]*(?:skip|matcher|hook)"),
    re.compile(r"(?i)charge\s+(?:none|zero|free)"),
)
_NEGATION = re.compile(r"(?i)\b(?:never|do not|don't|avoid|forbid|forbidden|ban|banned|reject)\b")

# Scorer-lint precision rework (2026-07-14): wave7 lost 282/955 decider runs to this
# lint at a measured ~100% in-window FP rate — every flagged line was a compliance
# disclaimer ("no scorer/pricer/tag edits"), an abort-guard, or a STEP-1 audit of an
# already-rejected plan. The all-time corpus DOES contain real proposals (task162's
# "Patch the att09 scorer hook…" lineage), so exemption is per-mention, not per-line:
# a scorer/pricer mention is exempt only when a disclaimer token precedes it in the
# line or a compliance suffix follows it. P3 (_ngolf_*skip) and P4 (charge none) are
# never exempted this way — their negation words are part of the hack phrase itself.
# Replay over 1688 quarantined texts: 807 flagged lines -> 23, all sampled TPs kept.
_REFUTATION = re.compile(
    r"(?i)→\s*WRONG|-> ?WRONG|\bauto(?:matic)?\s+NO\b|\bwas illegal\b|policy/cost NO")
_SCORER_NOUN = re.compile(r"(?i)(?<![a-z])(?:scorer|pricer|scoring_v2)")
_GUARD_BEFORE = re.compile(
    r"(?i)\b(?:no|not|never|without|don'?t|do\s+not|does\s+not|zero|none|avoid|"
    r"forbid\w*|ban(?:ned)?|reject\w*|abort|stop|outside|illegal|violat\w*|"
    r"leave|leaving|preserv\w*|unchanged|untouched|unmodified|remov\w*|disabl\w*|"
    r"strip|abandon\w*)\b|rather than|instead of")
_GUARD_AFTER = re.compile(
    r"(?i)\b(?:untouched|unchanged|unmodified|intact)\b"
    r"|without\s+(?:modifying|editing|touching|reading)"
    r"|would violate|violates\b|rejection\b|appears?, stop")


def expected_shas():
    out = {}
    try:
        for line in open(SHA_FILE, errors="replace"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, sha = line.split()[:2]
            out[name] = sha.lower()
    except (OSError, ValueError):
        return {}
    return out


def integrity_errors():
    expected = expected_shas()
    errors = []
    if set(expected) != set(PRICING_FILES):
        errors.append(f"{SHA_FILE} must contain exactly: {', '.join(PRICING_FILES)}")
    for name, path in PRICING_FILES.items():
        if name not in expected:
            continue
        try:
            have = hashlib.sha256(open(path, "rb").read()).hexdigest()
        except OSError as exc:
            errors.append(f"{name}: unreadable ({exc})")
            continue
        if have != expected[name]:
            errors.append(f"{name}: SHA256 {have} != baseline {expected[name]}")
    return errors


def unsafe_policy_lines(text):
    """Return actionable scorer-hack lines while ignoring explicit prohibitions."""
    p1, p2, p3, p4 = _UNSAFE_LINES
    bad = []
    for line_no, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if "cosmetic" in low or "not free" in low or "exemption" in low:
            continue
        if _NEGATION.search(line):
            continue
        if _REFUTATION.search(line):
            continue
        flag = bool(p3.search(line) or p4.search(line))
        if not flag and (p1.search(line) or p2.search(line)):
            for m in _SCORER_NOUN.finditer(line):
                if m.start() > 0 and line[m.start() - 1].isalpha():
                    continue  # repricer/reprices — measurement vocabulary, not a target
                if line[max(0, m.start() - 4):m.start()].lower().endswith("non-"):
                    continue
                if _GUARD_BEFORE.search(line[:m.start()]):
                    continue
                if _GUARD_AFTER.search(line[m.end():m.end() + 60]):
                    continue
                flag = True
                break
        if flag:
            bad.append((line_no, line.strip()[:240]))
    return bad


def lint_file(path):
    try:
        with open(path, errors="replace") as f:
            return unsafe_policy_lines(f.read())
    except OSError:
        return []


# Recipe-lint precision rework (2026-07-13): the old patterns rejected 303/574 wave6
# deciders at a measured ~98% false-positive rate (e.g. "Keep graph input FLOAT
# [1,10,30,30] … Add(one_i64)" tripped `input.*i64`; the family name EINSUM-COMPRESS
# tripped case-insensitive \bCompress\b; "no Where, Pad, OneHot, Loop" tripped because
# _NEGATION lacks plain "no"). Each rejection threw away a completed xhigh decider run
# and re-queued the task. New rules fire only on the op being INVOKED and on the input
# dtype being CHANGED. _NEGATION itself is untouched — it also guards the scorer-hack
# lint, which must stay strict.
_RECIPE_NEGATION = re.compile(
    r"(?i)\b(?:never|do not|don't|avoid|forbid|forbidden|ban|banned|reject|no|without|not)\b")
# case-SENSITIVE op reference in call/backtick/node context — prose "loop over rows",
# "scan", "nonzero-any", "EINSUM-COMPRESS" no longer match
_RECIPE_BANNED_OP = re.compile(
    r"`(?:Loop|Scan|NonZero|Unique|Compress)(?:\(|`)"
    r"|\b(?:Loop|Scan|NonZero|Unique|Compress)\s*\("
    r"|\b(?:Loop|Scan|NonZero|Unique|Compress)\s+(?:node|op|operator)\b")
# input DECLARATION change only: "change graph input to UINT8", "input dtype i64",
# "declare input as int8", "input: uint8". NOT co-occurrence of "input" + int dtype,
# and NOT `Cast(input, X)` — casting the input INSIDE the graph is legal (the gate124
# contract itself says "Cast input→uint8 as the FIRST node, never DECLARE input as
# uint8/int"). The gap excludes , . ; : so an unrelated clause after "input," can't
# bridge to a dtype word.
_RECIPE_INPUT_CHANGE = re.compile(
    r"(?i)\b(?:graph\s+)?input\b[^.;:,`\n]{0,30}?\b(?:as|to|dtype|declared?\s+as|becomes?)\s*"
    r"[`'\"]?\s*(?:u?int8|bool|i32|i64|int32|int64)\b"
    r"|\b(?:graph\s+)?input\s+(?:to|=)?\s*[`'\"]?(?:UINT8|INT8|BOOL|INT32|INT64)\s*`?\[1,\s*\d+"
    r"|\binput\s*[:=]\s*[`'\"]?(?:u?int8|bool|i32|i64|int32|int64)\b")
_RECIPE_KEEPS_FLOAT = re.compile(
    r"(?i)\binput\b[^.;\n]{0,30}\b(?:remains?|stays?|is|as)\s+(?:exactly\s+)?FLOAT\b"
    r"|keep\s+(?:the\s+)?(?:graph\s+)?input\s+(?:as\s+)?FLOAT\b"
    r"|input\s+FLOAT\s*`?\[1,\s*10,\s*30,\s*30\]")


def recipe_errors(text):
    """Cheap pre-build contract lint for the decider's BUILD-RECIPE section."""
    section = text.split("BUILD-RECIPE:", 1)[-1].split("FAMILY-BUDGET-JSON:", 1)[0]
    errors = []
    for line_no, line in enumerate(section.splitlines(), 1):
        if _RECIPE_NEGATION.search(line):
            continue
        if _RECIPE_BANNED_OP.search(line):
            errors.append((line_no, "banned op in BUILD-RECIPE: " + line.strip()[:200]))
        if _RECIPE_INPUT_CHANGE.search(line) and not _RECIPE_KEEPS_FLOAT.search(line):
            errors.append((line_no, "input must remain FLOAT [1,10,30,30]: " + line.strip()[:200]))
        if re.search(r"(?i)(?:dynamic|symbolic|unset)\s+(?:shape|dim)", line):
            errors.append((line_no, "dynamic shape in BUILD-RECIPE: " + line.strip()[:200]))
    return errors
