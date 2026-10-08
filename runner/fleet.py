#!/usr/bin/env python3
"""fleet — resize a LIVE wave's per-provider worker seats, no restart needed.

The orchestrator re-reads results/FLEET.json at the top of every spawn pass:
increase → new seats fill from the same queue on the next 10s tick; decrease → drain
(running attempts finish; no kills). Per-provider MAX_CONC env caps remain hard ceilings.

  python3 runner/fleet.py status
  python3 runner/fleet.py set codex=12
  python3 runner/fleet.py add codex 1
  python3 runner/fleet.py remove codex 2

Live DECIDER pool sizing (same ~10s pickup, drain-not-kill; 0 drains the lane).
Unset => the wave's launch-time DECIDER_MAX_CONC x shares split applies:

  python3 runner/fleet.py deciders codex=10
  python3 runner/fleet.py deciders          # show current

Group wind-down (mid-run, ~10s pickup, drain-not-kill; `on` restores exactly —
seat counts are masked at read time, never destroyed):

  python3 runner/fleet.py off sol        # codex arm → all gpt-5.5; sol decider lane off
  python3 runner/fleet.py off g55        # codex arm → all sol; gpt-5.5 decider lane off
  python3 runner/fleet.py off openai     # codex seats → 0 AND both decider lanes off (emergency)
  python3 runner/fleet.py on <group>     # restore
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
FLEET_FILE = os.path.join(RESULTS, "FLEET.json")
PROVIDERS = ("codex",)
DECIDER_PROVIDERS = ("codex",)
GROUPS = ("openai", "sol", "g55")


def read_fleet():
    try:
        with open(FLEET_FILE) as f:
            raw = json.load(f)
        return {p: max(0, int(raw.get(p, 0))) for p in PROVIDERS}
    except (OSError, ValueError, TypeError):
        return None


def read_off():
    try:
        with open(FLEET_FILE) as f:
            raw = json.load(f)
        return {str(x).strip().lower() for x in (raw.get("off") or [])
                if isinstance(x, str) and str(x).strip()}
    except (OSError, ValueError, TypeError):
        return set()


def read_deciders():
    """Live per-provider DECIDER pool sizes, or None when never set (then the wave's
    launch-time DECIDER_MAX_CONC x shares split applies)."""
    try:
        with open(FLEET_FILE) as f:
            raw = json.load(f)
        d = raw.get("deciders")
        if not isinstance(d, dict):
            return None
        out = {p: max(0, int(v)) for p, v in d.items() if p in DECIDER_PROVIDERS}
        return out or None
    except (OSError, ValueError, TypeError):
        return None


def write_fleet(desired, reason, off=None, deciders=None):
    off = read_off() if off is None else set(off)
    deciders = read_deciders() if deciders is None else deciders
    os.makedirs(RESULTS, exist_ok=True)
    tmp = FLEET_FILE + ".tmp"
    body = {**desired, "off": sorted(off)}
    if deciders:
        body["deciders"] = deciders
    body.update({"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "note": reason})
    with open(tmp, "w") as f:
        json.dump(body, f, indent=1)
    os.replace(tmp, FLEET_FILE)
    try:
        with open(os.path.join(RESULTS, "ORCHESTRATOR.log"), "a") as f:
            f.write(f"[{time.strftime('%Y%m%dT%H%M%S')}] FLEET operator update: "
                    f"{desired} off={sorted(off)} deciders={deciders} ({reason})\n")
    except OSError:
        pass
    print(f"fleet -> {desired} off={sorted(off)}"
          f"{' deciders=' + str(deciders) if deciders else ''}  "
          f"(picked up within ~10s; decrease drains, no kills)")


def active_counts():
    """Live per-provider active workers from the freshest RUN_STATE snapshot."""
    snaps = sorted(glob.glob(os.path.join(RESULTS, "RUN_STATE*.json")),
                   key=lambda p: -os.path.getmtime(p))
    for snap in snaps:
        try:
            with open(snap) as f:
                d = json.load(f)
            age = time.time() - os.path.getmtime(snap)
            workers = d.get("workers") or []
            act = {p: 0 for p in PROVIDERS}
            for w in workers:
                act[w.get("provider") or w.get("model", "?")] = \
                    act.get(w.get("provider") or w.get("model", "?"), 0) + 1
            return act, snap, age
        except (OSError, ValueError):
            continue
    return None, None, None


def cmd_status():
    desired = read_fleet()
    act, snap, age = active_counts()
    print(f"desired: {desired if desired is not None else '(no FLEET.json — legacy --workers mode)'}")
    dec = read_deciders()
    print(f"deciders:{dec if dec else ' (unset — wave uses its launch-time split)'}")
    off = read_off()
    if off:
        print(f"off:     {sorted(off)}  (masked at read: openai→codex=0, "
              f"sol/g55 steer the codex arm + decider lanes)")
    if act is not None:
        stale = " (STALE >5min — orchestrator down?)" if age > 300 else ""
        print(f"active:  {act}  [from {os.path.basename(snap)}, {int(age)}s old{stale}]")
    else:
        print("active:  (no RUN_STATE snapshot — wave not running?)")
    status = os.path.join(RESULTS, "STATUS.txt")
    if os.path.exists(status):
        print("STATUS: ", open(status).read().strip()[:300])


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("status", "set", "add", "remove", "off",
                                                "on", "deciders"):
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "status":
        return cmd_status()
    desired = read_fleet() or {p: 0 for p in PROVIDERS}
    if cmd == "deciders":
        cur = read_deciders() or {}
        if len(sys.argv) < 3:
            sys.exit(f"deciders: {cur or '(unset — launch-time split applies)'}\n"
                     f"usage: fleet.py deciders codex=N  (0 drains the lane)")
        for kv in sys.argv[2:]:
            if "=" not in kv or kv.split("=", 1)[0] not in DECIDER_PROVIDERS:
                sys.exit(f"usage: fleet.py deciders codex=N  "
                         f"(providers: {DECIDER_PROVIDERS})")
            p, v = kv.split("=", 1)
            try:
                cur[p] = max(0, int(v))
            except ValueError:
                sys.exit(f"not an integer: {kv}")
        return write_fleet(desired, f"deciders {' '.join(sys.argv[2:])}", deciders=cur)
    if cmd in ("off", "on"):
        if len(sys.argv) < 3 or sys.argv[2].lower() not in GROUPS:
            sys.exit(f"usage: fleet.py {cmd} <group>  with group in {GROUPS}")
        group = sys.argv[2].lower()
        off = read_off()
        (off.add if cmd == "off" else off.discard)(group)
        return write_fleet(desired, f"{cmd} {group}", off=off)
    if cmd == "set":
        for kv in sys.argv[2:]:
            p, _, n = kv.partition("=")
            if p not in PROVIDERS or not n.isdigit():
                sys.exit(f"bad arg {kv!r} — use provider=N with provider in {PROVIDERS}")
            desired[p] = int(n)
        return write_fleet(desired, "set " + " ".join(sys.argv[2:]))
    p, n = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if p not in PROVIDERS:
        sys.exit(f"unknown provider {p!r} — one of {PROVIDERS}")
    desired[p] = max(0, desired[p] + (n if cmd == "add" else -n))
    return write_fleet(desired, f"{cmd} {p} {n}")


if __name__ == "__main__":
    main()
