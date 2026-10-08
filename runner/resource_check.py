#!/usr/bin/env python3
"""Fail-closed host resource audit for a NeuroGolf wave launch."""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from config import CLEAN as _CLEAN
CLEAN = Path(_CLEAN)


def gib(n):
    return n / (1024 ** 3)


def orphaned_scorers():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().split()
            ppid = int(stat[3])
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, ValueError):
            continue
        if (ppid == 1 and "spec_from_file_location('neurogolf_utils'" in cmd
                and "neurogolf_clean" in cmd):
            found.append(int(entry.name))
    return sorted(found)


def audit(min_mem_gb=64, min_disk_gb=40, min_pids=1000, max_temp_gb=50):
    checks = []
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    available = gib(mem.get("MemAvailable", 0))
    cgroup_available = available
    try:
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
        raw_max = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw_max != "max":
            cgroup_available = gib(max(0, int(raw_max) - current))
        events = {line.split()[0]: int(line.split()[1]) for line in
                  Path("/sys/fs/cgroup/memory.events").read_text().splitlines()}
        # The cgroup counter is cumulative and unresettable in this container; a
        # pre-existing kill would otherwise block every future launch. OOM_KILL_MAX
        # (default 0 = fail-closed) lets a launch tolerate exactly the known stale
        # count — any NEW kill still trips the check.
        oom_max = int(os.environ.get("OOM_KILL_MAX", "0"))
        checks.append(("cgroup OOM history", events.get("oom_kill", 0) <= oom_max,
                       f"oom_kill={events.get('oom_kill', 0)} (tolerated<={oom_max})"))
    except (OSError, ValueError, IndexError):
        pass
    usable = min(available, cgroup_available)
    checks.append(("available RAM", usable >= min_mem_gb,
                   f"{usable:.1f} GiB usable (host {available:.1f}, cgroup "
                   f"{cgroup_available:.1f}; minimum {min_mem_gb})"))

    seen = set()
    for label, path in (("workspace disk", ROOT), ("pin disk", CLEAN),
                        ("temp disk", Path(os.environ.get("TMPDIR", "/tmp")))):
        try:
            usage = shutil.disk_usage(path)
            device = os.stat(path).st_dev
            if device in seen:
                continue
            seen.add(device)
            free = gib(usage.free)
            checks.append((label, free >= min_disk_gb,
                           f"{free:.1f} GiB free (minimum {min_disk_gb})"))
        except OSError as exc:
            checks.append((label, False, str(exc)))

    try:
        current = int(Path("/sys/fs/cgroup/pids.current").read_text())
        raw_max = Path("/sys/fs/cgroup/pids.max").read_text().strip()
        maximum = int(raw_max) if raw_max != "max" else current + min_pids
        headroom = maximum - current
        checks.append(("cgroup PID headroom", headroom >= min_pids,
                       f"{headroom} slots (minimum {min_pids})"))
    except (OSError, ValueError) as exc:
        checks.append(("cgroup PID headroom", False, str(exc)))

    managed_tmp = Path("/tmp/ngolf_ort")
    temp_bytes = 0
    try:
        for p in managed_tmp.rglob("*") if managed_tmp.exists() else ():
            try:
                if p.is_file() and not p.is_symlink():
                    temp_bytes += p.stat().st_size
            except OSError:
                pass
        temp_gb = gib(temp_bytes)
        checks.append(("managed scoring temp", temp_gb <= max_temp_gb,
                       f"{temp_gb:.1f} GiB (maximum {max_temp_gb})"))
    except OSError as exc:
        checks.append(("managed scoring temp", False, str(exc)))

    orphans = orphaned_scorers()
    checks.append(("orphaned scorers", not orphans,
                   "none" if not orphans else ",".join(map(str, orphans))))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mem-gb", type=float,
                    default=float(os.environ.get("RESOURCE_MIN_MEM_GB", 64)))
    ap.add_argument("--min-disk-gb", type=float,
                    default=float(os.environ.get("RESOURCE_MIN_DISK_GB", 40)))
    ap.add_argument("--min-pids", type=int,
                    default=int(os.environ.get("RESOURCE_MIN_PIDS", 1000)))
    ap.add_argument("--max-temp-gb", type=float,
                    default=float(os.environ.get("RESOURCE_MAX_TEMP_GB", 50)))
    a = ap.parse_args()
    checks = audit(a.min_mem_gb, a.min_disk_gb, a.min_pids, a.max_temp_gb)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
