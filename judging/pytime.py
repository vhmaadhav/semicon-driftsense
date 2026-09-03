#!/usr/bin/env python3
"""Drop-in helper for GNU time -v -o <outfile> cmd..."""
import os
import resource
import subprocess
import sys
import time

def main():
    args = sys.argv[1:]
    outfile = None
    cmd = []
    i = 0
    while i < len(args):
        if args[i] == "-v":
            i += 1
        elif args[i] == "-o":
            outfile = args[i + 1]
            i += 2
        elif args[i].startswith("-o"):
            outfile = args[i][2:]
            i += 1
        elif args[i] == "--":
            cmd = args[i + 1:]
            break
        else:
            cmd = args[i:]
            break

    if not cmd:
        sys.exit(1)

    t0 = time.perf_counter()
    p = subprocess.Popen(cmd)
    _, status, rusage = os.wait4(p.pid, 0)
    wall = time.perf_counter() - t0

    exit_code = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else (status >> 8)
    utime = rusage.ru_utime
    stime = rusage.ru_stime
    cpu_pct = int(round(100.0 * (utime + stime) / max(wall, 1e-6)))
    maxrss = rusage.ru_maxrss

    m = int(wall // 60)
    s = wall % 60
    wall_str = f"{m}:{s:05.2f}"

    lines = [
        f"\tPercent of CPU this job got: {cpu_pct}%",
        f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {wall_str}",
        f"\tMaximum resident set size (kbytes): {maxrss}",
    ]

    if outfile:
        with open(outfile, "w") as f:
            f.write("\n".join(lines) + "\n")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
