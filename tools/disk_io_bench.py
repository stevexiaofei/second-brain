#!/usr/bin/env python3
"""Standalone disk I/O benchmark for comparing mount points (no 3rd-party deps).

Usage:
    python3 tools/disk_io_bench.py /a2000 /home/weijun.zhang
    python3 tools/disk_io_bench.py /a2000/APS/steve /home/weijun.zhang --quick
    python3 tools/disk_io_bench.py /data/foo --size-mb 4096 --rand-ops 5000

Phases per target (scratch files live in a new <target>/.disk_io_bench_*/
directory; cleanup is attempted on normal exit, exceptions or KeyboardInterrupt):
    seq_write_buf       1MiB sequential writes, buffered, final fsync included
    seq_write_sync      1MiB sequential writes, fsync after every block
    seq_read            sequential read after requesting page-cache eviction
    rand_write_4k_buf   random 4KiB overwrites, final fsync included
    rand_read_4k        random 4KiB reads of fully initialized data after eviction advice
    rand_write_4k_sync  random 4KiB writes + fsync each (also ~fsync latency)

Caveats:
    * Linux only; not an O_DIRECT benchmark. Buffered write throughput includes
      the final fsync, but batching/coalescing and storage caches still matter.
      DONTNEED is advisory and does not evict server/controller caches. Increasing
      file size does not guarantee device-level measurements.
    * Payload blocks are random but reused; deduplicating storage may optimize
      them. Use fio with suitable buffer settings for controlled device tests.
    * Run while the disk is otherwise idle, otherwise results are noise.
    * If the given directory is read-only (e.g. /a2000 root), pass a writable
      subdirectory instead.

Methodology notes and result interpretation:
    docs/knowledge/engineering/profiling/08-disk-io-benchmark.md
"""
import argparse
import errno
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

MIB = 1024 * 1024
KIB = 1024
RAND_BLOCK = 4 * KIB
# No-op on Linux; preserve raw bytes in the cross-platform regression tests.
BINARY_FLAG = getattr(os, "O_BINARY", 0)
TARGET_ERRORS = {errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOSPC}
if hasattr(errno, "EDQUOT"):
    TARGET_ERRORS.add(errno.EDQUOT)

# (key, label, unit, higher_is_better)
METRICS = [
    ("seq_write_buf", "seq_write_buf", "MiB/s", True),
    ("seq_write_sync", "seq_write_sync", "MiB/s", True),
    ("seq_read", "seq_read", "MiB/s", True),
    ("rand_write_4k_buf", "rand_write_4k_buf", "IOPS", True),
    ("rand_read_4k", "rand_read_4k", "IOPS", True),
    ("rand_write_4k_sync", "rand_write_4k_sync", "IOPS", True),
    ("fsync_lat_p99", "fsync_latency_p99", "ms", False),
    ("rand_read_lat_avg", "rand_read_lat_avg", "ms", False),
]


def fmt_bw(num_bytes, dt):
    return num_bytes / dt / MIB


def fmt_iops(n, dt):
    return n / dt


def pct(sorted_lat, q):
    if not sorted_lat:
        return 0.0
    idx = min(int(q * (len(sorted_lat) - 1)), len(sorted_lat) - 1)
    return sorted_lat[idx]


def fs_type(path):
    try:
        out = subprocess.run(
            ["df", "-T", "-P", path],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return out[-1].split()[1] if len(out) > 1 else "?"
    except Exception:
        return "?"


def bench_seq_write(path, size_bytes, block, sync_each):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | BINARY_FLAG, 0o600)
    payload = os.urandom(block)
    written = 0
    t0 = time.perf_counter()
    try:
        while written < size_bytes:
            # Honor exact length and handle short writes without overshooting.
            view = memoryview(payload)[:min(block, size_bytes - written)]
            while view:
                n = os.write(fd, view)
                if n == 0:
                    raise OSError(errno.EIO, "write made no progress")
                written += n
                view = view[n:]
            if sync_each:
                os.fsync(fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    return written, time.perf_counter() - t0


def bench_seq_read(path):
    fd = os.open(path, os.O_RDONLY | BINARY_FLAG)
    try:
        # Best-effort eviction of clean pages; prior writes have been fsynced.
        # This does not flush server/controller caches or guarantee disk I/O.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        buf = bytearray(MIB)
        total = 0
        t0 = time.perf_counter()
        while True:
            n = os.readv(fd, [buf])
            if n == 0:
                break
            total += n
        return total, time.perf_counter() - t0
    finally:
        os.close(fd)


def bench_rand_write(path, offsets, payload, sync_each):
    # The file is populated before this function. Do not truncate it here:
    # random-read would otherwise observe mostly sparse holes.
    fd = os.open(path, os.O_WRONLY | BINARY_FLAG)
    lat = []
    t0 = time.perf_counter()
    try:
        for off in offsets:
            s = time.perf_counter()
            view = memoryview(payload)
            completed = 0
            while view:
                n = os.pwrite(fd, view, off + completed)
                if n == 0:
                    raise OSError(errno.EIO, "pwrite made no progress")
                completed += n
                view = view[n:]
            if sync_each:
                os.fsync(fd)
            lat.append(time.perf_counter() - s)
        if not sync_each:
            os.fsync(fd)
    finally:
        os.close(fd)
    return len(offsets), time.perf_counter() - t0, lat


def bench_rand_read(path, offsets, buf_size):
    fd = os.open(path, os.O_RDONLY | BINARY_FLAG)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        buf = bytearray(buf_size)
        lat = []
        t0 = time.perf_counter()
        for off in offsets:
            s = time.perf_counter()
            completed = 0
            while completed < buf_size:
                n = os.preadv(fd, [memoryview(buf)[completed:]], off + completed)
                if n == 0:
                    raise OSError(errno.EIO, "unexpected EOF in random read")
                completed += n
            lat.append(time.perf_counter() - s)
        return len(offsets), time.perf_counter() - t0, lat
    finally:
        os.close(fd)


def run_target(target, args):
    print(f"\n=== {target}  (fs: {fs_type(target)}) ===", flush=True)

    du = shutil.disk_usage(target)
    need = int(args.size_mb * 1.2) * MIB
    if du.free == 0:
        # Some NFS/Isilon exports report filesystem-level free space that
        # differs from user quota; it may also be genuinely full.
        print("  note: filesystem reports 0 bytes free (NFS quota artifact?), "
              "attempting anyway")
    elif du.free < need:
        print(f"  SKIP: only {du.free / MIB:.0f} MiB free, need ~{need / MIB:.0f} MiB")
        return None

    size_bytes = args.size_mb * MIB
    rng = random.Random(args.seed)
    n_blocks = size_bytes // RAND_BLOCK
    w_off = [rng.randrange(n_blocks) * RAND_BLOCK for _ in range(args.rand_ops)]
    r_off = [rng.randrange(n_blocks) * RAND_BLOCK for _ in range(args.rand_ops)]
    payload4k = os.urandom(RAND_BLOCK)

    # mkdtemp creates a new, uniquely owned directory; cleanup never touches
    # an existing directory with a coincidentally matching PID/timestamp.
    scratch = tempfile.mkdtemp(prefix=".disk_io_bench_", dir=target)
    seq_path = os.path.join(scratch, "seq.bin")
    rand_path = os.path.join(scratch, "rand.bin")

    res = {}
    try:
        print(f"  seq_write_buf  ({args.size_mb} MiB, 1MiB blocks)...", flush=True)
        n, dt = bench_seq_write(seq_path, size_bytes, MIB, sync_each=False)
        res["seq_write_buf"] = fmt_bw(n, dt)
        print(f"    -> {res['seq_write_buf']:.1f} MiB/s", flush=True)

        print(f"  seq_write_sync (fsync per block)...", flush=True)
        n, dt = bench_seq_write(seq_path, size_bytes, MIB, sync_each=True)
        res["seq_write_sync"] = fmt_bw(n, dt)
        print(f"    -> {res['seq_write_sync']:.1f} MiB/s", flush=True)

        print("  seq_read (page-cache eviction requested)...", flush=True)
        n, dt = bench_seq_read(seq_path)
        res["seq_read"] = fmt_bw(n, dt)
        print(f"    -> {res['seq_read']:.1f} MiB/s", flush=True)

        # Reuse the fully written/fsynced sequential file. Every random read
        # now addresses initialized data, not unwritten holes. Only one full
        # file is needed, and setup is outside the random-phase timers.
        os.rename(seq_path, rand_path)
        print(f"  rand_write_4k_buf ({args.rand_ops} ops)...", flush=True)
        n, dt, _ = bench_rand_write(rand_path, w_off, payload4k, False)
        res["rand_write_4k_buf"] = fmt_iops(n, dt)
        print(f"    -> {res['rand_write_4k_buf']:.1f} IOPS", flush=True)

        print("  rand_read_4k (page-cache eviction requested)...", flush=True)
        n, dt, lat = bench_rand_read(rand_path, r_off, RAND_BLOCK)
        res["rand_read_4k"] = fmt_iops(n, dt)
        lat.sort()
        res["rand_read_lat_avg"] = sum(lat) / len(lat) * 1e3
        print(f"    -> {res['rand_read_4k']:.1f} IOPS "
              f"(avg {res['rand_read_lat_avg']:.2f} ms)", flush=True)

        print(f"  rand_write_4k_sync ({args.rand_ops} ops, fsync each)...", flush=True)
        n, dt, lat = bench_rand_write(rand_path, w_off, payload4k, True)
        res["rand_write_4k_sync"] = fmt_iops(n, dt)
        lat.sort()
        res["fsync_lat_p99"] = pct(lat, 0.99) * 1e3
        print(f"    -> {res['rand_write_4k_sync']:.1f} IOPS "
              f"(fsync p99 {res['fsync_lat_p99']:.2f} ms)", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return res


def print_table(results, targets):
    print("\n" + "=" * 72)
    print("RESULTS  (* = best per row)")
    print("=" * 72)
    header = f"{'metric':<22}" + "".join(f"{t[:24]:>25}" for t in targets)
    print(header)
    for key, label, unit, higher_better in METRICS:
        vals = [results.get(t, {}).get(key) for t in targets]
        numeric = [v for v in vals if v is not None]
        best = None
        if numeric:
            best = max(numeric) if higher_better else min(numeric)
        cells = []
        for v in vals:
            if v is None:
                cells.append(f"{'n/a':>25}")
            else:
                star = "*" if len(numeric) > 1 and v == best else ""
                cells.append(f"{v:>19.1f} {unit:>5}{star}")
        print(f"{label:<22}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description="Disk I/O benchmark (see file docstring)")
    ap.add_argument("targets", nargs="+", help="directories to test (one per mount)")
    ap.add_argument("--size-mb", type=int, default=256,
                    help="seq test file size in MiB (default 256)")
    ap.add_argument("--rand-ops", type=int, default=1000,
                    help="number of random 4K operations per phase (default 1000)")
    ap.add_argument("--quick", action="store_true",
                    help="small fast run: 64 MiB / 300 ops")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.size_mb, args.rand_ops = 64, 300
    if args.size_mb <= 0 or args.rand_ops <= 0:
        ap.error("--size-mb and --rand-ops must be positive")
    required = ("pwrite", "preadv", "readv", "posix_fadvise", "POSIX_FADV_DONTNEED")
    if not sys.platform.startswith("linux") or any(not hasattr(os, name) for name in required):
        ap.error("this benchmark requires Linux positional/vectored I/O and posix_fadvise")

    results = {}
    t0 = time.perf_counter()
    for target in args.targets:
        if not os.path.isdir(target):
            print(f"\n=== {target}: SKIP (not a directory)")
            continue
        try:
            res = run_target(target, args)
        except OSError as exc:
            if exc.errno not in TARGET_ERRORS:
                raise
            print(f"  SKIP: {exc.strerror}; use a writable directory with "
                  "sufficient space/quota")
            continue
        if res is not None:
            results[target] = res

    if len(results) > 0:
        tested = [t for t in args.targets if t in results]
        print_table(results, tested)
        print(f"\ntotal wall time: {time.perf_counter() - t0:.1f}s")
        print("note: writes include fsync; cache-eviction advice is best-effort. "
              "Server/controller caches and batching still affect results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
