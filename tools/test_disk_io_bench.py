#!/usr/bin/env python3
"""Small regression tests; write only temporary files, never user mounts.

Run: python -I tools/test_disk_io_bench.py
The portable lifecycle test emulates positional/vectored syscalls using ordinary
file I/O. A separate real-syscall lifecycle test runs only on Linux. Both use
small temporary files and are not performance measurements.
"""
import contextlib
import errno
import importlib.util
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "disk_io_bench", Path(__file__).with_name("disk_io_bench.py")
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


@contextlib.contextmanager
def linux_api_stubs():
    """Expose missing API names for CLI tests that mock all actual I/O."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(BENCH.sys, "platform", "linux"))
        for name in ("pwrite", "preadv", "readv", "posix_fadvise"):
            stack.enter_context(mock.patch.object(BENCH.os, name, create=True))
        stack.enter_context(mock.patch.object(
            BENCH.os, "POSIX_FADV_DONTNEED", 4, create=True
        ))
        yield


class DiskIoBenchTests(unittest.TestCase):
    def test_percentile_order_statistics(self):
        self.assertEqual(BENCH.pct(list(range(1000)), 0.99), 989)
        self.assertEqual(BENCH.pct(list(range(300)), 0.99), 296)
        self.assertEqual(BENCH.pct([], 0.99), 0.0)

    def test_seq_write_exact_size_and_final_fsync(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            with mock.patch.object(BENCH.os, "fsync", wraps=os.fsync) as sync:
                n, _ = BENCH.bench_seq_write(path, 5000, 4096, False)
            self.assertEqual(n, 5000)
            self.assertEqual(path.stat().st_size, 5000)
            sync.assert_called_once()

    def test_seq_short_writes_do_not_lose_data(self):
        write = os.write
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            with mock.patch.object(BENCH.os, "write", side_effect=(
                lambda fd, data: write(fd, data[:100])
            )):
                n, _ = BENCH.bench_seq_write(path, 4096, 4096, False)
            self.assertEqual(n, 4096)
            self.assertEqual(path.stat().st_size, 4096)

    def test_zero_progress_write_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(BENCH.os, "write", return_value=0):
                with self.assertRaises(OSError) as error:
                    BENCH.bench_seq_write(Path(directory) / "data.bin", 4096, 4096, False)
            self.assertEqual(error.exception.errno, errno.EIO)

    def test_short_random_write_preserves_size_and_unwritten_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(b"a" * 12288)

            def short_pwrite(fd, data, offset):
                os.lseek(fd, offset, os.SEEK_SET)
                return os.write(fd, data[:1024])

            with mock.patch.object(BENCH.os, "pwrite", side_effect=short_pwrite, create=True):
                n, _, latencies = BENCH.bench_rand_write(
                    path, [4096], b"b" * 4096, True
                )
            self.assertEqual(n, 1)
            self.assertEqual(len(latencies), 1)
            self.assertEqual(path.read_bytes(), b"a" * 4096 + b"b" * 4096 + b"a" * 4096)

    def test_random_read_short_reads_and_eof(self):
        with tempfile.TemporaryDirectory() as directory, linux_api_stubs():
            path = Path(directory) / "data.bin"
            path.write_bytes(b"a" * 4096)

            def short_preadv(fd, buffers, offset):
                os.lseek(fd, offset, os.SEEK_SET)
                data = os.read(fd, min(len(buffers[0]), 1024))
                buffers[0][:len(data)] = data
                return len(data)

            BENCH.os.preadv.side_effect = short_preadv
            n, _, _ = BENCH.bench_rand_read(path, [0], 4096)
            self.assertEqual(n, 1)
            with self.assertRaises(OSError) as error:
                BENCH.bench_rand_read(path, [4096], 4096)
            self.assertEqual(error.exception.errno, errno.EIO)

    def test_lifecycle_uses_4k_initialized_data_and_cleans_only_own_directory(self):
        writes, reads = [], []

        def pwrite(fd, data, offset):
            writes.append((offset, len(data)))
            self.assertEqual(os.fstat(fd).st_size, BENCH.MIB)
            os.lseek(fd, offset, os.SEEK_SET)
            return os.write(fd, data)

        def preadv(fd, buffers, offset):
            reads.append((offset, len(buffers[0])))
            os.lseek(fd, offset, os.SEEK_SET)
            data = os.read(fd, len(buffers[0]))
            # Every read, including locations not randomly overwritten,
            # must address previously initialized data rather than holes.
            self.assertEqual(data, b"r" * BENCH.RAND_BLOCK)
            buffers[0][:len(data)] = data
            return len(data)

        def readv(fd, buffers):
            data = os.read(fd, len(buffers[0]))
            buffers[0][:len(data)] = data
            return len(data)

        args = SimpleNamespace(size_mb=1, rand_ops=8, seed=0)
        with tempfile.TemporaryDirectory() as directory, linux_api_stubs():
            sentinel = Path(directory) / ".disk_io_bench_existing"
            sentinel.write_text("keep", encoding="utf-8")
            BENCH.os.pwrite.side_effect = pwrite
            BENCH.os.preadv.side_effect = preadv
            BENCH.os.readv.side_effect = readv
            with mock.patch.object(BENCH, "fs_type", return_value="test"), \
                 mock.patch.object(BENCH.os, "urandom", side_effect=lambda n: b"r" * n), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = BENCH.run_target(directory, args)
            self.assertEqual(len(writes), 16)
            self.assertEqual(len(reads), 8)
            self.assertEqual(BENCH.RAND_BLOCK, 4096)
            for offset, length in writes + reads:
                self.assertEqual(length, 4096)
                self.assertEqual(offset % 4096, 0)
                self.assertLessEqual(offset + length, BENCH.MIB)
            self.assertEqual(set(result), {row[0] for row in BENCH.METRICS})
            self.assertEqual(list(Path(directory).iterdir()), [sentinel])

    @unittest.skipUnless(BENCH.sys.platform.startswith("linux"), "requires real Linux I/O APIs")
    def test_real_linux_lifecycle_in_temporary_directory(self):
        args = SimpleNamespace(size_mb=1, rand_ops=8, seed=0)
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(io.StringIO()):
            result = BENCH.run_target(directory, args)
            self.assertEqual(set(result), {row[0] for row in BENCH.METRICS})
            self.assertTrue(all(value > 0 for value in result.values()))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_phase_cleans_scratch_directory(self):
        args = SimpleNamespace(size_mb=1, rand_ops=1, seed=0)
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(BENCH, "fs_type", return_value="test"), \
             mock.patch.object(BENCH, "bench_seq_write", side_effect=OSError(errno.ENOSPC, "full")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                BENCH.run_target(directory, args)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_main_continues_after_read_only_target(self):
        with linux_api_stubs(), \
             mock.patch.object(BENCH.sys, "argv", ["bench", "readonly", "writable"]), \
             mock.patch.object(BENCH.os.path, "isdir", return_value=True), \
             mock.patch.object(BENCH, "run_target", side_effect=[
                 OSError(errno.EROFS, "Read-only file system"), {"seq_read": 1.0}
             ]) as run, \
             mock.patch.object(BENCH, "print_table") as table, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(BENCH.main(), 0)
            self.assertEqual(run.call_count, 2)
            table.assert_called_once_with({"writable": {"seq_read": 1.0}}, ["writable"])
            self.assertIn("Read-only file system", output.getvalue())

    def test_unexpected_io_error_is_not_silently_skipped(self):
        with linux_api_stubs(), \
             mock.patch.object(BENCH.sys, "argv", ["bench", "target"]), \
             mock.patch.object(BENCH.os.path, "isdir", return_value=True), \
             mock.patch.object(BENCH, "run_target", side_effect=OSError(errno.EIO, "I/O error")):
            with self.assertRaises(OSError):
                BENCH.main()

    def test_rejects_nonpositive_arguments(self):
        for flag in ("--size-mb", "--rand-ops"):
            with self.subTest(flag=flag), \
                 mock.patch.object(BENCH.sys, "argv", ["bench", "target", flag, "0"]), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    BENCH.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
