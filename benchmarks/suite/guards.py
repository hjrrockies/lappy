"""Resource guards for benchmark solves. Import and call `install()`.

Every entry point that runs a solve should install these. A single solve is
small (~170MB), but this run has produced two failure modes that are not:

  * unbounded memory -- a `rect_thin` solve once reached a 59.8GB footprint with
    40GB of swap, taking the machine with it;
  * non-termination -- a noisy tension curve can keep the bracket search
    refining indefinitely.

Design notes, each learned the hard way:

**Poll system swap, not process RSS.** On macOS memory is compressed, so
`ru_maxrss` reports the *resident* size while the true footprint can be an order
of magnitude larger -- the 59.8GB runaway read as 4.7GB resident. Swap growth is
what actually correlates with the machine becoming unusable.

**Measure growth against a baseline, not an absolute.** Other applications use
swap too; what matters is how much *this* process added.

**A hard per-process cap is unavailable.** `setrlimit(RLIMIT_AS/RLIMIT_DATA)`
fails on macOS with "current limit exceeds maximum limit" when the hard limit is
infinity, and `ulimit -v` is a no-op. So the guard has to observe and abort
rather than prevent.

`os._exit` is deliberate: it skips interpreter cleanup, which can itself hang or
allocate when the process is already in trouble.
"""
import os
import subprocess
import sys
import threading
import time

EXIT_MEMORY = 97
EXIT_TIMEOUT = 98

DEFAULT_SWAP_MB = float(os.environ.get('LAPPY_RUN_SWAP_MB', '3500'))
DEFAULT_TIMEOUT_S = float(os.environ.get('LAPPY_RUN_TIMEOUT_S', '1500'))


def swap_used_mb():
    """System-wide swap in use, MB. Returns 0.0 where unavailable."""
    try:
        out = subprocess.run(['sysctl', '-n', 'vm.swapusage'],
                             capture_output=True, text=True, timeout=5).stdout
        used = out.split('used =')[1].split()[0]      # "used = 619.62M"
        return float(used.rstrip('MG')) * (1024 if used.endswith('G') else 1)
    except Exception:
        return 0.0


def install(swap_mb=None, timeout_s=None, poll=5.0, label=''):
    """Abort this process on runaway memory or runtime.

    Returns the watchdog thread (daemon, so it never blocks exit).
    """
    swap_mb = DEFAULT_SWAP_MB if swap_mb is None else swap_mb
    timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    baseline = swap_used_mb()
    started = time.time()
    tag = f' [{label}]' if label else ''

    def watch():
        while True:
            time.sleep(poll)
            elapsed = time.time() - started
            if timeout_s and elapsed > timeout_s:
                sys.stderr.write(
                    f'GUARD{tag}: timeout after {elapsed:.0f}s '
                    f'(limit {timeout_s:.0f}s), aborting\n')
                sys.stderr.flush()
                os._exit(EXIT_TIMEOUT)
            sw = swap_used_mb()
            if swap_mb and sw > baseline + swap_mb:
                sys.stderr.write(
                    f'GUARD{tag}: system swap {sw:.0f}MB exceeds baseline '
                    f'{baseline:.0f}MB + {swap_mb:.0f}MB, aborting\n')
                sys.stderr.flush()
                os._exit(EXIT_MEMORY)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    return t


def pin_blas():
    """One BLAS thread. With one solve per process there is nothing to gain
    from threading it, and oversubscription was measured at load 17 on 10
    cores, plus a thread-local workspace per fork."""
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(v, '1')
