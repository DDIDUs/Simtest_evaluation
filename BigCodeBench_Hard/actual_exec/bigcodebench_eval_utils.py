from __future__ import annotations

import contextlib
import faulthandler
import io
import itertools
import json
import multiprocessing as mp
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from typing import Dict, List, Tuple, Union, Optional

import numpy as np


# -----------------------------
# Constants / Status Mapping
# -----------------------------
PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}

TIMEOUT_LIMIT = 240.0  # Default fallback, can be overridden via env BIGCODEBENCH_TIMEOUT_PER_TASK


# -----------------------------
# pass@k estimator (official-like)
# -----------------------------
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    def estimator(n: int, c: int, k: int) -> float:
        # 1 - C(n-c, k) / C(n, k)
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


# -----------------------------
# Context Managers / Guards
# -----------------------------
class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that raises if read is attempted (prevents leakage)."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_subprocess_output():
    """Force subprocess stdout/stderr to PIPE unless caller explicitly captures."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if kwargs.get("capture_output"):
            kwargs.pop("stdout", None)
            kwargs.pop("stderr", None)
        else:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if kwargs.get("capture_output"):
            kwargs.pop("stdout", None)
            kwargs.pop("stderr", None)
        else:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                with swallow_subprocess_output():
                    yield


@contextlib.contextmanager
def time_limit(seconds: float):
    """SIGALRM-based wall-clock timeout (Unix)."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def chdir(root: str):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def safe_environment():
    """
    Light-weight guardrail: prevent kill/exec abuse; track child PIDs started via subprocess.Popen.
    (Not a complete sandbox; mirrors typical BigCodeBench approach.)
    """
    original_kill = os.kill
    original_killpg = os.killpg
    original_system = os.system
    original_subprocess_call = subprocess.call
    original_subprocess_check_output = subprocess.check_output
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    original_os_popen = os.popen
    original_os_execv = os.execv
    original_os_execvp = os.execvp
    original_os_execvpe = os.execvpe

    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    child_pids: List[int] = []

    def safe_kill(pid, sig):
        try:
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        try:
            if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
                original_killpg(pgid, sig)
        except ProcessLookupError:
            pass

    def _contains_kill(cmd: str) -> bool:
        return ("kill" in cmd) or ("killall" in cmd)

    def safe_system(command):
        if _contains_kill(str(command)):
            return 0
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        try:
            joined = " ".join(map(str, command)) if isinstance(command, (list, tuple)) else str(command)
            if _contains_kill(joined):
                return 0
        except Exception:
            pass
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        try:
            if isinstance(command, (list, tuple)) and command and str(command[0]) == "ps":
                return b""
            if isinstance(command, str) and command.strip().startswith("ps"):
                return b""
        except Exception:
            pass
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        cmd0 = args[0] if args else ""
        try:
            joined = " ".join(map(str, cmd0)) if isinstance(cmd0, (list, tuple)) else str(cmd0)
            if _contains_kill(joined):
                return subprocess.CompletedProcess(args, 0, b"", b"")
        except Exception:
            pass
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs["preexec_fn"] = os.setsid
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def kill(self):
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        if _contains_kill(str(command)):
            return os.popen("echo Intercepted")
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        return None  # disable exec*

    # Apply overrides
    os.kill = safe_kill
    os.killpg = safe_killpg
    os.system = safe_system
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_subprocess_check_output
    subprocess.run = safe_subprocess_run
    subprocess.Popen = SafePopen
    os.popen = safe_os_popen
    os.execv = safe_exec
    os.execvp = safe_exec
    os.execvpe = safe_exec

    try:
        yield
    finally:
        # best-effort terminate tracked subprocess children
        for pid in child_pids:
            try:
                original_kill(pid, signal.SIGTERM)
            except Exception:
                pass

        # Restore originals
        os.kill = original_kill
        os.killpg = original_killpg
        os.system = original_system
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_subprocess_check_output
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        os.popen = original_os_popen
        os.execv = original_os_execv
        os.execvp = original_os_execvp
        os.execvpe = original_os_execvpe


# -----------------------------
# TF / Thread / Allocator guard
# -----------------------------
def _set_ml_env_for_child():
    """
    Must run in the CHILD *before* importing TensorFlow / JAX / PyTorch etc.
    This is where most OOM mitigation comes from.
    """
    os.environ.setdefault("TZ", "UTC")
    try:
        time.tzset()
    except Exception:
        pass

    # Generic thread limiting (prevents massive thread pools / memory)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # TensorFlow: reduce noise and GPU grabs, reduce fragmentation.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # Disable oneDNN optimizations (can increase memory in some envs)
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    # If GPUs exist, do not pre-allocate all memory.
    # For TF2, this is largely controlled via API (set_memory_growth), but env helps on some builds.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Some TF builds support this allocator; can reduce fragmentation.
    # If unsupported, it is ignored.
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    # XLA can spike memory; disable by default.
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
    os.environ.setdefault("MPLBACKEND", "Agg")


def reliability_guard(max_as_limit_mb: float, max_data_limit_mb: float, max_stack_limit_mb: float):
    """
    BigCodeBench sets resource limits; but RLIMIT_AS often causes mmap failures with scipy/sklearn/tf.
    Here we keep DATA/STACK, and leave AS relaxed.
    """
    _set_ml_env_for_child()

    # Resource limits (child-only)
    if max_data_limit_mb and max_stack_limit_mb:
        try:
            import resource

            max_data = int(max_data_limit_mb * 1024 * 1024)
            max_stack = int(max_stack_limit_mb * 1024 * 1024)

            resource.setrlimit(resource.RLIMIT_DATA, (max_data, max_data))
            if platform.uname().system != "Darwin":
                resource.setrlimit(resource.RLIMIT_STACK, (max_stack, max_stack))
        except Exception:
            pass

    # reduce crash spew / dangerous exits
    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None

    # close matplotlib figures if imported
    try:
        import matplotlib.pyplot as plt  # noqa
        plt.close("all")
    except Exception:
        pass


# -----------------------------
# Execution Logic (child)
# -----------------------------
def _write_json_safely(path: str, payload: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def unsafe_execute(
    entry_point: str,
    code: str,
    test_code: str,
    result_file: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
):
    """
    Runs in CHILD process (spawn), executes code + test harness, produces json result file.
    """
    stat_code = _UNKNOWN
    details_dict: Dict[str, str] = {}
    runtime_val = 0.0
    per_test_times: Dict[str, float] = {}

    try:
        with safe_environment(), create_tempdir():
            # critical: set env + resource limits *before* any heavy imports happen via user code
            reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

            module_name = "__test__"
            mod = types.ModuleType(module_name)

            # write a real file so frameworks relying on __file__/inspect work better
            src_path = os.path.join(os.getcwd(), f"{module_name}.py")
            full_code = (code or "") + "\n\n" + (test_code or "")
            try:
                with open(src_path, "w", encoding="utf-8") as f:
                    f.write(full_code)
            except Exception:
                src_path = f"{module_name}.py"

            # module dict
            mod.__dict__.update(
                {
                    "__name__": module_name,
                    "__file__": src_path,
                    "__package__": None,
                }
            )
            sys.modules[module_name] = mod

            class TimeTrackingTestResult(unittest.TestResult):
                def __init__(self):
                    super().__init__()
                    self.test_times: Dict[str, float] = {}
                    self._t0 = 0.0

                def startTest(self, test):
                    self._t0 = time.time()
                    super().startTest(test)

                def stopTest(self, test):
                    self.test_times[test.id().split(".")[-1]] = time.time() - self._t0
                    super().stopTest(test)

            with swallow_io():
                try:
                    compiled = compile(full_code, src_path, "exec")
                    exec(compiled, mod.__dict__)
                except BaseException as e:
                    stat_code = _FAILED
                    details_dict["ALL"] = f"{type(e).__name__}: {e}"
                    _write_json_safely(result_file, {
                        "stat": stat_code,
                        "details": details_dict,
                        "runtime": runtime_val,
                        "per_test_times": per_test_times,
                    })
                    return

                # locate tests
                TestCases = mod.__dict__.get("TestCases")
                if not TestCases:
                    for _, v in mod.__dict__.items():
                        if isinstance(v, type) and issubclass(v, unittest.TestCase):
                            TestCases = v
                            break
                if not TestCases:
                    stat_code = _FAILED
                    details_dict["ALL"] = "No TestCases class found"
                    _write_json_safely(result_file, {
                        "stat": stat_code,
                        "details": details_dict,
                        "runtime": runtime_val,
                        "per_test_times": per_test_times,
                    })
                    return

                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                result = TimeTrackingTestResult()

                t0 = time.time()
                try:
                    with time_limit(timeout):
                        suite.run(result)
                    runtime_val = time.time() - t0
                except TimeoutException as e:
                    stat_code = _TIMEOUT
                    details_dict["ALL"] = str(e)
                    per_test_times = result.test_times
                    _write_json_safely(result_file, {
                        "stat": stat_code,
                        "details": details_dict,
                        "runtime": runtime_val,
                        "per_test_times": per_test_times,
                    })
                    return

                per_test_times = result.test_times
                issues = result.failures + result.errors
                if issues:
                    stat_code = _FAILED
                    for test, trace in issues:
                        details_dict[test.id().split(".")[-1]] = trace
                else:
                    stat_code = _SUCCESS

    except BaseException as e:
        # if the process is about to die (incl. some MemoryError paths), best effort file write
        stat_code = _FAILED
        details_dict["ALL"] = f"{type(e).__name__}: {e}"

    _write_json_safely(
        result_file,
        {
            "stat": stat_code,
            "details": details_dict,
            "runtime": runtime_val,
            "per_test_times": per_test_times,
        },
    )


# -----------------------------
# Public API (parent)
# -----------------------------
def _classify_exitcode(exitcode: Optional[int]) -> str:
    """
    multiprocessing.Process.exitcode
    - None: still running / not set
    - 0: normal exit
    - >0: python-level sys.exit(code) or unhandled exception sometimes becomes 1
    - <0: terminated by signal (-SIGNAL)
    """
    if exitcode is None:
        return "EXITCODE=None (unknown; still running or not reported)"
    if exitcode == 0:
        return "EXITCODE=0 (normal exit) but no result file (unexpected)"
    if exitcode > 0:
        # 1 is most common for unhandled exception if it bubbled to top
        return f"EXITCODE={exitcode} (non-zero exit; likely crash/exception)"
    # Signal-based
    sig = -exitcode
    # Common signals
    if sig == 9:
        return "SIGKILL(9): very likely OOM-kill or external kill -9"
    if sig == 15:
        return "SIGTERM(15): terminated (timeout/terminate or external SIGTERM)"
    if sig == 11:
        return "SIGSEGV(11): segmentation fault (native crash)"
    if sig == 6:
        return "SIGABRT(6): abort (native abort)"
    return f"Signal({sig}): terminated by signal"


def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float = 30 * 1024,   # MB
    max_data_limit: float = 30 * 1024, # MB
    max_stack_limit: float = 10,       # MB
    min_time_limit: float = 1,
    gt_time_limit: float = 60,
) -> Tuple[str, Dict, float, Dict[str, float]]:
    """
    BigCodeBench-compatible entrypoint.
    - Forces spawn start method
    - Uses file-based IPC
    - Adds exitcode-based failure categorization when result file is missing/empty
    """
    # Force spawn globally (safe to call repeatedly)
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    base_timeout = float(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT))
    timeout = max(base_timeout, float(gt_time_limit), float(min_time_limit)) + 1.0

    fd, result_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=unsafe_execute,
        args=(
            entry_point,
            code,
            test_code,
            result_file,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
        ),
    )

    p.start()
    p.join(timeout=timeout + 1.0)

    # capture exitcode as soon as possible after join
    exitcode = p.exitcode

    killed = False
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
        killed = True
    if p.is_alive():
        p.kill()
        time.sleep(0.1)
        killed = True

    # refresh exitcode after forced kill/terminate attempt
    if killed:
        exitcode = p.exitcode

    stat_res = FAIL
    details_dict: Dict = {}
    runtime_val = 0.0
    per_test_times: Dict[str, float] = {}

    try:
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "r", encoding="utf-8") as f:
                res = json.load(f)

            stat_code = res.get("stat", _FAILED)
            if stat_code == _TIMEOUT:
                stat_res = TIMEOUT
            else:
                stat_res = _mapping.get(stat_code, FAIL) or FAIL

            details_dict = res.get("details", {}) or {}
            runtime_val = float(res.get("runtime", 0.0) or 0.0)
            per_test_times = res.get("per_test_times", {}) or {}

            # Optional: if child returned "FAILED" but didn't include details, enrich with exitcode
            if stat_res == FAIL and (not details_dict):
                details_dict = {"ALL": f"Execution failed (no details). { _classify_exitcode(exitcode) }"}

        else:
            # result file missing/empty
            if killed:
                stat_res = TIMEOUT
                details_dict = {
                    "ALL": f"Execution timed out (process killed). { _classify_exitcode(exitcode) }"
                }
            else:
                stat_res = FAIL
                details_dict = {
                    "ALL": f"Execution failed (no result file written). { _classify_exitcode(exitcode) }"
                }

    except json.JSONDecodeError as e:
        # result file exists but corrupted/partial (often crash during write)
        stat_res = FAIL
        details_dict = {
            "ALL": f"Execution failed (result JSON decode error): {type(e).__name__}: {e}. { _classify_exitcode(exitcode) }"
        }
    except Exception as e:
        stat_res = FAIL
        details_dict = {
            "ALL": f"Execution failed (result read error): {type(e).__name__}: {e}. { _classify_exitcode(exitcode) }"
        }
    finally:
        try:
            if os.path.exists(result_file):
                os.remove(result_file)
        except Exception:
            pass

    # BigCodeBench-like: if PASS but details exist => FAIL
    if stat_res == PASS and details_dict:
        stat_res = FAIL

    return stat_res, details_dict, runtime_val, per_test_times