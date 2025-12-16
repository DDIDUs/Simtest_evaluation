import contextlib
import faulthandler
import io
import itertools
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
import json
import numpy as np
from multiprocessing import Value
from typing import Dict, List, Tuple, Union, Optional

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
# pass@k estimator (unchanged)
# -----------------------------
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    def estimator(n: int, c: int, k: int) -> float:
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
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
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
    # Save original functions
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

    # NOTE: local list only (avoid Manager deadlocks under OOM)
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

    def safe_system(command):
        if "kill" in command or "killall" in command:
            return 0
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        try:
            if isinstance(command, (list, tuple)):
                joined = " ".join(map(str, command))
            else:
                joined = str(command)
            if "kill" in joined or "killall" in joined:
                return 0
        except Exception:
            pass
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        try:
            if isinstance(command, (list, tuple)) and command and command[0] == "ps":
                return b""
            if isinstance(command, str) and command.strip().startswith("ps"):
                return b""
        except Exception:
            pass
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        # args[0] is usually list/tuple or string
        cmd0 = args[0] if args else ""
        try:
            if isinstance(cmd0, (list, tuple)):
                joined = " ".join(map(str, cmd0))
            else:
                joined = str(cmd0)
            if "kill" in joined or "killall" in joined:
                return subprocess.CompletedProcess(args, 0, b"", b"")
        except Exception:
            pass
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs["preexec_fn"] = os.setsid
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            except subprocess.TimeoutExpired:
                return None, None

        def kill(self):
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        if "kill" in command or "killall" in command:
            return os.popen("echo Intercepted")
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        # disable exec* entirely
        return None

    # Override
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
        # best-effort cleanup for children
        for pid in child_pids:
            try:
                original_kill(pid, signal.SIGTERM)
            except Exception:
                pass

        # restore
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


def reliability_guard(max_as_limit, max_data_limit, max_stack_limit):
    os.environ["TZ"] = "UTC"
    time.tzset()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # IMPORTANT: you intentionally relaxed RLIMIT_AS to avoid mmap failures
    # Keep DATA + STACK limits
    if max_as_limit and max_data_limit and max_stack_limit:
        import resource

        max_data = int(max_data_limit * 1024 * 1024)
        max_stack = int(max_stack_limit * 1024 * 1024)

        try:
            resource.setrlimit(resource.RLIMIT_DATA, (max_data, max_data))
            if platform.uname().system != "Darwin":
                resource.setrlimit(resource.RLIMIT_STACK, (max_stack, max_stack))
        except Exception:
            pass

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    try:
        import matplotlib.pyplot as plt  # noqa
        plt.close("all")
    except Exception:
        pass


# -----------------------------
# Execution Logic (FIXED)
# -----------------------------
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
    stat_code = _UNKNOWN
    details_dict: Dict[str, str] = {}
    runtime_val = 0.0
    per_test_times_dict: Dict[str, float] = {}

    with safe_environment(), create_tempdir():
        import builtins

        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

        # IMPORTANT: run as a "real module", not a bare dict
        module_name = "__test__"
        new_module = types.ModuleType(module_name)

        # Make Flask (and others) able to compute root_path
        # Option A: virtual filename (works for most cases)
        # Option B: write to a real file. We'll do B for maximum compatibility.
        src_path = os.path.join(os.getcwd(), f"{module_name}.py")

        full_code = code + "\n\n" + test_code
        try:
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(full_code)
        except Exception:
            # fallback if FS fails; keep a synthetic filename
            src_path = f"{module_name}.py"

        new_module.__dict__.update(
            {
                "__builtins__": builtins,
                "__name__": module_name,
                "__file__": src_path,
                "__package__": None,
            }
        )
        sys.modules[module_name] = new_module

        class TimeTrackingTestResult(unittest.TestResult):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.test_times: Dict[str, float] = {}
                self._current_start_time = 0.0

            def startTest(self, test):
                self._current_start_time = time.time()
                super().startTest(test)

            def stopTest(self, test):
                elapsed = time.time() - self._current_start_time
                self.test_times[test.id().split(".")[-1]] = elapsed
                super().stopTest(test)

        try:
            with swallow_io():
                # Compile with a meaningful filename (critical for frameworks)
                compiled = compile(full_code, src_path, "exec")
                exec(compiled, new_module.__dict__)

                # Locate TestCases
                TestCases = getattr(new_module, "TestCases", None)
                if not TestCases:
                    for _, val in new_module.__dict__.items():
                        if isinstance(val, type) and issubclass(val, unittest.TestCase):
                            TestCases = val
                            break
                if not TestCases:
                    raise ValueError("No TestCases class found")

                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = TimeTrackingTestResult()

                start_time = time.time()
                with time_limit(timeout):
                    suite.run(test_result)
                runtime_val = time.time() - start_time

                per_test_times_dict.update(test_result.test_times)

                issues = test_result.failures + test_result.errors
                if issues:
                    for test, trace in issues:
                        details_dict[test.id().split(".")[-1]] = trace
                    stat_code = _FAILED
                else:
                    stat_code = _SUCCESS

        except TimeoutException as e:
            stat_code = _TIMEOUT
            details_dict["ALL"] = str(e)
        except BaseException as e:
            stat_code = _FAILED
            details_dict["ALL"] = str(e)

        # Write results to file (best-effort)
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stat": stat_code,
                        "details": details_dict,
                        "runtime": runtime_val,
                        "per_test_times": per_test_times_dict,
                    },
                    f,
                )
        except Exception:
            pass


def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float = 30 * 1024,
    max_data_limit: float = 30 * 1024,
    max_stack_limit: float = 10,
    min_time_limit: float = 1,
    gt_time_limit: float = 60,
) -> Tuple[str, Dict, float, Dict[str, float]]:
    # Official-like: per-task timeout (suite-level)
    base_timeout = float(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT))
    timeout = max(base_timeout, gt_time_limit, min_time_limit) + 1.0

    # file-based IPC (avoid Manager deadlocks under OOM)
    fd, result_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    p = multiprocessing.Process(
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

    killed = False
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
        killed = True
    if p.is_alive():
        p.kill()
        time.sleep(0.1)
        killed = True

    # defaults
    stat_res = FAIL
    details_dict: Dict = {}
    runtime_val = 0.0
    per_test_times_dict: Dict[str, float] = {}

    try:
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            stat_code = res_data.get("stat", _FAILED)

            if stat_code == _TIMEOUT:
                stat_res = TIMEOUT
            else:
                stat_res = _mapping.get(stat_code, FAIL) or TIMEOUT

            details_dict = res_data.get("details", {}) or {}
            runtime_val = float(res_data.get("runtime", 0.0) or 0.0)
            per_test_times_dict = res_data.get("per_test_times", {}) or {}
        else:
            if killed:
                stat_res = TIMEOUT
                details_dict = {"ALL": "Execution timed out (process killed)."}
            else:
                stat_res = FAIL
                details_dict = {"ALL": "Execution failed (crash/OOM; no result file written)."}
    except Exception as e:
        stat_res = FAIL
        details_dict = {"ALL": f"Execution failed (result read error): {str(e)}"}
    finally:
        try:
            if os.path.exists(result_file):
                os.remove(result_file)
        except Exception:
            pass

    # Guard: PASS but has details -> FAIL (BigCodeBench style)
    if stat_res == PASS and details_dict:
        stat_res = FAIL

    return stat_res, details_dict, runtime_val, per_test_times_dict