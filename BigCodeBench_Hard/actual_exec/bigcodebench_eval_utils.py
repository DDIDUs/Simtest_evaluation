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
import numpy as np
from multiprocessing import Value, Manager
from typing import Dict, List, Tuple, Union, Optional

# Constants
PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}

TIMEOUT_LIMIT = 240.0 # Default fallback, usually overridden by env or arg

# --- Pass@k Estimator ---

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

# --- Context Managers for Safety ---

class TimeoutException(Exception):
    pass

class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

@contextlib.contextmanager
def swallow_subprocess_output():
    """Context manager to swallow stdout and stderr for subprocesses."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
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
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
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
    manager = multiprocessing.Manager()
    child_pids = manager.list()

    def safe_kill(pid, sig):
        try:
            pgid = os.getpgid(pid)
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
            else:
                pass # Prevent killing other processes, silent fail or log if needed

        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
            original_killpg(pgid, sig)
        else:
            pass

    def safe_system(command):
        if 'kill' in command or 'killall' in command:
            return 0
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        if 'kill' in command or 'killall' in command:
            return 0
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        if 'ps' in command:
            return b""
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        if 'kill' in args[0] or 'killall' in args[0]:
            return subprocess.CompletedProcess(args, 0, b'', b'')
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs['preexec_fn'] = os.setsid
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
        if 'kill' in command or 'killall' in command:
            return os.popen('echo Intercepted')
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        pass # Disable exec

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
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                # Cleanup logic
            except Exception:
                pass

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
    import os
    import time
    
    os.environ['TZ'] = 'UTC'
    time.tzset()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

    if max_as_limit and max_data_limit and max_stack_limit:
        import resource
        max_as_limit = int(max_as_limit * 1024 * 1024)
        max_data_limit = int(max_data_limit * 1024 * 1024)
        max_stack_limit = int(max_stack_limit * 1024 * 1024)

        try:
            resource.setrlimit(resource.RLIMIT_AS, (max_as_limit, max_as_limit))
            resource.setrlimit(resource.RLIMIT_DATA, (max_data_limit, max_data_limit))
            if not platform.uname().system == "Darwin":
                 resource.setrlimit(resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit))
        except Exception:
            pass # resource limit might fail on some systems/containers

    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None

    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass

# --- Execution Logic ---

def unsafe_execute(
    entry_point: str,
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    stat,
    details,
    runtime,
    per_test_times
):
    with safe_environment(), create_tempdir():
        import shutil
        import builtins
        
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)
        
        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            #'__package__': None,
            #'__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        try:
            full_code = code + "\n" + test_code
            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                
                # Check for TestCases class which is standard in BigCodeBench
                TestCases = getattr(new_module, 'TestCases', None)
                if not TestCases:
                    # Fallback to finding any unittest.TestCase
                    for name, val in new_module.__dict__.items():
                        if isinstance(val, type) and issubclass(val, unittest.TestCase):
                            TestCases = val
                            break
                            
                if not TestCases:
                    raise ValueError("No TestCases class found")

                class TimeTrackingTestResult(unittest.TestResult):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.test_times = {}
                        self._current_start_time = 0

                    def startTest(self, test):
                        self._current_start_time = time.time()
                        super().startTest(test)

                    def stopTest(self, test):
                        elapsed = time.time() - self._current_start_time
                        self.test_times[test.id().split(".")[-1]] = elapsed
                        super().stopTest(test)

                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = TimeTrackingTestResult()
                
                start_time = time.time()
                with time_limit(timeout):
                    suite.run(test_result)
                runtime.value = time.time() - start_time
                
                # Copy per-test times
                for k, v in test_result.test_times.items():
                    per_test_times[k] = v
            
            issues = test_result.failures + test_result.errors
            if issues:
                for test, trace in issues:
                    details[test.id().split(".")[-1]] = trace
                stat.value = _FAILED
            else:
                stat.value = _SUCCESS

        except BaseException as e:
            details["ALL"] = str(e)
            stat.value = _FAILED

        # Restore
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float = 30 * 1024, 
    max_data_limit: float = 30 * 1024,
    max_stack_limit: float = 10,
    min_time_limit: float = 1,
    gt_time_limit: float = 60
) -> Tuple[str, Dict, float, Dict[str, float]]:
    
    timeout = max(float(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT)), gt_time_limit) + 1

    stat = Value("i", _UNKNOWN)
    runtime = Value("d", 0.0)
    manager = Manager()
    details = manager.dict()
    per_test_times = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            entry_point,
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            stat,
            details,
            runtime,
            per_test_times
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat_res = _mapping[stat.value]
    details_dict = dict(details)
    runtime_val = runtime.value
    per_test_times_dict = dict(per_test_times)

    if not stat_res:
         stat_res = TIMEOUT
    
    if stat_res == PASS:
        if details_dict:
            stat_res = FAIL

    return stat_res, details_dict, runtime_val, per_test_times_dict
