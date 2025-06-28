import signal
import time
from contextlib import contextmanager
from functools import wraps
from builtins import TimeoutError


class TimerResult:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.end_time = None

    @property
    def seconds(self):
        return self.end_time - self.start_time

    @property
    def millis(self):
        return self.seconds * 1000


@contextmanager
def time_execution_scope():
    """Usage:

    with time_execution_scope() as timer_result:
        do_some_operation()

    print(
        'do_some_operation took {timer_result.millis} milliseconds'.format(
            timer_result=timer_result
        )
    )
    """

    timer_result = TimerResult()
    yield timer_result
    timer_result.end_time = time.perf_counter()


def timeout(seconds_before_timeout):
    def decorate(f):
        # Just do without the timeout on Windows: see
        # https://github.com/natekupp/ffx/issues/17
        if not hasattr(signal, "SIGALRM"):
            return f

        def handler(signum, frame):
            raise TimeoutError()

        @wraps(f)
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds_before_timeout)
            try:
                result = f(*args, **kwargs)
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
            return result

        return new_f

    return decorate
