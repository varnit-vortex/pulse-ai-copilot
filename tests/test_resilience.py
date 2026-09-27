import pytest
from resilience_retries import execute_with_exponential_backoff, execute_with_timeout

def test_exponential_backoff_recovery():
    calls = 0

    def flaky_fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionResetError('Transient network glitch')
        return 'SUCCESS'
    res = execute_with_exponential_backoff(flaky_fn)
    assert res == 'SUCCESS'
    assert calls == 3

def test_timeout_enforcement():
    import time

    def slow_fn():
        time.sleep(0.3)
        return 'DONE'
    with pytest.raises(TimeoutError):
        execute_with_timeout(slow_fn, timeout_seconds=0.1)