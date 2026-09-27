import time, random
from typing import Callable, Any

def execute_with_exponential_backoff(operation: Callable[[], Any], max_attempts: int=4, initial_interval: float=0.05, backoff_factor: float=2.0, max_interval: float=0.5) -> Any:
    attempt = 1
    current_interval = initial_interval
    while attempt <= max_attempts:
        try:
            return operation()
        except Exception as e:
            if attempt == max_attempts:
                raise e
            jitter = random.uniform(0.005, 0.02)
            sleep_time = min(max_interval, current_interval) + jitter
            time.sleep(sleep_time)
            current_interval *= backoff_factor
            attempt += 1

def execute_with_timeout(operation: Callable[[], Any], timeout_seconds: float=0.2) -> Any:
    start_time = time.time()
    res = operation()
    elapsed = time.time() - start_time
    if elapsed > timeout_seconds:
        raise TimeoutError(f'Operation exceeded per-node timeout deadline of {timeout_seconds}s (took {elapsed:.3f}s)')
    return res
if __name__ == '__main__':
    calls = [0]

    def flaky_api():
        calls[0] += 1
        if calls[0] < 3:
            raise ConnectionResetError(f'Simulated network glitch (Call #{calls[0]})')
        return {'status': 'SUCCESS', 'calls': calls[0]}
    result = execute_with_exponential_backoff(flaky_api)
    print('Result:', result)