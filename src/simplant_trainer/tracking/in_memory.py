from collections import deque

from simplant_trainer.value_objects.metrics import EpochMetrics


class InMemoryTracker:
    def __init__(self, max_entries: int = 1000) -> None:
        self._buffer: deque[EpochMetrics] = deque(maxlen=max_entries)

    def record(self, metrics: EpochMetrics) -> None:
        self._buffer.append(metrics)

    def history(self, last_n: int = 0) -> list[EpochMetrics]:
        if last_n <= 0:
            return list(self._buffer)
        else:
            return list(self._buffer)[-last_n:]

    def flush(self) -> None:
        pass
