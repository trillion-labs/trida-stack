from __future__ import annotations

from collections.abc import Callable


class InferenceCancelled(Exception):
    """Raised when cooperative inference cancellation reaches a safe barrier."""


CancellationCheck = Callable[[], bool] | None


def check_cancelled(should_cancel: CancellationCheck) -> None:
    if should_cancel is not None and should_cancel():
        raise InferenceCancelled
