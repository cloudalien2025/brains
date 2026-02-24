from __future__ import annotations

from contextlib import contextmanager


def get_engine():
    return None


@contextmanager
def db_session():
    yield None
