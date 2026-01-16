#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/1/15
"""
import threading
from typing import TypeVar, Generic, Dict, Hashable, Callable, Tuple


__all__ = ['BaseDataStore']

T = TypeVar("T")


class BaseDataStore(Generic[T]):
    _instances: Dict[Tuple[type, Tuple], 'BaseDataStore'] = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        key = (cls, tuple(sorted(kwargs.items())))
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._cache: Dict[Hashable, T] = {}
        self._initialized = True

    def _get(self, key: Hashable, loader: Callable[[], T], cache_used: bool = True) -> T:
        if cache_used and key in self._cache:
            return self._cache[key]

        data = loader()

        if cache_used:
            self._cache[key] = data

        return data

    def clear_cache(self) -> None:
        self._cache.clear()
