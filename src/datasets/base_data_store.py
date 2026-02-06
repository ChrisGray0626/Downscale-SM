#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/1/15
"""
import os
import threading
from typing import TypeVar, Generic, Dict, Hashable, Callable, Tuple

import numpy as np

from constants import TIFF_SUFFIX
from utils.date_util import list_date_from_dir
from utils.raster_util import read_tiff_data

__all__ = [
    'BaseDataStore',
    'BaseTiffStore',
]

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


class BaseTiffStore(BaseDataStore[np.ndarray]):

    def __init__(self, base_dir: str, resolution: str):
        super().__init__()
        self.base_dir = base_dir
        self.resolution = resolution

    def get(self, date: str, cache_used: bool = True) -> np.ndarray:
        return self._get((date,), lambda: self._load(date), cache_used=cache_used)

    def _load(self, date: str) -> np.ndarray:
        file_path = os.path.join(self.base_dir, self.resolution, f"{date}{TIFF_SUFFIX}")
        return read_tiff_data(file_path).astype(np.float32)

    def list_date(self):
        return list_date_from_dir(os.path.join(self.base_dir, self.resolution))
