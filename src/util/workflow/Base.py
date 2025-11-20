#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Task-based data processing workflow with context-based data passing
  @Author Chris
  @Date 2025/11/19
"""
import abc
import copy
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from tqdm import tqdm

SRC_FILE_PATH_KEY = "src_file_path"
DST_FILE_PATH_KEY = "dst_file_path"
SRC_DIR_PATH_KEY = "src_dir_path"
DST_DIR_PATH_KEY = "dst_dir_path"
SRC_FILE_PATHS_KEY = "src_file_paths"

TIFF_DIR_PATH_KEY = "tiff_dir_path"
MERGED_DIR_PATH_KEY = "merged_dir_path"
REF_GRID_PATH_KEY = "ref_grid_path"
RESOLUTION_CONFIGS_KEY = "resolution_configs"

DATA_KEY = "data"
GAP_VALUE_KEY = "gap_value"
SCALE_FACTOR_KEY = "scale_factor"
TRANSFORM_KEY = "transform"
PROJECTION_KEY = "projection"
X_SIZE_KEY = "x_size"
Y_SIZE_KEY = "y_size"


@dataclass
class Context:
    state: Dict[str, Any] = field(default_factory=dict)
    local_state: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.local_state:
            return self.local_state[key]
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.local_state[key] = value

    def get_global(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set_global(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_local(self, key: str, default: Any = None) -> Any:
        return self.local_state.get(key, default)

    def set_local(self, key: str, value: Any) -> None:
        self.local_state[key] = value

    def clear_local(self) -> None:
        self.local_state.clear()

    def copy(self) -> 'Context':
        new_context = Context()
        new_context.state = copy.deepcopy(self.state)
        new_context.local_state = copy.deepcopy(self.local_state)

        return new_context

    def global_copy(self) -> 'Context':
        new_context = Context()
        new_context.state = copy.deepcopy(self.state)

        return new_context


class Executable(ABC):

    @abc.abstractmethod
    def execute(self, context: Context) -> Context:
        pass


class BaseTask(Executable, ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def run(self, context: Context) -> Context:
        return self.execute(context)

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"


class BaseReader(BaseTask, ABC):
    pass


class BaseWriter(BaseTask, ABC):
    pass


class Batchable(Executable, ABC):

    @abc.abstractmethod
    def build_batch_context(self, context: Context) -> List[Context]:
        pass

    def run_batch(self, context: Context) -> Context:
        batch_contexts = self.build_batch_context(context)
        batch_results = []

        for batch_context in tqdm(batch_contexts, desc=f"Executing Batch {getattr(self, 'name', self.__class__.__name__)}"):
            batch_result = self.execute(batch_context)
            batch_results.append(batch_result)

        return self.collect(context, batch_results)

    def collect(self, context: Context, batch_results: List[Context]) -> Context:
        return context


class Job(BaseTask):
    def __init__(self, name: Optional[str] = None, *tasks: BaseTask):
        super().__init__(name)
        self.tasks = list(tasks)

    def add(self, *tasks: BaseTask) -> 'Job':
        for task in tasks:
            if isinstance(task, list):
                self.tasks.extend(task)
            else:
                self.tasks.append(task)
        return self

    def execute(self, context: Context) -> Context:
        for task in self.tasks:
            context = task.run(context)
        return context

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name}, tasks={len(self.tasks)})>"


class BatchJob(Job, Batchable, ABC):

    def run(self, context: Context) -> Context:
        return self.run_batch(context)

