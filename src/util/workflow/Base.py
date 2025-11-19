#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Task-based data processing workflow with context-based data passing
  @Author Chris
  @Date 2025/11/19
"""
import abc
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

TIFF_DIR_PATH_KEY = "tiff_dir_path"
MERGED_DIR_PATH_KEY = "merged_dir_path"
OUTPUT_DIR_PATH_KEY = "output_dir_path"
STANDARD_GRID_PATH_KEY = "standard_grid_path"

INPUT_PATH_KEY = "input_path"
OUTPUT_PATH_KEY = "output_path"
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


class BaseTask(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def run(self, context: Context) -> Context:
        return self.execute(context)

    @abc.abstractmethod
    def execute(self, context: Context) -> Context:
        raise NotImplementedError("Subclasses must implement the execute method.")

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"


class BaseReader(BaseTask, ABC):
    pass


class BaseWriter(BaseTask, ABC):
    pass


class Batchable(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_results: List[Context] = []

    def run_batch_mode(self, context: Context) -> Context:
        batch_contexts = self.build_batch_context(context)

        for batch_context in batch_contexts:
            batch_result = self.execute(batch_context)
            self.batch_results.append(batch_result)

        return self.collect(context)

    @abc.abstractmethod
    def build_batch_context(self, context: Context) -> List[Context]:
        pass

    @abc.abstractmethod
    def execute(self, batch_context: Context) -> Context:
        pass

    def collect(self, context: Context) -> Context:
        return context


class BaseJob(BaseTask):
    def __init__(self, name: Optional[str] = None, tasks: Optional[List[BaseTask]] = None):
        super().__init__(name)
        self.tasks = tasks or []

    def add(self, task: BaseTask) -> 'BaseJob':
        self.tasks.append(task)
        return self

    def execute(self, context: Context) -> Context:
        for task in self.tasks:
            context = task.run(context)
        return context

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name}, tasks={len(self.tasks)})>"
