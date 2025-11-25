#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Base class for Workflow
  @Author Chris
  @Date 2025/11/19
"""
import abc
import copy
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union, Iterable

from tqdm import tqdm

from util.workflow.WorkflowConstant import (
    IS_SKIP_KEY,
)

__all__ = [
    'Context',
    'Executable',
    'BaseTask',
    'Batchable',
    'Job',
    'BatchJob',
    'BaseFilter',
]


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

    def has(self, key: str) -> bool:
        return key in self.local_state or key in self.state

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
    required_context_keys: Tuple[str, ...] = ()

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def run(self, context: Context) -> Context:
        self._validate_context(context)
        return self.execute(context)

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"

    def _validate_context(self, context: Context) -> None:
        missing = [key for key in self.required_context_keys if not context.has(key)]
        if missing:
            raise KeyError(f"{self.name} missing required context keys :{missing}")


class Batchable(Executable, ABC):

    @abc.abstractmethod
    def build_batch_context(self, context: Context) -> List[Context]:
        pass

    def _run_batch(self, context: Context) -> Context:
        batch_contexts = self.build_batch_context(context)
        batch_results = []

        for batch_context in tqdm(batch_contexts,
                                  desc=f"Executing Batch {getattr(self, 'name', self.__class__.__name__)}"):
            if batch_context.get_local(IS_SKIP_KEY, False):
                batch_context.set(IS_SKIP_KEY, False)
                continue
            batch_result = self.execute(batch_context)
            batch_results.append(batch_result)

        return self.collect(context, batch_results)

    def collect(self, context: Context, batch_results: List[Context]) -> Context:
        return context


class Job(BaseTask):
    def __init__(self, name: Optional[str] = None, *tasks: Union['BaseTask', Iterable['BaseTask']]):
        super().__init__(name)
        self.tasks: List[BaseTask] = []
        if tasks:
            self.add(*tasks)

    def add(self, *tasks: Union['BaseTask', Iterable['BaseTask']]) -> 'Job':
        for task in tasks:
            if isinstance(task, list):
                self.tasks.extend(task)
            else:
                self.tasks.append(task)
        return self

    def execute(self, context: Context) -> Context:
        for task in self.tasks:
            if context.get_local(IS_SKIP_KEY, False):
                context.set(IS_SKIP_KEY, False)
                break
            context = task.run(context)
        return context

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name}, tasks={len(self.tasks)})>"


class BatchJob(Job, Batchable, ABC):

    def run(self, context: Context) -> Context:
        return self._run_batch(context)


class BaseFilter(BaseTask, ABC):

    @abc.abstractmethod
    def filter(self, context: Context) -> bool:
        pass

    def execute(self, context: Context) -> Context:
        is_skip = self.filter(context)
        context.set(IS_SKIP_KEY, is_skip)

        return context
