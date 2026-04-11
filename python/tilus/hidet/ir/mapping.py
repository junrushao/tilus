# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import itertools
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.expr import Expr, convert
from tilus.hidet.ir.node import Node
from tilus.hidet.utils import gcd, prod

Int = Union[Expr, int]


def is_atom(expr):
    from tilus.hidet.ir import Constant, Var

    return isinstance(expr, (Constant, Var))


def var(hint):
    from tilus.hidet import ir

    return ir.var(hint)


def strides_from_ranks(shape: Sequence[Int], ranks: Sequence[int]) -> List[Int]:
    if any(isinstance(v, int) and v < 0 for v in shape):
        raise ValueError("Shape must be non-negative, got {}".format(shape))
    if len(set(ranks)) != len(ranks):
        raise ValueError("Duplicated ranks: {}".format(ranks))
    if any(isinstance(v, int) and v < 0 or v >= len(shape) for v in ranks):
        raise ValueError("Ranks {} out of bound for shape {}".format(ranks, shape))
    if len(ranks) != len(shape):
        raise ValueError("Ranks must have the same length as shape, got shape {} and ranks {}".format(shape, ranks))
    strides: List[Optional[Int]] = [None] * len(shape)
    acc = 1
    for i in reversed(range(len(shape))):
        dim = ranks.index(i)
        strides[dim] = acc
        acc = acc * shape[dim]
    return strides


@py_class
class TaskMapping(Node):
    registered: ClassVar[list] = []

    num_workers: Any = None
    task_shape: Any = None
    worker2task: Any = None

    def __post_init__(self):
        from tilus.hidet.ir.tools import simplify

        if self.num_workers is not None:
            self.num_workers = simplify(self.num_workers)
        if self.task_shape is not None:
            self.task_shape = tuple(simplify(v) for v in self.task_shape)

    def __call__(self, w: Int) -> List[Tuple[Int, ...]]:
        return self.worker2task(w)

    def __mul__(self, other) -> TaskMapping:
        return ComposedTaskMapping(outer=self, inner=other)

    def __getitem__(self, w: Int) -> List[Tuple[Int, ...]]:
        return self.worker2task(w)

    def on(self, w: Int, bind_tuple=False):
        from tilus.hidet.lang.constructs.loops import TaskMappingLoopIterable

        return TaskMappingLoopIterable(self, w, bind_tuple)

    def map(self, w: Int) -> List[Int]:
        return self.single_task_of(w)

    def single_task_of(self, w: Int) -> List[Int]:
        tasks = self.worker2task(w)
        if len(tasks) != 1:
            raise ValueError("Expect to have a single task, but got {} tasks".format(len(tasks)))
        return list(tasks[0])

    @staticmethod
    def row_major(task_shape: Sequence[int]):
        return SpatialTaskMapping(task_shape=task_shape, ranks=list(range(len(task_shape))))

    @staticmethod
    def column_major(task_shape: Sequence[int]):
        return SpatialTaskMapping(task_shape=task_shape, ranks=list(reversed(range(len(task_shape)))))

    @staticmethod
    def full_layout(task_shape: Sequence[int]):
        return row_repeat(*task_shape)

    def projection(self, dim2value: Dict[int, Int]) -> TaskMapping:
        return ProjectedTaskMapping(base=self, dim2value=dim2value)

    # chain api
    def spatial(self, *task_shape, ranks: List[int] = None) -> TaskMapping:
        return self * spatial_map(task_shape, ranks)

    def repeat(self, *task_shape, ranks: List[int] = None, attrs: Optional[str] = None) -> TaskMapping:
        return self * repeat_map(task_shape, ranks, attrs)


@py_class
class RepeatTaskMapping(TaskMapping):
    ranks: Any = None
    strides: Any = None
    attrs: Any = None

    def __post_init__(self):
        from tilus.hidet.ir.stmt import ForStmtAttr
        from tilus.hidet.ir.tools import simplify

        self.num_workers = 1
        if self.ranks is not None:
            self.ranks = list(self.ranks)
        if self.task_shape is not None:
            self.task_shape = tuple(self.task_shape)
        if self.strides is None and self.task_shape is not None and self.ranks is not None:
            self.strides = [simplify(v) for v in strides_from_ranks(self.task_shape, self.ranks)]
        if self.attrs is not None:
            self.attrs = list(self.attrs)
        self.worker2task = self._worker2task
        super().__post_init__()

    # noinspection PyUnusedLocal
    def _worker2task(self, w: Int) -> List[Tuple[Int]]:  # pylint: disable=unused-argument
        def key_func(task: Tuple[Int, ...]) -> Int:
            global_index = sum(a * b for a, b in zip(task, self.strides))
            return global_index

        ranges = [range(int(s)) for s in self.task_shape]
        tasks = list(tuple(task) for task in itertools.product(*ranges))
        return list(sorted(tasks, key=key_func))


@py_class
class SpatialTaskMapping(TaskMapping):
    ranks: Any = None
    strides: Any = None

    def __post_init__(self):
        from tilus.hidet.ir.tools import simplify

        if self.task_shape is not None:
            self.num_workers = prod(self.task_shape)
            self.task_shape = tuple(self.task_shape)
        if self.ranks is not None:
            self.ranks = list(self.ranks)
        if self.strides is None and self.task_shape is not None and self.ranks is not None:
            self.strides = [simplify(v) for v in strides_from_ranks(self.task_shape, self.ranks)]
        self.worker2task = self._worker2task
        assert self.task_shape is None or self.ranks is None or len(self.task_shape) == len(self.ranks)
        super().__post_init__()

    def _worker2task(self, w: Int) -> List[Tuple[Int, ...]]:
        task = []
        for mod, b in zip(self.task_shape, self.strides):
            task.append((w // b) % mod)
        return [tuple(task)]


@py_class
class ProjectedTaskMapping(TaskMapping):
    base: Any = None
    dim2value: Any = None

    def __post_init__(self):
        assert all(int(v) == 0 for v in self.dim2value.values())
        self.task_shape = tuple(
            self.base.task_shape[i] if i not in self.dim2value else 1 for i in range(len(self.base.task_shape))
        )
        self.num_workers = self.base.num_workers
        self.worker2task = self._worker2task
        super().__post_init__()

    def _worker2task(self, w: Int) -> List[Tuple[Int, ...]]:
        rank = len(self.task_shape)
        projected_tasks = []
        for task in self.base(w):
            projected_tasks.append(tuple(self.dim2value[i] if i in self.dim2value else task[i] for i in range(rank)))
        return projected_tasks


@py_class
class ComposedTaskMapping(TaskMapping):
    outer: Any = None
    inner: Any = None

    def __post_init__(self):
        self.num_workers = self.outer.num_workers * self.inner.num_workers
        self.task_shape = tuple(a * b for a, b in zip(self.outer.task_shape, self.inner.task_shape))
        self.worker2task = self._worker2task
        assert len(self.outer.task_shape) == len(self.inner.task_shape)
        super().__post_init__()

    def _worker2task(self, worker_index: Int) -> List[Tuple[Int, ...]]:
        outer_worker_index = worker_index // self.inner.num_workers
        inner_worker_index = worker_index % self.inner.num_workers
        outer_tasks = self.outer.worker2task(outer_worker_index)
        inner_tasks = self.inner.worker2task(inner_worker_index)
        tasks = []
        for outer_task in outer_tasks:
            for inner_task in inner_tasks:
                task = tuple(a * self.inner.task_shape[i] + b for i, (a, b) in enumerate(zip(outer_task, inner_task)))
                tasks.append(task)
        return tasks


def spatial_map(task_shape: Sequence[Int], ranks: Optional[Sequence[int]] = None):
    from tilus.hidet.ir.tools import simplify

    task_shape = [simplify(v) for v in task_shape]
    if ranks is None:
        ranks = list(range(len(task_shape)))
    return SpatialTaskMapping(task_shape=task_shape, ranks=ranks)


def row_spatial(*task_shape: Int):
    return spatial_map(task_shape)


def col_spatial(*task_shape: Int):
    return spatial_map(task_shape, ranks=list(reversed(range(len(task_shape)))))


def repeat_map(task_shape: Sequence[Int], ranks: Optional[Sequence[int]] = None, attrs: Optional[str] = None):
    from tilus.hidet.ir.stmt import ForStmtAttr
    from tilus.hidet.ir.tools import simplify

    task_shape = [simplify(v) for v in task_shape]
    if ranks is None:
        ranks = list(range(len(task_shape)))
    if attrs is None:
        attrs = [ForStmtAttr.from_extent(task_shape[i]) for i in range(len(task_shape))]
    else:
        assert isinstance(attrs, str)
        attrs: List[ForStmtAttr] = ForStmtAttr.parse(attrs, len(task_shape))
        if len(attrs) == 1:
            attrs = attrs * len(task_shape)
        if len(attrs) != len(task_shape):
            raise ValueError(f"Invalid number of attributes: {len(attrs)} vs {len(task_shape)}")
    return RepeatTaskMapping(task_shape=task_shape, ranks=ranks, attrs=attrs)


def row_repeat(*task_shape: Int, attrs: Optional[str] = None):
    return repeat_map(task_shape, attrs=attrs)


def col_repeat(*task_shape: Int, attrs: Optional[str] = None):
    return repeat_map(task_shape, ranks=list(reversed(range(len(task_shape)))), attrs=attrs)


def auto_map(
    *task_shape: Int,
    workers: Int,
    on_fail: Optional[Callable[[str], None]] = None,
    ranks: Optional[Sequence[int]] = None,
) -> Optional[TaskMapping]:
    """
    Automatically generate a task mapping composed by spatial and repeat mappings.

    Given the task shape [d1, d2, ..., dn] and the number of workers m, this function tries to find a task mapping
    composed by spatial and repeat mappings that distribute the tasks to different workers. The goal is to make the
    contiguous workers to process contiguous tasks. The number of tasks must be a multiple of the number of workers.

    Some examples:
        - the default ranks will be used if not specified:
        auto_map(8, 32, workers=64) = repeat(4, 1).spatial(2, 32)
        auto_map(8, 64, workers=64) = repeat(8, 1).spatial(1, 64)
        auto_map(8, 96, workers=64) = repeat(4, 3).spatial(2, 32)
        auto_map(8, 128, workers=64) = repeat(8, 2).spatial(1, 64)

        - the ranks specify the order of dimensions to be mapped for contiguous workers:
        auto_map(8, 96, workers=64, ranks=[1, 0]) = repeat(1, 12).spatial(8, 8)
        auto_map(64, 8, workers=64, ranks=[1, 0]) = repeat(1, 8).spatial(64, 1)

        - on_fail will be called if the task shape cannot be mapped to the number of workers:
        auto_map(8, 41, workers=64) = None (on_fail will be called with an error message)

        - task shape can have arbitrary dimensions:
        auto_map(8, 16, 24, workers=32) = repeat(8, 4, 3).spatial(1, 4, 8)

    Parameters
    ----------
    task_shape: Sequence[Expr or int]
        The task shape.
    workers: Expr or int
        The number of workers.
    on_fail: Callable[[str], None], optional
        The callback function to be called when the task shape cannot be mapped to the number of workers.
        This function it not necessarily to return (e.g., it can raise an exception).
        The default is None, which means no callback function will be called, but a RuntimeError will be raised.
    ranks:
        The ranks of the task shape to be mapped to the workers. If not specified, the ranks will be
        [0, 1, ..., n-1], where n is the number of dimensions.

    Returns
    -------
    ret: Optional[TaskMapping]
        The task mapping. If the task shape cannot be mapped to the number of workers and on_fail returns, None will be
        returned.
    """
    from tilus.hidet.ir.tools import simplify_to_int

    # automatic mapping requires both task shape and number of workers to be constant
    task_shape: List[int] = [simplify_to_int(v) for v in task_shape]
    workers: int = simplify_to_int(workers)

    num_tasks = prod(task_shape)
    if num_tasks % workers != 0:
        msg = "Expect the number of tasks {} in task shape {} be a multiple of number of workers {}.".format(
            num_tasks, task_shape, workers
        )
        if on_fail is None:
            raise RuntimeError(msg)
        else:
            on_fail(msg)
        return None

    num_dims = len(task_shape)
    if ranks is None:
        ranks = list(range(num_dims))
    else:
        ranks = list(ranks)
    spatial_shape = [0] * num_dims
    repeat_shape = [0] * num_dims

    remain: int = workers
    for rank in reversed(range(num_dims)):
        dim = ranks.index(rank)
        spatial_shape[dim] = gcd(remain, task_shape[dim])
        repeat_shape[dim] = task_shape[dim] // spatial_shape[dim]
        remain //= spatial_shape[dim]
    return repeat_map(repeat_shape, ranks) * spatial_map(spatial_shape, ranks)


def predicated_auto_map(
    *task_shape: Int, workers: Int, ranks: Optional[Sequence[int]] = None
) -> Tuple[TaskMapping, Callable[[Expr], Expr]]:
    from tilus.hidet.ir.tools import simplify_to_int

    # automatic mapping requires both task shape and number of workers to be constant
    task_shape: List[int] = [simplify_to_int(v) for v in task_shape]
    workers: int = simplify_to_int(workers)

    num_dims: int = len(task_shape)
    ranks: List[int]
    if ranks is None:
        ranks = list(range(num_dims))
    else:
        ranks = list(ranks)
    spatial_shape: List[int] = [0] * num_dims
    repeat_shape: List[int] = [0] * num_dims

    remain: int = workers
    for rank in reversed(range(num_dims)):
        dim = ranks.index(rank)

        if remain >= task_shape[dim]:
            spatial_shape[dim] = task_shape[dim]
            repeat_shape[dim] = 1
            remain //= task_shape[dim]
        else:
            spatial_shape[dim] = gcd(remain, task_shape[dim])
            repeat_shape[dim] = task_shape[dim] // spatial_shape[dim]
            remain = 1

    used_workers: int = prod(spatial_shape)

    def predicate(worker_idx: Union[Expr, int]) -> Expr:
        return convert(worker_idx < used_workers, dtype="bool")

    return repeat_map(repeat_shape, ranks) * spatial_map(spatial_shape, ranks), predicate


# Several mappings are represented in TaskMapping as a reqursive sequence of mappings
# Turn it to the list.
def mapping_2_list(mapping: TaskMapping) -> List[TaskMapping]:
    if not isinstance(mapping, ComposedTaskMapping):
        return [mapping]
    res = []
    while isinstance(mapping, ComposedTaskMapping):
        outer, inner = mapping.outer, mapping.inner
        res.append(inner)
        mapping = outer
    res.append(mapping)
    return res


# Convert the list of task mappings to `TaskMapping`(`ComposedTaskMapping`).
# `ComposedTaskMapping` is reqursive representation of sequence of mappings
def list_2_mapping(mapping_list: List[TaskMapping]) -> TaskMapping:
    assert len(mapping_list) != 0
    outer = mapping_list[-1]
    for mapp in reversed(mapping_list[:-1]):
        outer = ComposedTaskMapping(outer=outer, inner=mapp)
    return outer
