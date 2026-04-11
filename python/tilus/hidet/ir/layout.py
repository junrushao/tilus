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
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import tvm_ffi
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.node import Node
from tilus.hidet.utils import prod

# typing forward declaration
Expr = "Expr"
Int = Union["Expr", int]
Bool = Union["Expr", bool]


def is_power_of_two(n: int):
    return n != 0 and (n & (n - 1)) == 0


def is_atom(expr: Expr):
    from tilus.hidet.ir import Constant, Var

    return isinstance(expr, (Constant, Var))


def variablize(expr_list: Sequence[Expr], var2value: Dict["Var", Expr]) -> List["Var"]:
    from tilus.hidet.ir import var

    out = []
    for expr in expr_list:
        if is_atom(expr):
            out.append(expr)
        else:
            v = var("v")
            var2value[v] = expr
            out.append(v)
    return out


def concat_let_expr(var2value, body: Expr):
    from tilus.hidet.ir import Let

    for var, value in reversed(var2value.items()):
        body = Let(var, value, body)
    return body


def to_data_layout(obj):
    if isinstance(obj, (tuple, list, tvm_ffi.Array)):
        return row_major(*obj)
    elif isinstance(obj, DataLayout):
        return obj
    else:
        raise ValueError("Can not convert {} to a DataLayout, expect a list or tuple of ints".format(obj))


# data layout
@py_class
class DataLayout(Node):
    shape: Any = None
    size: Any = None

    def __post_init__(self):
        from tilus.hidet import ir

        if self.shape is None:
            self.shape = ()
        else:
            self.shape = tuple(int(v) if isinstance(v, ir.Constant) else v for v in self.shape)
        assert all(isinstance(v, (ir.Expr, int)) for v in self.shape)

    def __call__(self, *args: Int):
        return self.serialize(*args)

    def __add__(self, other):
        return concat(lhs=self, rhs=other)

    def __radd__(self, other):
        return concat(lhs=other, rhs=self)

    def __mul__(self, other):
        return compose(outer=self, inner=other)

    def __str__(self):
        import numpy as np

        from tilus.hidet.ir.expr import Constant

        if not isinstance(self.size, (int, Constant)) or int(self.size) > 1024:
            return "{}(shape={}, size={})".format(self.__class__.__name__, self.shape, self.size)
        else:
            shape = [int(v) for v in self.shape]
            table = np.zeros(shape=shape, dtype=int)
            ranges = [range(v) for v in shape]
            for indices in itertools.product(*ranges):
                local_index = self.global2local(*indices)
                table[indices] = int(local_index)
            return np.array_str(table, max_line_width=120)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]

    def global2local(self, *args: Int) -> Int:
        raise NotImplementedError()

    def global2cond(self, *args: Int) -> Bool:
        raise NotImplementedError()

    def serialize(self, *args: Int):
        if len(args) == 1 and isinstance(args[0], (tuple, list, tvm_ffi.Array)):
            # support usage such as within_bound([1, 2, 3])
            args = args[0]
        assert len(args) == len(self.shape)
        scalar_index = self.global2local(*args)
        return scalar_index

    def within_bound(self, *args: Int):
        if isinstance(args[0], (tuple, list, tvm_ffi.Array)) and len(args) == 1:
            # support usage such as within_bound([1, 2, 3])
            args = args[0]
        assert len(args) == len(self.shape)
        var2value = OrderedDict()
        arg_vars = variablize(args, var2value)
        cond = self.global2cond(*arg_vars)
        cond = concat_let_expr(var2value=var2value, body=cond)
        return cond

    def swizzle(self, dim: int, regards_dim: Optional[int] = None, log_step: int = 0):
        return SwizzleLayout(base=self, dim=dim, regards_dim=regards_dim, log_step=log_step)

    def permute(self, perm: Sequence[int]):
        return PermuteLayout(base=self, perm=perm)

    def reshape(self, shape: Sequence[Int]):
        return ReshapeLayout(base=self, shape=shape)

    def local(self, *shape: Int):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple, tvm_ffi.Array)):
            shape = shape[0]
        inner = LocalLayout(shape=shape)
        return compose(self, inner)

    def row_major(self, *shape: Int):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple, tvm_ffi.Array)):
            shape = shape[0]
        inner = row_major(*shape)
        return compose(self, inner)

    def column_major(self, *shape: Int):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple, tvm_ffi.Array)):
            shape = shape[0]
        inner = column_major(*shape)
        return compose(self, inner)


@py_class
class StridesLayout(DataLayout):
    strides: Any = None

    def __post_init__(self):
        super().__post_init__()
        if self.size is None and self.strides is not None:
            self.size = StridesLayout.storage_size(self.shape, self.strides)

    def global2local(self, *args: Int) -> Int:
        return sum(v * self.strides[i] for i, v in enumerate(args))

    def global2cond(self, *args: Int) -> Bool:
        from tilus.hidet.ir.expr import logical_and

        return logical_and(*[v < s for s, v in zip(self.shape, args)])

    @staticmethod
    def storage_size(shape, strides) -> Expr:
        # assume the strides are positive, but do not assume the tensor is contiguous.
        from tilus.hidet.ir.tools import simplify

        max_index = sum((a - 1) * b for a, b in zip(shape, strides)) + 1
        return simplify(max_index)

    @staticmethod
    def from_shape(shape: Sequence[Int], perm: Sequence[int]):
        return StridesLayout(shape=shape, strides=StridesLayout.shape2strides(shape, perm))

    @staticmethod
    def shape2strides(shape: Sequence[Int], perm: Sequence[int]):
        assert len(shape) == len(perm)
        rank = len(shape)
        tuples = [[i, p, None] for i, p in zip(range(rank), perm)]
        tuples = sorted(tuples, key=lambda t: t[1])
        reordered_shape = [shape[t[0]] for t in tuples]
        for i in range(rank):
            tuples[i][2] = prod(reordered_shape[i + 1 :])
        tuples = sorted(tuples, key=lambda t: t[0])
        strides = [t[2] for t in tuples]
        return strides


@py_class
class RowMajorLayout(StridesLayout):
    pass


@py_class
class ColumnMajorLayout(StridesLayout):
    pass


@py_class
class LocalLayout(DataLayout):
    def __post_init__(self):
        super().__post_init__()
        self.size = 1

    def global2local(self, *args: Int) -> Int:
        return 0

    def global2cond(self, *args: Int) -> Bool:
        from tilus.hidet.ir.expr import logical_and

        return logical_and(*[v < s for s, v in zip(self.shape, args)])


@py_class
class SwizzleLayout(DataLayout):
    """
    Swizzle a layout (called base layout) to get a swizzled data layout. The shape of swizzled layout is the same as
    the base layout.

    Example:
        A 2-dimension tensor with shape [a, b] where a = 2^m for some m and b <= a,
        After swizzle(plan={0: [1]}), we get a data layout with shape [a, b], and
          swizzled_layout(i, j) = base_layout(i ^ j, j)
        (Note, swizzle requires the swizzled dimension to be a power of 2)
    """

    base: Any = None
    dim: int = 0
    regards_dim: Any = None
    log_step: int = 0

    def __post_init__(self):
        if self.regards_dim is None:
            if len(self.base.shape) != 2:
                raise ValueError(
                    "Optional regards_dim is only available for 2-rank layout, got layout with shape {}.".format(
                        self.base.shape
                    )
                )
            self.regards_dim = 1 - self.dim

        if self.dim == self.regards_dim:
            raise ValueError(
                "The swizzle dim and regards dim can not be the same, got {} and {}".format(self.dim, self.regards_dim)
            )
        rank = len(self.base.shape)
        if not (0 <= self.dim < rank and 0 <= self.regards_dim < rank):
            raise ValueError(
                "The dim {} (regards dim {}) out of bound for layout {}".format(
                    self.dim, self.regards_dim, self.base.shape
                )
            )
        if not is_power_of_two(self.base.shape[self.dim]):
            raise ValueError(
                "The swizzled dim {} must be a power of 2, got length {}".format(self.dim, self.base.shape[self.dim])
            )
        self.shape = self.base.shape
        self.size = self.base.size
        super().__post_init__()

    def global2local(self, *args: Int) -> Int:
        assert len(args) == len(self.shape)
        origin_indices = list(args)
        indices = []
        for dim, origin_index in enumerate(origin_indices):
            if dim == self.dim:
                regards_index = origin_indices[self.regards_dim] // (2**self.log_step)
                regards_extent = self.shape[self.regards_dim] // (2**self.log_step)
                if regards_extent > self.shape[dim]:
                    regards_index = regards_index % self.shape[dim]  # prevent the xor making the index out of bound
                indices.append(origin_index ^ regards_index)
            else:
                indices.append(origin_index)
        return self.base.global2local(*indices)

    def global2cond(self, *args: Int) -> Bool:
        return self.base.global2cond(*args)


@py_class
class PermuteLayout(DataLayout):
    base: Any = None
    perm: Any = None
    perm_shape: Any = None

    def __post_init__(self):
        assert len(self.base.shape) == len(self.perm)
        perm_lst = [i for i in self.perm]
        perm_lst.sort()
        assert perm_lst == list(range(len(self.perm)))

        self.perm = list(self.perm)
        if self.perm_shape is None:
            self.perm_shape = [self.base.shape[i] for i in self.perm]
        self.shape = tuple(self.perm_shape)
        self.size = self.base.size
        super().__post_init__()

    def global2local(self, *args: Int) -> Int:
        assert len(args) == len(self.shape)
        permuted_args = [args[i] for i in self.perm]
        return self.base.global2local(*permuted_args)

    def global2cond(self, *args: Int) -> Bool:
        permuted_args = [args[i] for i in self.perm]
        return self.base.global2cond(*permuted_args)


@py_class
class ReshapeLayout(DataLayout):
    base: Any = None
    stride: Any = None
    base_stride: Any = None

    def __post_init__(self):
        # shape is set by the caller; normalize first, then compute derived fields
        super().__post_init__()
        assert prod(self.shape) == self.base.size
        self.size = self.base.size
        if self.stride is None:
            self.stride = [prod(self.shape[i:]) for i in range(1, len(self.shape))] + [1]
        if self.base_stride is None:
            self.base_stride = [prod(self.base.shape[i:]) for i in range(1, len(self.base.shape))] + [1]

    def to_linear_index(self, args: List[Int]) -> Int:
        assert len(args) == len(self.shape)
        return sum(v * s for v, s in zip(args, self.stride))

    def to_base_index(self, lin_index: Int) -> List[Int]:
        return [(lin_index // st) % sh for st, sh in zip(self.base_stride, self.base.shape)]

    def global2local(self, *args: Int) -> Int:
        return self.base.global2local(*self.to_base_index(self.to_linear_index(args)))

    def global2cond(self, *args: Int) -> Bool:
        return self.base.global2cond(*self.to_base_index(self.to_linear_index(args)))


@py_class
class ComposedLayout(DataLayout):
    outer: Any = None
    inner: Any = None

    def __post_init__(self):
        assert len(self.outer.shape) == len(self.inner.shape)
        self.shape = tuple(a * b for a, b in zip(self.outer.shape, self.inner.shape))
        self.size = self.outer.size * self.inner.size
        super().__post_init__()

    def global2local(self, *args: Int) -> Int:
        outer_args = [v // b for v, b in zip(args, self.inner.shape)]
        inner_args = [v % b for v, b in zip(args, self.inner.shape)]
        return self.outer(*outer_args) * self.inner.size + self.inner(*inner_args)

    def global2cond(self, *args: Int) -> Bool:
        from tilus.hidet.ir.expr import LogicalAnd

        outer_args = [v // b for v, b in zip(args, self.inner.shape)]
        inner_args = [v % b for v, b in zip(args, self.inner.shape)]
        return LogicalAnd(self.outer.within_bound(*outer_args), self.inner.within_bound(*inner_args))


@py_class
class ConcatLayout(DataLayout):
    lhs: Any = None
    rhs: Any = None

    def __post_init__(self):
        self.shape = tuple(list(self.lhs.shape) + list(self.rhs.shape))
        self.size = self.lhs.size * self.rhs.size
        super().__post_init__()

    def global2local(self, *args: Int) -> Int:
        lhs_args = args[: len(self.lhs.shape)]
        rhs_args = args[len(self.lhs.shape) :]
        return self.lhs(*lhs_args) * self.rhs.size + self.rhs(*rhs_args)

    def global2cond(self, *args: Int) -> Bool:
        from tilus.hidet.ir.expr import LogicalAnd

        lhs_args = args[: len(self.lhs.shape)]
        rhs_args = args[len(self.lhs.shape) :]
        return LogicalAnd(self.lhs.within_bound(*lhs_args), self.rhs.within_bound(*rhs_args))


def row_major(*shape: Int):
    strides = StridesLayout.shape2strides(shape, list(range(len(shape))))
    return RowMajorLayout(shape=shape, strides=strides)


def column_major(*shape: Int):
    strides = StridesLayout.shape2strides(shape, list(reversed(range(len(shape)))))
    return ColumnMajorLayout(shape=shape, strides=strides)


def local_layout(*shape: Int):
    return LocalLayout(shape=shape)


def strided_layout(shape: Sequence[Int], ranks: Optional[List[int]] = None):
    if ranks is None:
        ranks = list(range(len(shape)))
    return StridesLayout.from_shape(shape, ranks)


def compose(outer: DataLayout, inner: DataLayout) -> DataLayout:
    return ComposedLayout(outer=outer, inner=inner)


def concat(lhs: Union[DataLayout, List[int]], rhs: Union[DataLayout, List[int]]) -> DataLayout:
    lhs = to_data_layout(lhs)
    rhs = to_data_layout(rhs)
    return ConcatLayout(lhs=lhs, rhs=rhs)
