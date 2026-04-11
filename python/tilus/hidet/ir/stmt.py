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
from __future__ import annotations

import enum
from typing import Any, ClassVar, List, Optional, Sequence, Tuple, Union

import tvm_ffi
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.expr import Constant, Expr, Var, convert
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.type import DataType, PointerType, ReferenceType, TensorPointerType

try:
    from tilus.hidet.ir.compute import TensorNode
except ImportError:
    TensorNode = None
from tilus.hidet.ir.mapping import TaskMapping


# scope
class DeclareScope(enum.Enum):
    """
    The scope of a tensor variable used in declaration statement.
    """

    Default = 0
    Global = 1
    Shared = 2
    Register = 3
    Host = 4

    @staticmethod
    def from_str(name):
        if name == "global":
            return DeclareScope.Global
        elif name == "shared":
            return DeclareScope.Shared
        elif name == "register":
            return DeclareScope.Register
        else:
            return DeclareScope.Default

    def is_global(self):
        return self == DeclareScope.Global

    def is_shared(self):
        return self == DeclareScope.Shared

    def is_register(self):
        return self in (DeclareScope.Register, DeclareScope.Default)

    def is_memory(self):
        return not self.is_register()


@py_class
class ForStmtAttr(tvm_ffi.Object):
    unroll: bool = False
    unroll_factor: Any = None
    unroll_explicit: bool = False
    parallel: bool = False
    parallel_threads: Any = None

    def __str__(self):
        if self.unroll:
            if self.unroll_explicit:
                return "u+"
            elif self.unroll_factor:
                return f"u{self.unroll_factor}"
            else:
                return "u"
        elif self.parallel:
            if self.parallel_threads:
                return f"p{self.parallel_threads}"
            else:
                return "p"
        else:
            return "."

    @staticmethod
    def from_extent(extent: Union[int, Expr]):
        if isinstance(extent, Expr):
            if isinstance(extent, Constant):
                extent = int(extent)
            else:
                return ForStmtAttr()
        if extent < 4:
            return ForStmtAttr(unroll=True, unroll_explicit=True)
        else:
            return ForStmtAttr()

    @staticmethod
    def parse(attr: Optional[str], num_loops: int) -> List[ForStmtAttr]:
        """
        Parse the attribute string and return a list of ForStmtAttr.

        attr-string: attr*
        attr:
             | unroll
             | parallel
             | default
        unroll:
             | 'u'          # unroll
             | 'u' INT+     # unroll with factor, e.g., u1 u2 u3. u1 indicates unroll with factor 1 (i.e., no unroll)
             | 'u' '+'      # explicit unroll, will be unrolled by hidet instead of underlying compiler
        parallel:
             | 'p'          # parallel with available number of threads
             | 'p' INT+     # parallel with specified number of threads
        default: '.'


        Parameters
        ----------
        attr: str
            The attribute string.

        num_loops: int
            The number of loops this attr string describe.

        Returns
        -------
        attrs: List[ForStmtAttr]
            The list of ForStmtAttr.
        """
        if attr is None:
            attr = ""
        s = attr.replace(" ", "")
        idx = 0

        def cur() -> Optional[str]:
            if idx >= len(s):
                return None
            return s[idx]

        attrs: List[ForStmtAttr] = []
        while idx < len(s):
            if s[idx] == ".":
                idx += 1
                attrs.append(ForStmtAttr())
            elif s[idx] == "u":
                idx += 1
                c = cur()
                if c == "+":
                    attrs.append(ForStmtAttr(unroll=True, unroll_explicit=True))
                    idx += 1
                elif c and c.isdigit():
                    unroll_factor = 0
                    while c and c.isdigit():
                        unroll_factor = unroll_factor * 10 + int(c)
                        idx += 1
                        c = cur()
                    if unroll_factor == 0:
                        raise ValueError(f"Invalid attribute string: {attr}")
                    attrs.append(ForStmtAttr(unroll=True, unroll_factor=unroll_factor))
                else:
                    attrs.append(ForStmtAttr(unroll=True, unroll_explicit=False))
            elif s[idx] == "p":
                idx += 1
                c = cur()
                if c and c.isdigit():
                    parallel_threads = 0
                    while c and c.isdigit():
                        parallel_threads = parallel_threads * 10 + int(c)
                        idx += 1
                        c = cur()
                    if parallel_threads == 0:
                        raise ValueError(f"Invalid attribute string: {attr}")
                    attrs.append(ForStmtAttr(parallel=True, parallel_threads=parallel_threads))
                else:
                    attrs.append(ForStmtAttr(parallel=True))
            else:
                raise ValueError(f"Invalid attribute string: {attr}")
        if len(attrs) == 0:
            attrs = [ForStmtAttr() for _ in range(num_loops)]
        elif len(attrs) == 1:
            attrs = attrs * num_loops
        elif len(attrs) != num_loops:
            raise ValueError("Invalid attribute string: {} for {} loops".format(attr, num_loops))
        return attrs


@py_class
class Stmt(Node):
    pass


@py_class
class EvaluateStmt(Stmt):
    expr: Any

    def __post_init__(self):
        self.expr = convert(self.expr)


@py_class
class DeclareStmt(Stmt):
    var: Any
    init: Any = None
    is_static: bool = False
    scope: Any = None

    def __post_init__(self):
        assert isinstance(self.var, Var)
        self.init = convert(self.init)
        self.scope = self.scope if self.scope else DeclareScope.Default


@py_class
class BufferStoreStmt(Stmt):
    buf: Any
    indices: Any
    value: Any
    protected: bool = False

    def __post_init__(self):
        assert isinstance(self.indices, (list, tuple, tvm_ffi.Array)), type(self.indices)
        self.indices = convert(self.indices)
        self.value = convert(self.value)


@py_class
class AssignStmt(Stmt):
    var: Any
    value: Any

    def __post_init__(self):
        assert isinstance(self.var, Var)
        self.value = convert(self.value)


@py_class
class ReturnStmt(Stmt):
    ret_value: Any = None


@py_class
class LetStmt(Stmt):
    bind_vars: Any
    bind_values: Any
    body: Any = None

    def __post_init__(self):
        if not isinstance(self.bind_vars, (list, tuple, tvm_ffi.Array)):
            self.bind_vars = [self.bind_vars]
        if not isinstance(self.bind_values, (list, tuple, tvm_ffi.Array)):
            self.bind_values = [self.bind_values]
        assert len(self.bind_vars) == len(self.bind_values)
        assert len(self.bind_vars) > 0
        self.bind_values = [convert(bind_value) for bind_value in self.bind_values]


@py_class
class ForStmt(Stmt):
    DEFAULT_UNROLL_LIMIT: ClassVar[int] = 32

    loop_var: Any
    extent: Any
    body: Any = None
    attr: Any = None

    def __post_init__(self):
        from tilus.hidet.ir.tools import simplify  # pylint: disable=import-outside-toplevel

        if not self.attr:
            self.attr = ForStmtAttr.from_extent(self.extent)
        self.extent = simplify(convert(self.extent))


@py_class
class ForMappingStmt(Stmt):
    loop_vars: Any
    mapping: Any
    worker: Any
    body: Any

    def __post_init__(self):
        self.loop_vars = list(self.loop_vars)


@py_class
class WhileStmt(Stmt):
    cond: Any
    body: Any


@py_class
class BreakStmt(Stmt):
    pass


@py_class
class ContinueStmt(Stmt):
    pass


@py_class
class IfStmt(Stmt):
    cond: Any
    then_body: Any = None
    else_body: Any = None

    def __post_init__(self):
        self.cond = convert(self.cond)


@py_class
class AssertStmt(Stmt):
    cond: Any
    msg: Any = None

    def __post_init__(self):
        self.cond = convert(self.cond)


@py_class
class AsmStmt(Stmt):
    template_string: Any = ""
    output_labels: Any = ()
    output_exprs: Any = ()
    input_labels: Any = ()
    input_exprs: Any = ()
    is_volatile: bool = False
    memory_fence: bool = False

    @staticmethod
    def from_pairs(
        template_string: str = "",
        outputs: Sequence[Tuple[str, Expr]] = (),
        inputs: Sequence[Tuple[str, Expr]] = (),
        is_volatile=False,
        memory_fence=False,
    ):
        return AsmStmt(
            template_string=template_string,
            output_labels=[pr[0] for pr in outputs],
            output_exprs=[pr[1] for pr in outputs],
            input_labels=[pr[0] for pr in inputs],
            input_exprs=[pr[1] for pr in inputs],
            is_volatile=is_volatile,
            memory_fence=memory_fence,
        )


@py_class
class BlackBoxStmt(Stmt):
    template_string: Any
    exprs: Any = ()

    def __post_init__(self):
        self.exprs = convert(self.exprs)
        expect_args_num = self.template_string.count("{}")
        if expect_args_num != len(self.exprs):
            raise ValueError("Invalid template string: {} for {} args".format(self.template_string, len(self.exprs)))


@py_class
class SeqStmt(Stmt):
    seq: Any

    def __post_init__(self):
        self.seq = tuple(self.seq)
        for stmt in self.seq:
            assert isinstance(stmt, Stmt), str(type(stmt))


@py_class
class LaunchKernelStmt(Stmt):
    _supported_targets: ClassVar[list] = ["cuda", "hip", "cpu"]

    func_var: Any
    args: Any
    grid_dim: Any
    cluster_dim: Any
    block_dim: Any
    shared_mem_bytes: Any
    target: Any

    def __post_init__(self):
        if self.target is not None and self.target not in self._supported_targets:
            raise ValueError(f"Unsupported target: {self.target}")
        self.args = list(self.args)
        assert self.func_var.name is not None


def asm(
    template: str,
    *,
    outputs: Sequence[Any] = (),
    output_inputs: Sequence[Any] = (),
    inputs: Sequence[Any] = (),
    is_volatile=False,
    memory_fence=False,
):
    from tilus.hidet.ir.tools import infer_type  # pylint: disable=import-outside-toplevel

    if not isinstance(outputs, Sequence):
        raise TypeError("outputs must be a sequence")
    if not isinstance(output_inputs, Sequence):
        raise TypeError("output_inputs must be a sequence")
    if not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence")

    updated_outputs = []
    updated_inputs = []

    def get_register_type(expr: Expr) -> str:
        expr = convert(expr)
        expr_type = infer_type(expr)

        if isinstance(expr_type, ReferenceType):
            expr_type = expr_type.base_type

        if isinstance(expr_type, DataType):
            if isinstance(expr, Constant):
                return "n"
            else:
                dtype2reg = {
                    "float16": "h",
                    "float32": "f",
                    "bfloat16": "h",
                    "float64": "d",
                    "uint8": "h",
                    "uint16": "h",
                    "uint32": "r",
                    "uint64": "l",
                    "int8": "h",
                    "int16": "h",
                    "int32": "r",
                    "int64": "l",
                }
                if expr_type.name not in dtype2reg:
                    raise NotImplementedError("{}".format(expr_type))
                return dtype2reg[expr_type.name]
        elif isinstance(expr_type, (PointerType, TensorPointerType)):
            return "l"
        else:
            raise ValueError("Can not deal with type {} in asm code.".format(expr_type))

    for output in outputs:
        constraint = "=" + get_register_type(output)
        updated_outputs.append((constraint, convert(output)))
    for output_input in output_inputs:
        constraint = "+" + get_register_type(output_input)
        updated_outputs.append((constraint, convert(output_input)))
    for x in inputs:
        constraint = get_register_type(x)
        updated_inputs.append((constraint, convert(x)))
    return AsmStmt.from_pairs(template, updated_outputs, updated_inputs, is_volatile, memory_fence)


Int = Union[Expr, int]


def launch_kernel(
    func_var: Var,
    args: Sequence[Expr],
    grid_dim: Union[Sequence[Int], Int],
    block_dim: Union[Sequence[Int], Int],
    cluster_dim: Union[Sequence[Int], Int] = 1,
    shared_mem: Optional[Int] = 0,
    target: str = None,
) -> LaunchKernelStmt:
    launch_config: List[Tuple[Expr, Expr, Expr]] = []
    for dims in [grid_dim, cluster_dim, block_dim]:
        if not isinstance(dims, (list, tuple, tvm_ffi.Array)):
            dims = [dims]
        dims = list(dims)
        if len(dims) > 3:
            raise ValueError("Grid/Cluster/Block dimension must be 3 or less.")
        while len(dims) < 3:
            dims.append(1)
        launch_config.append(convert(dims))
    grid_dim, cluster_dim, block_dim = launch_config
    return LaunchKernelStmt(
        func_var=func_var,
        args=args,
        grid_dim=grid_dim,
        cluster_dim=cluster_dim,
        block_dim=block_dim,
        shared_mem_bytes=convert(shared_mem),
        target=target,
    )
