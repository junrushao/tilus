# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

from typing import List, Optional, Sequence

from tvm_ffi import ir_traits as tr
from tvm_ffi import pyast
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.expr import Expr, Var
from tilus.ir.inst import Instruction
from tilus.ir.node import IRNode
from tilus.ir.tensor import Tensor


@py_class
class Stmt(IRNode):
    pass


@py_class
class SeqStmt(Stmt):
    __ffi_ir_traits__ = tr.SeqTraits("$field:seq")

    seq: tuple[Stmt, ...]

    @staticmethod
    def create(seq: Sequence[Stmt]) -> SeqStmt:
        return SeqStmt(tuple(seq))


@py_class
class ForStmt(Stmt):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:iter_var", None, None),
        None, "$field:extent", None, None, None, None, None,
    )

    iter_var: Var
    extent: Expr
    body: Stmt

    # candidates:
    # - None (no annotation),
    # - -1 (unroll all),
    # - n (n >= 1, unroll with factor n)
    unroll_factor: Optional[int]


@py_class
class ThreadGroupStmt(Stmt):
    """Restricts execution of its body to a contiguous, aligned subset of threads.

    Parameters
    ----------
    thread_begin : int
        The index of the first thread in the sub-group, relative to the parent
        thread group.

        **Special value ``-1`` (elect-any):** When ``thread_begin == -1``, the
        hardware is free to choose *any* contiguous, naturally-aligned group of
        ``num_threads`` threads from the parent group.  For example, inside a
        128-thread parent group with ``thread_begin=-1, num_threads=32``, the
        runtime may pick any of the four warps (threads 0-31, 32-63, 64-95, or
        96-127).

        This "elect-any" mode enables the compiler back-end to use **uniform
        registers and uniform predicates** for the thread selection, avoiding
        per-thread branch divergence.  In particular:

        * ``num_threads=1`` corresponds to the ``elect.sync`` semantic — one
          arbitrary thread in the warp executes the body.
        * ``num_threads=32`` lets the back-end pick one warp via a uniform
          predicate instead of a divergent ``threadIdx / 32 == N`` branch.

    num_threads : int
        The number of threads that will execute the body.  Must be a
        power-of-two when ``thread_begin == -1``.
    body : Stmt
        The statement tree to execute within the thread sub-group.
    """

    thread_begin: int
    num_threads: int
    body: Stmt

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: thread_begin=-1 means "elect-any" which we render specially as
        # `with elect_any(num_threads=N):` vs `with thread_group(begin, num_threads):`
        if self.thread_begin == -1:
            ctx = pyast.Call(pyast.Id("elect_any"), [], ["num_threads"], [pyast.Literal(self.num_threads)])
        else:
            ctx = pyast.Call(
                pyast.Id("thread_group"),
                [pyast.Literal(self.thread_begin), pyast.Literal(self.num_threads)],
            )
        body = printer(self.body, path.attr("body"))
        if isinstance(body, pyast.StmtBlock):
            body_stmts = list(body.stmts)
        elif isinstance(body, pyast.Stmt):
            body_stmts = [body]
        else:
            body_stmts = [pyast.ExprStmt(body)]
        return pyast.With(None, ctx, body_stmts)

    @staticmethod
    def create(thread_begin: int, num_threads: int, body: Stmt) -> ThreadGroupStmt:
        return ThreadGroupStmt(thread_begin, num_threads, body)


@py_class
class IfStmt(Stmt):
    __ffi_ir_traits__ = tr.IfTraits("$field:cond", tr.RegionTraits("$field:then_body", None, None, None), tr.RegionTraits("$field:else_body", None, None, None))

    cond: Expr
    then_body: Stmt
    else_body: Stmt

    def with_else_body(self, else_body: Stmt) -> IfStmt:
        return IfStmt(self.cond, self.then_body, else_body)


@py_class
class WhileStmt(Stmt):
    __ffi_ir_traits__ = tr.WhileTraits("$field:cond", tr.RegionTraits("$field:body", None, None, None))

    cond: Expr
    body: Stmt


@py_class
class BreakStmt(Stmt):
    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        return pyast.ExprStmt(pyast.Id("break"))


@py_class
class ReturnStmt(Stmt):
    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        return pyast.ExprStmt(pyast.Id("return"))


@py_class
class DeclareStmt(Stmt):
    __ffi_ir_traits__ = tr.AssignTraits("$field:var", "$field:init", None, None, None, None)

    var: Var
    init: Optional[Expr]


@py_class
class AssignStmt(Stmt):
    __ffi_ir_traits__ = tr.AssignTraits("$field:var", "$field:value", None, None, None, None)

    var: Var
    value: Expr


@py_class
class LetStmt(Stmt):
    bind_vars: tuple[Var, ...]
    bind_values: tuple[Expr, ...]
    body: Stmt

    def __post_init__(self):
        assert len(self.bind_vars) == len(self.bind_values) > 0

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: multi-binding semantics need var_def for each bound variable
        for bv in self.bind_vars:
            if not printer.var_is_defined(bv):
                printer.var_def(bv.name or bv.hint or "v", bv, None)
        items = []
        for i, (bv, bval) in enumerate(zip(self.bind_vars, self.bind_values)):
            lhs = printer(bv, path.attr("bind_vars").array_item(i))
            rhs = printer(bval, path.attr("bind_values").array_item(i))
            items.append(pyast.Assign(lhs, rhs))
        if self.body is not None:
            items.append(printer(self.body, path.attr("body")))
        if len(items) == 1:
            return items[0]
        return items

    @staticmethod
    def create(bind_vars: Sequence[Var], bind_values: Sequence[Expr], body: Stmt) -> LetStmt:
        return LetStmt(tuple(bind_vars), tuple(bind_values), body)


@py_class
class EvaluateStmt(Stmt):
    __ffi_ir_traits__ = tr.AssignTraits(None, "$field:expr", None, None, None, None)

    expr: Expr
    pred: Optional[Expr]


@py_class
class TensorItemPtrStmt(Stmt):
    ptr_var: Var
    tensor: Tensor
    space: str  # 'generic', 'shared', 'global', 'local'

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: binds a Hidet Var to a tensor item pointer — need var_def + structured call
        if not printer.var_is_defined(self.ptr_var):
            printer.var_def(self.ptr_var.name or self.ptr_var.hint or "v", self.ptr_var, None)
        lhs = printer(self.ptr_var, path.attr("ptr_var"))
        tensor_expr = printer(self.tensor, path.attr("tensor"))
        rhs = pyast.Call(
            pyast.Attr(tensor_expr, "item_ptr"),
            [],
            ["space"],
            [pyast.Literal(self.space)],
        )
        ty = printer(self.ptr_var.type, path.attr("ptr_var").attr("type")) if self.ptr_var.type is not None else None
        return pyast.Assign(lhs, rhs, ty)


@py_class
class TensorItemValueStmt(Stmt):
    var: Var
    tensor: Tensor

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        # Reason: binds a Hidet Var to a tensor item value — need var_def + structured call
        if not printer.var_is_defined(self.var):
            printer.var_def(self.var.name or self.var.hint or "v", self.var, None)
        lhs = printer(self.var, path.attr("var"))
        tensor_expr = printer(self.tensor, path.attr("tensor"))
        rhs = pyast.Call(pyast.Attr(tensor_expr, "item"), [])
        ty = printer(self.var.type, path.attr("var").attr("type")) if self.var.type is not None else None
        return pyast.Assign(lhs, rhs, ty)


@py_class
class InstStmt(Stmt):
    inst: Instruction

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath):
        return printer(self.inst, path.attr("inst"))


def seq_stmt(seq: Sequence[Stmt | Instruction]) -> Stmt:
    stmt_seq: List[Stmt] = [InstStmt(item) if isinstance(item, Instruction) else item for item in seq]
    if len(stmt_seq) == 1:
        return stmt_seq[0]
    else:
        return SeqStmt(tuple(stmt_seq))
