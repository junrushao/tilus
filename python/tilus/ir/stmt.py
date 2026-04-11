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
    seq: tuple[Stmt, ...]

    @staticmethod
    def create(seq: Sequence[Stmt]) -> SeqStmt:
        return SeqStmt(tuple(seq))


@py_class
class ForStmt(Stmt):
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

    @staticmethod
    def create(thread_begin: int, num_threads: int, body: Stmt) -> ThreadGroupStmt:
        return ThreadGroupStmt(thread_begin, num_threads, body)


@py_class
class IfStmt(Stmt):
    cond: Expr
    then_body: Stmt
    else_body: Stmt

    def with_else_body(self, else_body: Stmt) -> IfStmt:
        return IfStmt(self.cond, self.then_body, else_body)


@py_class
class WhileStmt(Stmt):
    cond: Expr
    body: Stmt


@py_class
class BreakStmt(Stmt):
    pass


@py_class
class ReturnStmt(Stmt):
    pass


@py_class
class DeclareStmt(Stmt):
    var: Var
    init: Optional[Expr]


@py_class
class AssignStmt(Stmt):
    var: Var
    value: Expr


@py_class
class LetStmt(Stmt):
    bind_vars: tuple[Var, ...]
    bind_values: tuple[Expr, ...]
    body: Stmt

    def __post_init__(self):
        assert len(self.bind_vars) == len(self.bind_values) > 0

    @staticmethod
    def create(bind_vars: Sequence[Var], bind_values: Sequence[Expr], body: Stmt) -> LetStmt:
        return LetStmt(tuple(bind_vars), tuple(bind_values), body)


@py_class
class EvaluateStmt(Stmt):
    expr: Expr
    pred: Optional[Expr]


@py_class
class TensorItemPtrStmt(Stmt):
    ptr_var: Var
    tensor: Tensor
    space: str  # 'generic', 'shared', 'global', 'local'


@py_class
class TensorItemValueStmt(Stmt):
    var: Var
    tensor: Tensor


@py_class
class InstStmt(Stmt):
    inst: Instruction


def seq_stmt(seq: Sequence[Stmt | Instruction]) -> Stmt:
    stmt_seq: List[Stmt] = [InstStmt(item) if isinstance(item, Instruction) else item for item in seq]
    if len(stmt_seq) == 1:
        return stmt_seq[0]
    else:
        return SeqStmt(tuple(stmt_seq))
