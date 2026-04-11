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
# pylint: disable=bad-staticmethod-argument
from typing import List

from tilus.hidet.ir.expr import Expr, Var
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.stmt import (
    AsmStmt,
    AssertStmt,
    AssignStmt,
    BlackBoxStmt,
    BreakStmt,
    BufferStoreStmt,
    ContinueStmt,
    DeclareStmt,
    EvaluateStmt,
    ForMappingStmt,
    ForStmt,
    IfStmt,
    LaunchKernelStmt,
    LetStmt,
    ReturnStmt,
    SeqStmt,
    Stmt,
    WhileStmt,
)
from tilus.hidet.utils import same_list

from .base_functor import BaseFunctor, BaseRewriter, BaseVisitor


def _unchanged(a, b):
    """Check if a is the same as b, using same_as for tvm_ffi Objects."""
    return a is b or (hasattr(a, "same_as") and a.same_as(b))


class StmtFunctor(BaseFunctor):
    _type_dispatch = {
        EvaluateStmt: "visit_EvaluateStmt",
        DeclareStmt: "visit_DeclareStmt",
        BufferStoreStmt: "visit_BufferStoreStmt",
        AssignStmt: "visit_AssignStmt",
        LetStmt: "visit_LetStmt",
        ForStmt: "visit_ForStmt",
        ForMappingStmt: "visit_ForTaskStmt",
        WhileStmt: "visit_WhileStmt",
        BreakStmt: "visit_BreakStmt",
        ContinueStmt: "visit_ContinueStmt",
        IfStmt: "visit_IfStmt",
        ReturnStmt: "visit_ReturnStmt",
        AsmStmt: "visit_AsmStmt",
        LaunchKernelStmt: "visit_LaunchKernelStmt",
        AssertStmt: "visit_AssertStmt",
        BlackBoxStmt: "visit_BlackBoxStmt",
        SeqStmt: "visit_SeqStmt",
    }

    def visit_dispatch(self, node: Node):
        method_name = StmtFunctor._type_dispatch.get(type(node))
        if method_name is not None:
            return getattr(self, method_name)(node)
        return NotImplemented

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        raise NotImplementedError()

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        raise NotImplementedError()

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        raise NotImplementedError()

    def visit_AssignStmt(self, stmt: AssignStmt):
        raise NotImplementedError()

    def visit_LetStmt(self, stmt: LetStmt):
        raise NotImplementedError()

    def visit_ForStmt(self, stmt: ForStmt):
        raise NotImplementedError()

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        raise NotImplementedError()

    def visit_WhileStmt(self, stmt: WhileStmt):
        raise NotImplementedError()

    def visit_BreakStmt(self, stmt: BreakStmt):
        raise NotImplementedError()

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        raise NotImplementedError()

    def visit_IfStmt(self, stmt: IfStmt):
        raise NotImplementedError()

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        raise NotImplementedError()

    def visit_AssertStmt(self, stmt: AssertStmt):
        raise NotImplementedError()

    def visit_AsmStmt(self, stmt: AsmStmt):
        raise NotImplementedError()

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        raise NotImplementedError()

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt):
        raise NotImplementedError()


class StmtVisitor(StmtFunctor, BaseVisitor):
    def visit_DeclareStmt(self, stmt: DeclareStmt):
        self.visit(stmt.var)
        if stmt.init is not None:
            self.visit(stmt.init)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        self.visit(stmt.expr)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        self.visit(stmt.buf)
        self.visit(stmt.value)
        for idx in stmt.indices:
            self.visit(idx)

    def visit_AssignStmt(self, stmt: AssignStmt):
        self.visit(stmt.var)
        self.visit(stmt.value)

    def visit_LetStmt(self, stmt: LetStmt):
        for _, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit(bind_value)
        self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        self.visit(stmt.extent)
        self.visit(stmt.body)

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        for loop_var in stmt.loop_vars:
            self.visit(loop_var)
        self.visit(stmt.worker)
        self.visit(stmt.body)

    def visit_WhileStmt(self, stmt: WhileStmt):
        self.visit(stmt.cond)
        self.visit(stmt.body)

    def visit_BreakStmt(self, stmt: BreakStmt):
        pass

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        pass

    def visit_IfStmt(self, stmt: IfStmt):
        self.visit(stmt.cond)
        self.visit(stmt.then_body)
        if stmt.else_body:
            self.visit(stmt.else_body)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        self.visit(stmt.ret_value)

    def visit_AssertStmt(self, stmt: AssertStmt):
        self.visit(stmt.cond)

    def visit_AsmStmt(self, stmt: AsmStmt):
        for expr in stmt.input_exprs:
            self.visit(expr)
        for expr in stmt.output_exprs:
            self.visit(expr)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        self.visit(stmt.func_var)
        for arg in stmt.args:
            self.visit(arg)
        for dim in stmt.grid_dim:
            self.visit(dim)
        for dim in stmt.block_dim:
            self.visit(dim)
        self.visit(stmt.shared_mem_bytes)

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        for expr in stmt.exprs:
            self.visit(expr)

    def visit_SeqStmt(self, stmt: SeqStmt):
        for s in stmt.seq:
            self.visit(s)


class StmtRewriter(StmtFunctor, BaseRewriter):
    def __init__(self, use_memo: bool = True):
        super().__init__(use_memo=use_memo)
        self._prologues: List[Stmt] = []

    def append_prologue_stmt(self, stmt: Stmt):
        self._prologues.append(stmt)

    def visit_dispatch(self, node: Node):
        ret = super().visit_dispatch(node)
        if ret is NotImplemented:
            # can not dispatch to the statement functors
            return NotImplemented
        else:
            assert isinstance(ret, Stmt), node

        if self._prologues:
            if isinstance(ret, SeqStmt):
                seq = self._prologues + list(ret.seq)
                ret = SeqStmt(seq)
            else:
                ret = SeqStmt(self._prologues + [ret])
            self._prologues = []
        return ret

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        orig_var = stmt.var
        orig_init = stmt.init
        v = self.visit(orig_var)
        init = self.visit(orig_init) if orig_init is not None else None
        if _unchanged(v, orig_var) and _unchanged(init, orig_init):
            return stmt
        else:
            return DeclareStmt(v, init, stmt.is_static, stmt.scope)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        orig_expr = stmt.expr
        e = self.visit(orig_expr)
        if _unchanged(e, orig_expr):
            return stmt
        else:
            return EvaluateStmt(e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        orig_buf = stmt.buf
        orig_indices = stmt.indices
        orig_value = stmt.value
        buf = self.visit(orig_buf)
        indices = [self.visit(e) for e in orig_indices]
        value = self.visit(orig_value)
        if (
            _unchanged(buf, orig_buf)
            and all(_unchanged(a, b) for a, b in zip(indices, orig_indices))
            and _unchanged(value, orig_value)
        ):
            return stmt
        else:
            return BufferStoreStmt(buf, indices, value, stmt.protected)

    def visit_AssignStmt(self, stmt: AssignStmt):
        orig_var = stmt.var
        orig_value = stmt.value
        v = self.visit(orig_var)
        value = self.visit(orig_value)
        if _unchanged(v, orig_var) and _unchanged(value, orig_value):
            return stmt
        else:
            return AssignStmt(v, value)

    def visit_LetStmt(self, stmt: LetStmt):
        orig_bind_vars = stmt.bind_vars
        orig_bind_values = stmt.bind_values
        orig_body = stmt.body
        bind_vars = [self.visit(bind_var) for bind_var in orig_bind_vars]
        bind_values = [self.visit(bind_value) for bind_value in orig_bind_values]
        body = self.visit(orig_body)
        if (
            same_list(bind_vars, orig_bind_vars)
            and same_list(bind_values, orig_bind_values)
            and _unchanged(body, orig_body)
        ):
            return stmt
        else:
            return LetStmt(bind_vars, bind_values, body)

    def visit_ForStmt(self, stmt: ForStmt):
        orig_loop_var = stmt.loop_var
        orig_extent = stmt.extent
        orig_body = stmt.body
        loop_var = self.visit(orig_loop_var)
        extent = self.visit(orig_extent)
        body = self.visit(orig_body)
        if _unchanged(loop_var, orig_loop_var) and _unchanged(extent, orig_extent) and _unchanged(body, orig_body):
            return stmt
        else:
            return ForStmt(loop_var, extent, body=body, attr=stmt.attr)

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        orig_loop_vars = stmt.loop_vars
        orig_mapping = stmt.mapping
        orig_worker = stmt.worker
        orig_body = stmt.body
        loop_vars: List[Expr] = [self.visit(v) for v in orig_loop_vars]
        mapping = self.visit(orig_mapping)
        worker = self.visit(orig_worker)
        body = self.visit(orig_body)
        if (
            same_list(loop_vars, orig_loop_vars)
            and _unchanged(worker, orig_worker)
            and _unchanged(body, orig_body)
            and _unchanged(mapping, orig_mapping)
        ):
            return stmt
        else:
            assert all(isinstance(v, Var) for v in loop_vars)
            asserted_loop_vars: List[Var] = [v for v in loop_vars if isinstance(v, Var)]  # avoid IDE warning
            return ForMappingStmt(loop_vars=asserted_loop_vars, mapping=mapping, worker=worker, body=body)

    def visit_WhileStmt(self, stmt: WhileStmt):
        orig_cond = stmt.cond
        orig_body = stmt.body
        cond = self.visit(orig_cond)
        body = self.visit(orig_body)
        if _unchanged(cond, orig_cond) and _unchanged(body, orig_body):
            return stmt
        else:
            return WhileStmt(cond, body)

    def visit_BreakStmt(self, stmt: BreakStmt):
        return stmt

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        return stmt

    def visit_IfStmt(self, stmt: IfStmt):
        orig_cond = stmt.cond
        orig_then_body = stmt.then_body
        orig_else_body = stmt.else_body
        cond = self.visit(orig_cond)
        then_body = self.visit(orig_then_body)
        else_body = self.visit(orig_else_body) if orig_else_body else None
        if (
            _unchanged(cond, orig_cond)
            and _unchanged(then_body, orig_then_body)
            and _unchanged(else_body, orig_else_body)
        ):
            return stmt
        else:
            return IfStmt(cond, then_body, else_body)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        orig_ret_value = stmt.ret_value
        ret_value = self.visit(orig_ret_value) if orig_ret_value is not None else None
        if _unchanged(ret_value, orig_ret_value):
            return stmt
        else:
            return ReturnStmt(ret_value)

    def visit_AssertStmt(self, stmt: AssertStmt):
        orig_cond = stmt.cond
        cond = self.visit(orig_cond)
        if _unchanged(cond, orig_cond):
            return stmt
        else:
            return AssertStmt(cond, stmt.msg)

    def visit_AsmStmt(self, stmt: AsmStmt):
        orig_input_exprs = stmt.input_exprs
        orig_output_exprs = stmt.output_exprs
        input_exprs = [self.visit(e) for e in orig_input_exprs]
        output_exprs = [self.visit(e) for e in orig_output_exprs]
        if same_list(input_exprs, orig_input_exprs) and same_list(output_exprs, orig_output_exprs):
            return stmt
        else:
            return AsmStmt(
                template_string=stmt.template_string,
                output_labels=stmt.output_labels,
                output_exprs=output_exprs,
                input_labels=stmt.input_labels,
                input_exprs=input_exprs,
                is_volatile=stmt.is_volatile,
                memory_fence=stmt.memory_fence,
            )

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        orig_func_var = stmt.func_var
        orig_args = stmt.args
        orig_grid_dim = stmt.grid_dim
        orig_cluster_dim = stmt.cluster_dim
        orig_block_dim = stmt.block_dim
        orig_shared_mem_bytes = stmt.shared_mem_bytes
        func_var = self.visit(orig_func_var)
        args = [self.visit(e) for e in orig_args]
        grid_dim = tuple(self.visit(orig_grid_dim[i]) for i in range(3))
        cluster_dim = tuple(self.visit(orig_cluster_dim[i]) for i in range(3))
        block_dim = tuple(self.visit(orig_block_dim[i]) for i in range(3))
        shared_mem_bytes = self.visit(orig_shared_mem_bytes)
        if same_list(
            [func_var, *args, *grid_dim, *block_dim, shared_mem_bytes],
            [orig_func_var, *orig_args, *orig_grid_dim, *orig_block_dim, orig_shared_mem_bytes],
        ):
            return stmt
        else:
            return LaunchKernelStmt(
                func_var=func_var,
                args=args,
                grid_dim=grid_dim,
                cluster_dim=cluster_dim,
                block_dim=block_dim,
                shared_mem_bytes=shared_mem_bytes,
                target=stmt.target,
            )

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        orig_exprs = stmt.exprs
        exprs = [self.visit(e) for e in orig_exprs]
        if same_list(exprs, orig_exprs):
            return stmt
        else:
            return BlackBoxStmt(template_string=stmt.template_string, exprs=exprs)

    def visit_SeqStmt(self, stmt: SeqStmt):
        orig_seq = stmt.seq
        seq = []
        for s in orig_seq:
            seq.append(self.visit(s))
        if all(_unchanged(a, b) for a, b in zip(seq, orig_seq)):
            return stmt
        else:
            return SeqStmt(seq)
