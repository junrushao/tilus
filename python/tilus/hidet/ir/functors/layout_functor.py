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
from tilus.hidet.ir.layout import (
    ColumnMajorLayout,
    ComposedLayout,
    ConcatLayout,
    DataLayout,
    LocalLayout,
    PermuteLayout,
    ReshapeLayout,
    RowMajorLayout,
    StridesLayout,
    SwizzleLayout,
    row_major,
)
from tilus.hidet.utils import same_list

from .base_functor import BaseFunctor, BaseRewriter, BaseVisitor


def _unchanged(a, b):
    """Check if a is the same as b, using same_as for tvm_ffi Objects."""
    return a is b or (hasattr(a, "same_as") and a.same_as(b))


class LayoutFunctor(BaseFunctor):
    _type_dispatch = {
        StridesLayout: "visit_StridesLayout",
        RowMajorLayout: "visit_StridesLayout",
        ColumnMajorLayout: "visit_StridesLayout",
        LocalLayout: "visit_LocalLayout",
        ComposedLayout: "visit_ComposedLayout",
        SwizzleLayout: "visit_SwizzleLayout",
        ConcatLayout: "visit_ConcatLayout",
        PermuteLayout: "visit_PermuteLayout",
        ReshapeLayout: "visit_ReshapeLayout",
    }

    def visit_dispatch(self, node):
        method_name = LayoutFunctor._type_dispatch.get(type(node))
        if method_name is not None:
            return getattr(self, method_name)(node)
        if isinstance(node, DataLayout):
            raise ValueError("Can not recognize layout {}".format(node))
        return NotImplemented

    def visit_StridesLayout(self, layout: StridesLayout):
        raise NotImplementedError()

    def visit_LocalLayout(self, layout: LocalLayout):
        raise NotImplementedError()

    def visit_ComposedLayout(self, layout: ComposedLayout):
        raise NotImplementedError()

    def visit_SwizzleLayout(self, layout: SwizzleLayout):
        raise NotImplementedError()

    def visit_ConcatLayout(self, layout: ConcatLayout):
        raise NotImplementedError()

    def visit_PermuteLayout(self, layout: PermuteLayout):
        raise NotImplementedError()

    def visit_ReshapeLayout(self, layout: ReshapeLayout):
        raise NotImplementedError()


class LayoutVisitor(BaseVisitor, LayoutFunctor):
    def visit_StridesLayout(self, layout: StridesLayout):
        self.visit(layout.size)
        self.visit(layout.shape)
        self.visit(layout.strides)

    def visit_LocalLayout(self, layout: LocalLayout):
        self.visit(layout.shape)

    def visit_ComposedLayout(self, layout: ComposedLayout):
        self.visit(layout.outer)
        self.visit(layout.inner)

    def visit_SwizzleLayout(self, layout: SwizzleLayout):
        self.visit(layout.base)
        self.visit(layout.shape)
        self.visit(layout.size)

    def visit_ConcatLayout(self, layout: ConcatLayout):
        self.visit(layout.lhs)
        self.visit(layout.rhs)

    def visit_PermuteLayout(self, layout: PermuteLayout):
        self.visit(layout.base)
        self.visit(layout.perm)

    def visit_ReshapeLayout(self, layout: ReshapeLayout):
        self.visit(layout.base)
        self.visit(layout.shape)


class LayoutRewriter(BaseRewriter, LayoutFunctor):
    def visit_StridesLayout(self, layout: StridesLayout):
        orig_shape = layout.shape
        orig_strides = layout.strides
        shape = self.visit(orig_shape)
        strides = self.visit(orig_strides)
        if same_list(shape, orig_shape) and same_list(strides, orig_strides):
            return layout
        else:
            if isinstance(layout, RowMajorLayout):
                return row_major(*shape)
            return StridesLayout(shape=shape, strides=strides)

    def visit_LocalLayout(self, layout: LocalLayout):
        orig_shape = layout.shape
        shape = self.visit(orig_shape)
        if same_list(shape, orig_shape):
            return layout
        else:
            return LocalLayout(shape=shape)

    def visit_ComposedLayout(self, layout: ComposedLayout):
        orig_outer = layout.outer
        orig_inner = layout.inner
        outer = self.visit(orig_outer)
        inner = self.visit(orig_inner)
        if _unchanged(outer, orig_outer) and _unchanged(inner, orig_inner):
            return layout
        else:
            return ComposedLayout(outer=outer, inner=inner)

    def visit_SwizzleLayout(self, layout: SwizzleLayout):
        orig_base = layout.base
        base = self.visit(orig_base)
        if _unchanged(base, orig_base):
            return layout
        else:
            return SwizzleLayout(base=base, dim=layout.dim, regards_dim=layout.regards_dim, log_step=layout.log_step)

    def visit_PermuteLayout(self, layout: PermuteLayout):
        orig_base = layout.base
        base = self.visit(orig_base)
        if _unchanged(base, orig_base):
            return layout
        else:
            return PermuteLayout(base=base, perm=layout.perm)

    def visit_ReshapeLayout(self, layout: ReshapeLayout):
        orig_base = layout.base
        base = self.visit(orig_base)
        if _unchanged(base, orig_base):
            return layout
        else:
            return ReshapeLayout(base=base, shape=layout.shape)

    def visit_ConcatLayout(self, layout: ConcatLayout):
        orig_lhs = layout.lhs
        orig_rhs = layout.rhs
        lhs = self.visit(orig_lhs)
        rhs = self.visit(orig_rhs)
        if _unchanged(lhs, orig_lhs) and _unchanged(rhs, orig_rhs):
            return layout
        else:
            return ConcatLayout(lhs=lhs, rhs=rhs)
