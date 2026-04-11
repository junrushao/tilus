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
from tilus.hidet.ir.dtypes.boolean import Boolean
from tilus.hidet.ir.dtypes.complex import ComplexType
from tilus.hidet.ir.dtypes.floats import FloatType
from tilus.hidet.ir.dtypes.floats_subbyte import FloatSubbyteType
from tilus.hidet.ir.dtypes.integer import IntegerType
from tilus.hidet.ir.dtypes.integer_subbyte import IntegerSubbyteType
from tilus.hidet.ir.dtypes.vector import VectorType
from tilus.hidet.ir.type import (
    ArrayType,
    DataType,
    FuncType,
    OpaqueType,
    PointerType,
    ReferenceType,
    StringType,
    TensorPointerType,
    TensorType,
    VoidType,
)
from tilus.hidet.utils import same_list

from .base_functor import BaseFunctor, BaseRewriter, BaseVisitor


def _unchanged(a, b):
    """Check if a is the same as b, using same_as for tvm_ffi Objects."""
    return a is b or (hasattr(a, "same_as") and a.same_as(b))


class TypeFunctor(BaseFunctor):
    _type_dispatch = {
        IntegerType: "visit_DataType",
        FloatType: "visit_DataType",
        Boolean: "visit_DataType",
        ComplexType: "visit_DataType",
        VectorType: "visit_DataType",
        IntegerSubbyteType: "visit_DataType",
        FloatSubbyteType: "visit_DataType",
        TensorType: "visit_TensorType",
        PointerType: "visit_PointerType",
        TensorPointerType: "visit_TensorPointerType",
        ReferenceType: "visit_ReferenceType",
        StringType: "visit_StringType",
        ArrayType: "visit_ArrayType",
        VoidType: "visit_VoidType",
        FuncType: "visit_FuncType",
        OpaqueType: "visit_OpaqueType",
    }

    def visit_dispatch(self, node):
        method_name = TypeFunctor._type_dispatch.get(type(node))
        if method_name is not None:
            return getattr(self, method_name)(node)
        # fallback for unknown DataType subclasses
        if isinstance(node, DataType):
            return self.visit_DataType(node)
        return NotImplemented

    def visit_DataType(self, t: DataType):
        raise NotImplementedError()

    def visit_TensorType(self, t: TensorType):
        raise NotImplementedError()

    def visit_ArrayType(self, t: ArrayType):
        raise NotImplementedError()

    def visit_PointerType(self, t: PointerType):
        raise NotImplementedError()

    def visit_TensorPointerType(self, t: TensorPointerType):
        raise NotImplementedError()

    def visit_ReferenceType(self, t: ReferenceType):
        raise NotImplementedError()

    def visit_StringType(self, t: StringType):
        raise NotImplementedError()

    def visit_VoidType(self, t: VoidType):
        raise NotImplementedError()

    def visit_FuncType(self, t: FuncType):
        raise NotImplementedError()

    def visit_OpaqueType(self, t: OpaqueType):
        raise NotImplementedError()


class TypeVisitor(TypeFunctor, BaseVisitor):
    def visit_DataType(self, t: DataType):
        pass

    def visit_TensorType(self, t: TensorType):
        self.visit(t.dtype)
        self.visit(t.shape)
        self.visit(t.layout)

    def visit_ArrayType(self, t: ArrayType):
        self.visit(t.base_type)

    def visit_PointerType(self, t: PointerType):
        self.visit(t.base_type)

    def visit_TensorPointerType(self, t: TensorPointerType):
        self.visit(t.tensor_type)

    def visit_ReferenceType(self, t: ReferenceType):
        self.visit(t.base_type)

    def visit_StringType(self, t: StringType):
        pass

    def visit_VoidType(self, t: VoidType):
        pass

    def visit_FuncType(self, t: FuncType):
        self.visit(t.ret_type)
        for param_type in t.param_types:
            self.visit(param_type)

    def visit_OpaqueType(self, t: OpaqueType):
        pass


class TypeRewriter(TypeFunctor, BaseRewriter):
    def visit_DataType(self, t: DataType):
        return t

    def visit_TensorType(self, t: TensorType):
        orig_dtype = t.dtype
        orig_shape = t.shape
        orig_layout = t.layout
        dtype = self.visit(orig_dtype)
        shape = self.visit(orig_shape)
        layout = self.visit(orig_layout)
        if dtype == orig_dtype and _unchanged(layout, orig_layout) and same_list(shape, orig_shape):
            return t
        else:
            return TensorType(dtype, shape, layout)

    def visit_ArrayType(self, t: ArrayType):
        orig_base_type = t.base_type
        base_type = self.visit(orig_base_type)
        if base_type == orig_base_type:
            return t
        else:
            return ArrayType(base_type, t.size)

    def visit_PointerType(self, t: PointerType):
        orig_base_type = t.base_type
        base_type = self.visit(orig_base_type)
        if base_type == orig_base_type:
            return t
        else:
            return PointerType(base_type)

    def visit_TensorPointerType(self, t: TensorPointerType):
        orig_tensor_type = t.tensor_type
        tensor_type = self.visit(orig_tensor_type)
        if tensor_type == orig_tensor_type:
            return t
        else:
            return TensorPointerType(tensor_type)

    def visit_ReferenceType(self, t: ReferenceType):
        orig_base_type = t.base_type
        base_type = self.visit(orig_base_type)
        if base_type == orig_base_type:
            return t
        else:
            return ReferenceType(base_type)

    def visit_StringType(self, t: StringType):
        return t

    def visit_VoidType(self, t: VoidType):
        return t

    def visit_FuncType(self, t: FuncType):
        if t.type_infer_func is not None:
            return t
        else:
            orig_ret_type = t.ret_type
            orig_param_types = t.param_types
            ret_type = self.visit(orig_ret_type)
            param_types = [self.visit(param_type) for param_type in orig_param_types]
            if ret_type == orig_ret_type and same_list(param_types, orig_param_types):
                return t
            else:
                return FuncType(param_types, ret_type)

    def visit_OpaqueType(self, t: OpaqueType):
        return t
