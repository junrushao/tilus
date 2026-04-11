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

import string
from typing import Any

from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.expr import Call
from tilus.hidet.ir.node import Node

_VALID_KINDS = {
    "cuda_kernel",
    "cuda_internal",
    "hip_kernel",
    "hip_internal",
    "cpu_kernel",
    "cpu_internal",
    "public",
}


def check_func_name(name: str):
    if len(name) == 0:
        raise ValueError("Do not allow empty function name.")
    for c in name:
        if not (c in string.ascii_lowercase or c in string.ascii_uppercase or c in string.digits or c in "_"):
            raise ValueError("Cannot use {} in function name".format(repr(c)))


@py_class
class Function(Node):
    """
    Valid Attrs:
        'kind': str,
            the kind of this function.
                - 'cuda_internal': this is a cuda device function, can only be called by cuda function
                - 'cuda_kernel': this is a cuda kernel function
                - 'cpu_kernel': this is a cpu kernel function
                - 'cpu_internal': this is a cpu function but not a kernel
                - 'public': this is a packed function that wraps kernel function(s)
        'cuda.grid_dim': Union[int, List[int]]
            the grid dimension in cuda launch configuration
        'cuda.cluster_dim': Union[int, List[int]]
            the cluster dimension in cuda launch configuration
        'cuda.block_dim': Union[int, List[int]]
            the block dimension in cuda launch configuration
        'cuda.dynamic_smem_bytes': int
            the dynamic shared memory in cuda launch configuration
        'cuda.min_blocks': int
            the minimal number of thread blocks in launch bound of cuda kernel function
    """

    name: str
    params: Any
    body: Any
    ret_type: Any
    kind: str
    attrs: Any = None

    def __post_init__(self):
        check_func_name(self.name)
        assert isinstance(self.kind, str) and self.kind in _VALID_KINDS
        if self.attrs is None:
            self.attrs = {}

    def __call__(self, *args, **kwargs) -> Call:
        raise ValueError("Can only call script function in another script function, or lower it to execute.")

    def get_attr(self, attr_name, default=None, allow_missing=False):
        """
        Get attribute of this function.

        When default is not None or allow_missing is True, this function will return the default value (in case
        default is not None) or None (in case default is None) when the attribute is not found. Otherwise,
        this function will raise a KeyError.

        Parameters
        ----------
        attr_name: str
            The name of attribute

        default: Any, optional
            The default value of attribute

        allow_missing: bool, default False

        Returns
        -------
        attr_value: Any
            The value of attribute
        """
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        if default is not None or allow_missing:
            return default
        else:
            raise KeyError("Attribute {} is not found in function {}".format(attr_name, self.name))
