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
import warnings
from typing import Any

import tvm_ffi
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.type import DataType


@py_class
class Boolean(DataType):
    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        # True for 1, False for 0
        return True

    def is_complex(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_boolean(self) -> bool:
        return True

    def constant(self, value: Any):
        from tilus.hidet.ir.expr import constant

        if isinstance(value, float):
            warnings.warn("Converting float to boolean when creating constant.")
        value = bool(value)
        return constant(value, self)

    @property
    def one(self):
        return self.constant(True)

    @property
    def zero(self):
        return self.constant(False)

    @property
    def true(self):
        return self.constant(True)

    @property
    def false(self):
        return self.constant(False)

    @property
    def min_value(self):
        raise ValueError("Boolean type has no minimum value.")

    @property
    def max_value(self):
        raise ValueError("Boolean type has no maximum value.")


boolean = Boolean(_name="bool", _short_name="bool", _nbytes=1)
