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
from dataclasses import dataclass
from typing import Any

import numpy as np
import tvm_ffi
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.type import DataType


@dataclass
class FloatInfo:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: DataType


@py_class
class FloatType(DataType):
    _min_value: Any
    _max_value: Any
    _eps: Any
    _smallest_normal: Any
    _mantissa_nbits: Any = None
    _exponent_nbits: Any = None

    @property
    def mantissa_nbits(self) -> int:
        return self._mantissa_nbits

    @property
    def exponent_nbits(self) -> int:
        return self._exponent_nbits

    def is_float(self) -> bool:
        return True

    def is_integer(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_boolean(self) -> bool:
        return False

    def constant(self, value: Any):
        from tilus.hidet.ir.expr import Constant, constant

        if isinstance(value, Constant):
            value = value.value
        value = float(value)

        if value > self._max_value:
            warnings.warn(
                (
                    "Constant value {} is larger than the maximum value {} of data type {}. "
                    "Truncated to maximum value of {}."
                ).format(value, self._max_value, self.name, self.name)
            )
            value = self._max_value

        if value < self._min_value:
            value = self._min_value

        return constant(value, self)

    @property
    def one(self):
        return self.constant(1.0)

    @property
    def zero(self):
        return self.constant(0.0)

    @property
    def min_value(self):
        return self.constant(self._min_value)

    @property
    def max_value(self):
        return self.constant(self._max_value)

    def finfo(self) -> FloatInfo:
        return FloatInfo(
            bits=self.nbytes * 8,
            eps=self._eps,
            max=self._max_value,
            min=self._min_value,
            smallest_normal=self._smallest_normal,
            dtype=self,
        )


float8_e4m3 = FloatType(
    _name="float8_e4m3",
    _short_name="f8e4m3",
    _nbytes=1,
    _min_value=float(-448),
    _max_value=float(448),
    _eps=2 ** (-2),
    _smallest_normal=2 ** (-6),
    _mantissa_nbits=3,
    _exponent_nbits=4,
)
float8_e5m2 = FloatType(
    _name="float8_e5m2",
    _short_name="f8e5m2",
    _nbytes=1,
    _min_value=float(-57344),
    _max_value=float(57344),
    _eps=2 ** (-2),
    _smallest_normal=2 ** (-14),
    _mantissa_nbits=2,
    _exponent_nbits=5,
)
float16 = FloatType(
    _name="float16",
    _short_name="f16",
    _nbytes=2,
    _min_value=np.finfo(np.float16).min,
    _max_value=np.finfo(np.float16).max,
    _eps=np.finfo(np.float16).eps,
    _smallest_normal=np.finfo(np.float16).tiny,
    _mantissa_nbits=10,
    _exponent_nbits=5,
)
float32 = FloatType(
    _name="float32",
    _short_name="f32",
    _nbytes=4,
    _min_value=np.finfo(np.float32).min,
    _max_value=np.finfo(np.float32).max,
    _eps=np.finfo(np.float32).eps,
    _smallest_normal=np.finfo(np.float32).tiny,
    _mantissa_nbits=23,
    _exponent_nbits=8,
)
float64 = FloatType(
    _name="float64",
    _short_name="f64",
    _nbytes=8,
    _min_value=np.finfo(np.float64).min,
    _max_value=np.finfo(np.float64).max,
    _eps=np.finfo(np.float64).eps,
    _smallest_normal=np.finfo(np.float64).tiny,
    _mantissa_nbits=52,
    _exponent_nbits=11,
)
bfloat16 = FloatType(
    _name="bfloat16",
    _short_name="bf16",
    _nbytes=2,
    _min_value=-3.4e38,
    _max_value=3.4e38,
    _eps=None,
    _smallest_normal=None,
    _mantissa_nbits=7,
    _exponent_nbits=8,
)  # TODO: find correct eps/smallest_normal values
tfloat32 = FloatType(
    _name="tfloat32",
    _short_name="tf32",
    _nbytes=4,
    _min_value=-3.4e38,
    _max_value=3.4e38,
    _eps=None,
    _smallest_normal=None,
    _mantissa_nbits=10,
    _exponent_nbits=8,
)

f8e4m3 = float8_e4m3
f8e5m2 = float8_e5m2
f16 = float16
f32 = float32
f64 = float64
bf16 = bfloat16
tf32 = tfloat32
