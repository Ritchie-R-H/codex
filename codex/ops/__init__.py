# Copyright 2022 CoDeX authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# ========================================================================================
"""Operations."""

from codex.ops.activation import verysoftplus
from codex.ops.gradient import lower_limit
from codex.ops.gradient import perturb_and_apply
from codex.ops.gradient import upper_limit
from codex.ops.quantization import soft_round
from codex.ops.quantization import soft_round_conditional_mean
from codex.ops.quantization import soft_round_inverse
from codex.ops.quantization import ste_argmax
from codex.ops.quantization import ste_round
