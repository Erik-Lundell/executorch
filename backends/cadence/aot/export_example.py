# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

# pyre-unsafe

import logging
import tempfile

from executorch.backends.cadence.aot.ops_registrations import *  # noqa
from typing import Any, Tuple

from executorch.backends.cadence.aot.compiler import (
    convert_pt2,
    export_to_executorch_gen_etrecord,
    fuse_pt2,
    prepare_pt2,
)

from executorch.backends.cadence.aot.quantizer.quantizer import CadenceDefaultQuantizer
from executorch.backends.cadence.runtime import runtime
from executorch.backends.cadence.runtime.executor import BundledProgramManager
from executorch.exir import ExecutorchProgramManager
from torch import nn

from .utils import save_bpte_program, save_pte_program


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def export_model(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    file_name: str = "CadenceDemoModel",
    run_and_compare: bool = True,
    eps_error: float = 1e-1,
    eps_warn: float = 1e-5,
):
    # create work directory for outputs and model binary
    working_dir = tempfile.mkdtemp(dir="/tmp")
    logging.debug(f"Created work directory {working_dir}")

    # Instantiate the quantizer
    quantizer = CadenceDefaultQuantizer()

    # Prepare the model
    prepared_gm = prepare_pt2(model, example_inputs, quantizer)

    # Calibrate the model
    for samples in [example_inputs]:
        prepared_gm(*samples)

    # Convert the model
    converted_model = convert_pt2(prepared_gm)

    # Get reference outputs from converted model
    ref_outputs = converted_model(*example_inputs)

    # Quantize the model (note: quantizer needs to be the same as
    # the one used in prepare_and_convert_pt2)
    quantized_model = fuse_pt2(converted_model, quantizer)

    # Get edge program after Cadence specific passes
    exec_prog: ExecutorchProgramManager = export_to_executorch_gen_etrecord(
        quantized_model, example_inputs, output_dir=working_dir
    )

    logging.info("Final exported graph:\n")
    exec_prog.exported_program().graph_module.graph.print_tabular()

    forward_test_data = BundledProgramManager.bundled_program_test_data_gen(
        method="forward", inputs=example_inputs, expected_outputs=ref_outputs
    )
    bundled_program_manager = BundledProgramManager([forward_test_data])
    buffer = bundled_program_manager._serialize(
        exec_prog,
        bundled_program_manager.get_method_test_suites(),
        forward_test_data,
    )
    # Save the program as pte (default name is CadenceDemoModel.pte)
    save_pte_program(exec_prog, file_name, working_dir)
    # Save the program as btpe (default name is CadenceDemoModel.bpte)
    save_bpte_program(buffer, file_name, working_dir)

    logging.debug(
        f"Executorch bundled program buffer saved to {file_name} is {len(buffer)} total bytes"
    )

    # TODO: move to test infra
    if run_and_compare:
        runtime.run_and_compare(
            executorch_prog=exec_prog,
            inputs=example_inputs,
            ref_outputs=ref_outputs,
            working_dir=working_dir,
            eps_error=eps_error,
            eps_warn=eps_warn,
        )
