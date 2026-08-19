# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from executorch.backends.arm import (
    EthosUCompileSpec,
    EthosUPartitioner,
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.ethosu._passes import AllocateEthosUScratchPass
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    memory,
    to_edge_transform_and_lower,
)
from executorch.exir.delegate import executorch_call_delegate
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchvision import models  # type: ignore[import-untyped]


def _build_quantized_program(compile_spec: EthosUCompileSpec):
    model = models.mobilenetv2.mobilenet_v2(weights=None).eval()
    model = model.to(memory_format=torch.channels_last)
    example_inputs = (torch.rand(1, 3, 224, 224).to(memory_format=torch.channels_last),)
    exported_program = torch.export.export(model, example_inputs)
    graph_module = exported_program.module(check_guards=False)

    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())

    prepared = prepare_pt2e(graph_module, quantizer)
    prepared(*example_inputs)
    converted = convert_pt2e(prepared)

    return torch.export.export(converted, example_inputs)


def _assert_ethosu_scratch_arg(executorch_program_manager) -> None:
    graph = executorch_program_manager.exported_program().graph_module.graph
    delegate_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == executorch_call_delegate
    ]
    if len(delegate_nodes) == 0:
        raise AssertionError("Expected at least one Ethos-U delegate")

    for delegate_node in delegate_nodes:
        scratch_args = [
            arg
            for arg in delegate_node.args[1:]
            if isinstance(arg, torch.fx.Node)
            and arg.op == "call_function"
            and arg.target == memory.alloc
        ]
        if len(scratch_args) != 1:
            raise AssertionError(
                f"Expected one planned scratch allocation argument, got {len(scratch_args)}"
            )

        spec = scratch_args[0].meta.get("spec")
        if spec is None or spec.dtype != torch.uint8 or spec.nbytes() == 0:
            raise AssertionError(f"Invalid Ethos-U scratch spec: {spec}")


def export_pte(output_path: Path, target: str) -> None:
    compile_spec = EthosUCompileSpec(
        target,
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
    )
    edge_manager = to_edge_transform_and_lower(
        programs=_build_quantized_program(compile_spec),
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    executorch_program_manager = edge_manager.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=AllocateEthosUScratchPass(),
            extract_delegate_segments=False,
        )
    )
    _assert_ethosu_scratch_arg(executorch_program_manager)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pte_program(executorch_program_manager, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an Ethos-U PTE with explicit planned delegate scratch."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="ethos-u55-128")
    args = parser.parse_args()

    export_pte(args.output, args.target)


if __name__ == "__main__":
    main()
