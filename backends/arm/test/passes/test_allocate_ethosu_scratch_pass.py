# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
import unittest
from typing import cast

import torch
from executorch.backends.arm.ethosu._passes import AllocateEthosUScratchPass
from executorch.exir import memory
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.tensor import TensorSpec
from torch.fx import Graph, GraphModule


_BLOCK_HEADER = struct.Struct("<16sIB11s")


def _vela_block(name: str, payload: bytes) -> bytes:
    padded_payload = payload + b"\x00" * (15 - (len(payload) - 1) % 16)
    return (
        _BLOCK_HEADER.pack(
            name.encode("ascii").ljust(16, b"\x00"),
            len(payload),
            0,
            b"\x00" * 11,
        )
        + padded_payload
    )


def _vela_stream(scratch_size: int) -> bytes:
    return b"".join(
        [
            _vela_block("vela_bin_stream", b""),
            _vela_block("scratch_size", struct.pack("<I", scratch_size)),
            _vela_block("vela_end_stream", b""),
        ]
    )


class AllocateEthosUScratchPassTest(unittest.TestCase):
    def _graph_module(self) -> tuple[GraphModule, torch.fx.Node]:
        root = torch.nn.Module()
        root.lowered_module_0 = LoweredBackendModule(
            None,  # type: ignore[arg-type]
            "EthosUBackend",
            _vela_stream(128),
            [],
        )

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["spec"] = TensorSpec.from_tensor(torch.empty(1, dtype=torch.uint8))
        lowered = graph.get_attr("lowered_module_0")
        delegate = graph.call_function(executorch_call_delegate, (lowered, x))
        delegate.meta["spec"] = TensorSpec.from_tensor(
            torch.empty(1, dtype=torch.uint8)
        )
        output = graph.output(delegate)
        output.meta["spec"] = delegate.meta["spec"]
        return GraphModule(root, graph), delegate

    def test_adds_and_plans_scratch_alloc_arg_to_ethosu_delegate(self) -> None:
        gm, delegate = self._graph_module()

        result = AllocateEthosUScratchPass(
            alloc_graph_input=False,
            alloc_graph_output=False,
        ).run(gm)

        self.assertTrue(result.modified)
        scratch_arg = delegate.args[2]
        self.assertIsInstance(scratch_arg, torch.fx.Node)
        scratch = cast(torch.fx.Node, scratch_arg)
        self.assertEqual(scratch.target, memory.alloc)
        self.assertEqual(scratch.args, (((128,), torch.uint8),))
        spec = scratch.meta["spec"]
        self.assertEqual(spec.dtype, torch.uint8)
        self.assertEqual(spec.nbytes(), 128)
        self.assertEqual(spec.alignment, 16)
        self.assertEqual(spec.mem_id, 1)
        self.assertEqual(spec.mem_offset, 0)
        self.assertEqual(spec.mem_offset % 16, 0)
        self.assertGreaterEqual(gm.meta["non_const_buffer_sizes"][1], 128)

    def test_rejects_alignment_that_cannot_align_scratch(self) -> None:
        gm, _ = self._graph_module()

        with self.assertRaisesRegex(ValueError, "multiple of 16 bytes"):
            AllocateEthosUScratchPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
                alignment=8,
            ).run(gm)


if __name__ == "__main__":
    unittest.main()
