# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import struct
from typing import Optional

import torch
from executorch.exir import memory
from executorch.exir._serialize._cord import FileBackedData
from executorch.exir._serialize._named_data_store import NamedDataStoreOutput
from executorch.exir.delegate import executorch_call_delegate, is_lowered_module
from executorch.exir.pass_base import PassResult
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature


_ETHOSU_BACKEND_ID = "EthosUBackend"
_BLOCK_NAME_LENGTH = 16
_BLOCK_HEADER = struct.Struct(f"<{_BLOCK_NAME_LENGTH}sIB11s")
_BLOCK_ALIGNMENT = 16
_ETHOSU_SCRATCH_ALIGNMENT = 16
_EXTERNAL_REFERENCE = 1
_SCRATCH_BLOCK = "scratch_size"


def _aligned_size(size: int) -> int:
    return (size + _BLOCK_ALIGNMENT - 1) & ~(_BLOCK_ALIGNMENT - 1)


def _buffer_to_bytes(buffer: bytes | FileBackedData) -> bytes:
    if isinstance(buffer, FileBackedData):
        return buffer.to_bytes()
    return buffer


def _resolve_external_payload(
    key: bytes,
    named_data_store_output: Optional[NamedDataStoreOutput],
) -> Optional[bytes]:
    if named_data_store_output is None:
        return None
    key_str = key.decode("ascii")

    entry = named_data_store_output.pte_data.get(key_str)
    if entry is not None:
        return _buffer_to_bytes(named_data_store_output.buffers[entry.buffer_index])

    for entries in named_data_store_output.external_data.values():
        entry = entries.get(key_str)
        if entry is not None:
            return _buffer_to_bytes(named_data_store_output.buffers[entry.buffer_index])
    return None


def _extract_ethosu_scratch_size(
    processed_bytes: bytes,
    named_data_store_output: Optional[NamedDataStoreOutput],
) -> Optional[int]:
    offset = 0
    while offset + _BLOCK_HEADER.size <= len(processed_bytes):
        encoded_name, size, external, _reserved = _BLOCK_HEADER.unpack_from(
            processed_bytes, offset
        )
        payload_start = offset + _BLOCK_HEADER.size
        payload_end = payload_start + size
        if payload_end > len(processed_bytes):
            return None

        name = encoded_name.rstrip(b"\x00").decode("ascii")
        payload = processed_bytes[payload_start:payload_end]
        if name == _SCRATCH_BLOCK:
            if external == _EXTERNAL_REFERENCE:
                external_payload = _resolve_external_payload(
                    payload, named_data_store_output
                )
                if external_payload is None:
                    return None
                payload = external_payload
            if len(payload) < 4:
                return None
            return struct.unpack("<I", payload[:4])[0]

        offset = payload_start + _aligned_size(size)
    return None


def _insert_ethosu_scratch_allocs(graph_module: torch.fx.GraphModule) -> bool:
    modified = False
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target != executorch_call_delegate:
            continue
        if len(node.args) == 0 or not isinstance(node.args[0], torch.fx.Node):
            continue

        lowered_node = node.args[0]
        if lowered_node.op != "get_attr" or not isinstance(lowered_node.target, str):
            continue
        lowered_module = getattr(graph_module, lowered_node.target)
        if (
            not is_lowered_module(lowered_module)
            or lowered_module.backend_id != _ETHOSU_BACKEND_ID
            or node.meta.get("_ethosu_scratch_arg") is not None
        ):
            continue

        scratch_size = _extract_ethosu_scratch_size(
            lowered_module.processed_bytes,
            lowered_module.named_data_store_output,
        )
        if scratch_size is None or scratch_size == 0:
            continue

        with graph_module.graph.inserting_before(node):
            scratch = graph_module.graph.call_function(
                memory.alloc, args=(((scratch_size,), torch.uint8),)
            )
        scratch.meta["spec"] = TensorSpec(
            dtype=torch.uint8,
            shape=torch.Size([scratch_size]),
        )
        scratch.meta["spec"].realign(_ETHOSU_SCRATCH_ALIGNMENT)
        scratch.meta["val"] = torch.empty(
            (scratch_size,), dtype=torch.uint8, device="meta"
        )
        node.args = (*node.args, scratch)
        node.meta["_ethosu_scratch_arg"] = scratch.name
        modified = True

    if modified:
        graph_module.graph.lint()
        graph_module.recompile()

    return modified


class AllocateEthosUScratchPass(MemoryPlanningPass):
    """Inserts Ethos-U scratch allocations before normal memory planning.

    Use this as ExecutorchBackendConfig.memory_planning_pass when exporting an
    Ethos-U program that should pass scratch as a planned delegate argument.

    """

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        if self.alignment % _ETHOSU_SCRATCH_ALIGNMENT != 0:
            raise ValueError(
                "AllocateEthosUScratchPass requires memory planning alignment "
                f"to be a multiple of {_ETHOSU_SCRATCH_ALIGNMENT} bytes"
            )
        _insert_ethosu_scratch_allocs(graph_module)
        return super().run(graph_module, graph_signature)
