# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from types import NoneType
from typing import cast, List, Optional, Union

import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema

import torch

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)
from executorch.backends.vulkan.utils import (
    is_constant,
    is_get_attr_node,
    is_mutable_buffer_node,
    is_param_node,
    is_symint_node,
    TensorRepr,
)
from executorch.exir.backend.utils import DelegateMappingBuilder

from executorch.exir.tensor import TensorSpec
from torch._export.utils import get_buffer, get_param, is_buffer, is_param
from torch.export import ExportedProgram
from torch.fx import Node

_ScalarType = Union[bool, int, float]
_Argument = Union[
    Node, NoneType, _ScalarType, TensorSpec, List[_ScalarType], List[Node], str
]

logger: logging.Logger = logging.getLogger("")
logger.setLevel(logging.INFO)


class VkGraphBuilder:
    def __init__(
        self,
        program: ExportedProgram,
        delegate_mapping_builder: DelegateMappingBuilder,
        downcast_64_bit: bool = True,
    ) -> None:
        self.program = program
        self.delegate_mapping_builder = delegate_mapping_builder
        self.downcast_64_bit = downcast_64_bit
        self.chain = []
        self.values = []
        self.input_ids = []
        self.output_ids = []
        self.const_tensors = []

        # Mapping from Node to VkValue id
        self.node_to_value_ids = {}
        # Mapping from const scalar value to created VkValue id
        self.const_scalar_to_value_ids = {}

        # For logging
        self.seen_ops = set()

    @staticmethod
    def get_vk_datatype(torch_dtype: torch.dtype) -> vk_graph_schema.VkDataType:
        if torch_dtype == torch.bool:
            return vk_graph_schema.VkDataType.BOOL
        elif torch_dtype == torch.uint8:
            return vk_graph_schema.VkDataType.UINT8
        elif torch_dtype == torch.int8:
            return vk_graph_schema.VkDataType.INT8
        elif torch_dtype == torch.int32:
            return vk_graph_schema.VkDataType.INT32
        elif torch_dtype == torch.int64:
            return vk_graph_schema.VkDataType.INT64
        elif torch_dtype == torch.float16:
            return vk_graph_schema.VkDataType.FLOAT16
        elif torch_dtype == torch.float32:
            return vk_graph_schema.VkDataType.FLOAT32
        elif torch_dtype == torch.float64:
            return vk_graph_schema.VkDataType.FLOAT64
        else:
            raise AssertionError(f"Invalid dtype for vulkan_preprocess ({torch_dtype})")

    def get_constant(self, node: Node) -> Optional[torch.Tensor]:
        """
        Returns the constant associated with the given node in the exported program.
        Returns None if the node is not a constant within the exported program
        """
        if is_constant(self.program, node):
            constant_name = (
                self.program.graph_signature.inputs_to_lifted_tensor_constants[
                    node.name
                ]
            )
            if constant_name in self.program.constants:
                return self.program.constants[constant_name]
            else:
                return None

        return None

    def get_param_tensor(self, node: Node) -> torch.Tensor:
        tensor = None
        if node is None:
            raise RuntimeError("node is None")
        elif is_param(self.program, node):
            tensor = get_param(self.program, node)
        elif is_buffer(self.program, node):
            tensor = get_buffer(self.program, node)
        elif is_constant(self.program, node):
            tensor = self.get_constant(node)
        elif is_get_attr_node(node):
            # This is a hack to support both lifted and unlifted graph
            try:
                tensor = getattr(node.graph.owning_module, node.target)
            except AttributeError:
                tensor = getattr(self.program.graph_module, node.target)
        else:
            raise RuntimeError(f"unsupported param type, {node.op}.")

        assert tensor is not None
        return tensor

    def maybe_add_constant_tensor(self, node: Node) -> int:
        constant_id = -1
        if is_param_node(self.program, node):
            constant_id = len(self.const_tensors)
            self.const_tensors.append(self.get_param_tensor(node))

        return constant_id

    def create_node_value(self, node: Node) -> int:
        # If the node has been marked as a scalar tensor, create a SymInt instead of a tensor
        if is_symint_node(node) or node.meta.get("etvk_is_scalar_tensor", False):
            new_id = self.create_symint_value()
            self.node_to_value_ids[node] = new_id
            return new_id

        spec = node.meta.get("spec")
        if isinstance(spec, TensorSpec):
            constant_id = self.maybe_add_constant_tensor(node)
            new_id = self.create_tensor_value(spec, constant_id)
            self.node_to_value_ids[node] = new_id
            return new_id
        elif isinstance(spec, list) or isinstance(spec, tuple):
            # pyre-ignore[6]: pyre having hard time to infer Node type inside
            # the container.
            new_id = self.create_value_list_value(spec)
            self.node_to_value_ids[node] = new_id
            return new_id
        else:
            raise RuntimeError(
                f"Cannot create value for node {node} with spec of type {type(spec)}"
            )

    def create_null_value(self) -> int:
        new_id = len(self.values)
        self.values.append(vk_graph_schema.VkValue(vk_graph_schema.Null()))
        return new_id

    def get_or_create_scalar_value(self, scalar: _ScalarType) -> int:
        scalar_key = scalar
        # Since Python considers 1 and True to be "equivalent" (as well as 0 and False)
        # to distinguish entries in the dictionary, if scalar is bool then convert it
        # to a string representation to use as a key for the dictionary
        if isinstance(scalar, bool):
            scalar_key = str(scalar)

        if scalar_key in self.const_scalar_to_value_ids:
            return self.const_scalar_to_value_ids[scalar_key]

        new_id = len(self.values)
        if isinstance(scalar, bool):
            self.values.append(vk_graph_schema.VkValue(vk_graph_schema.Bool(scalar)))
        elif isinstance(scalar, int):
            self.values.append(vk_graph_schema.VkValue(vk_graph_schema.Int(scalar)))
        elif isinstance(scalar, float):
            self.values.append(vk_graph_schema.VkValue(vk_graph_schema.Double(scalar)))

        self.const_scalar_to_value_ids[scalar_key] = new_id
        return new_id

    def create_symint_value(self) -> int:
        new_id = len(self.values)
        self.values.append(vk_graph_schema.VkValue(vk_graph_schema.SymInt(0)))
        return new_id

    def create_tensor_value(self, spec: TensorSpec, constant_id: int = -1) -> int:
        # Negative id indicates that this tensor will have its own dedicated memory.
        mem_obj_id = -1
        if spec.mem_obj_id is not None:
            mem_obj_id = spec.mem_obj_id

        storage_type = VkStorageType.DEFAULT_STORAGE
        memory_layout = VkMemoryLayout.DEFAULT_LAYOUT
        if hasattr(spec, "etvk_node_repr"):
            # pyre-ignore[16]
            assert isinstance(spec.etvk_node_repr, TensorRepr)
            storage_type = spec.etvk_node_repr.storage_type
            memory_layout = spec.etvk_node_repr.memory_layout

        # Apply downcast logic before getting VK datatype
        effective_dtype = spec.dtype
        if self.downcast_64_bit and spec.dtype == torch.float64:
            effective_dtype = torch.float32
        elif self.downcast_64_bit and spec.dtype == torch.int64:
            effective_dtype = torch.int32

        datatype = self.get_vk_datatype(effective_dtype)

        new_id = len(self.values)
        self.values.append(
            vk_graph_schema.VkValue(
                value=vk_graph_schema.VkTensor(
                    datatype=datatype,
                    dims=spec.shape,
                    constant_id=constant_id,
                    mem_obj_id=mem_obj_id,
                    storage_type=storage_type,
                    memory_layout=memory_layout,
                )
            )
        )
        return new_id

    def create_scalar_list_value(self, arg: List[_ScalarType]) -> int:
        new_id = len(self.values)

        if len(arg) == 0:
            self.values.append(
                vk_graph_schema.VkValue(vk_graph_schema.IntList(items=[]))
            )

        all_bool = True
        all_int = True
        all_float = True
        all_int_or_symint = True

        for val in arg:
            if not isinstance(val, bool):
                all_bool = False
            if not isinstance(val, int):
                all_int = False
                if not (isinstance(val, Node) and is_symint_node(val)):
                    all_int_or_symint = False
            if not isinstance(val, float):
                all_float = False

        if all_bool:
            self.values.append(
                vk_graph_schema.VkValue(
                    vk_graph_schema.BoolList(items=[cast(bool, e) for e in arg])
                )
            )
        if all_int:
            self.values.append(
                vk_graph_schema.VkValue(
                    vk_graph_schema.IntList(items=[cast(int, e) for e in arg])
                )
            )
        elif all_float:
            self.values.append(
                vk_graph_schema.VkValue(
                    vk_graph_schema.DoubleList(items=[cast(float, e) for e in arg])
                )
            )
        elif all_int_or_symint:
            return self.create_value_list_value(arg)
        else:
            raise NotImplementedError(f"Cannot add value for list {arg}")

        return new_id

    def create_value_list_value(self, arg: tuple | list) -> int:
        self.values.append(
            vk_graph_schema.VkValue(
                vk_graph_schema.ValueList(
                    items=[self.get_or_create_value_for(e) for e in arg]
                )
            )
        )
        return len(self.values) - 1

    def create_string_value(self, string: str) -> int:
        new_id = len(self.values)
        self.values.append(
            vk_graph_schema.VkValue(vk_graph_schema.String(string_val=string))
        )
        return new_id

    def get_or_create_value_for(self, arg: _Argument):
        if isinstance(arg, Node):
            # If the Node has already been processed, return the existing id.
            if arg in self.node_to_value_ids:
                return self.node_to_value_ids[arg]
            return self.create_node_value(arg)
        elif (
            isinstance(arg, NoneType)
            or isinstance(arg, torch.device)
            or isinstance(arg, torch.dtype)
            or isinstance(arg, torch.layout)
            or isinstance(arg, torch.memory_format)
        ):
            return self.create_null_value()
        elif isinstance(arg, _ScalarType):
            return self.get_or_create_scalar_value(arg)
        elif isinstance(arg, TensorSpec):
            return self.create_tensor_value(arg)
        elif isinstance(arg, list) and (
            len(arg) == 0 or any(isinstance(val, _ScalarType) for val in arg)
        ):
            # pyre-ignore[6]
            return self.create_scalar_list_value(arg)
        elif isinstance(arg, list) and isinstance(arg[0], Node):
            return self.create_value_list_value(arg)
        elif isinstance(arg, torch.fx.immutable_collections.immutable_list):
            return self.create_value_list_value(arg)
        elif isinstance(arg, str):
            return self.create_string_value(arg)
        else:
            raise RuntimeError(f"Cannot create value for arg of type {type(arg)}")

    def process_placeholder_node(self, node: Node) -> None:
        # ignores any tensors that don't get used in any ops
        if len(node.users) == 0:
            return None
        ids = self.create_node_value(node)
        if not is_param_node(self.program, node):
            if isinstance(ids, int):
                self.input_ids.append(ids)
            else:
                self.input_ids += ids

    def process_getitem_node(self, node: Node) -> None:
        # Find ValueList id from the collection node.
        collection_node = node.all_input_nodes[0]
        list_id = self.node_to_value_ids[collection_node]

        # Extract the target Value id from ValueList.
        valuelist_id = node.args[1]
        value_id = self.values[list_id].value.items[valuelist_id]

        # Map Node to Value id.
        self.node_to_value_ids[node] = value_id

    def process_call_function_node(self, node) -> None:
        operator_call_args = []

        self.seen_ops.add(node.target)

        if hasattr(node.target, "_schema"):
            for i, schema_arg in enumerate(node.target._schema.arguments):
                if not schema_arg.kwarg_only and i < len(node.args):
                    function_arg = node.args[i]
                elif schema_arg.name in node.kwargs:
                    function_arg = node.kwargs[schema_arg.name]
                else:
                    function_arg = schema_arg.default_value

                # Create a Value for each function argument. If the argument has been
                # previously encountered, then use the existing Value id.
                operator_call_args.append(self.get_or_create_value_for(function_arg))
        else:
            for _, arg_node in enumerate(node.args):
                operator_call_args.append(self.get_or_create_value_for(arg_node))

        # Add output node
        operator_call_args.append(self.create_node_value(node))
        operator_node_id = (
            0
            if not self.delegate_mapping_builder
            else self.delegate_mapping_builder.insert_delegate_mapping_entry(node)
        )
        self.chain.append(
            vk_graph_schema.OperatorCall(
                node_id=operator_node_id,  # pyre-ignore[6]: this is going to be an int
                name=node.target.__name__,
                args=operator_call_args,
            ),
        )

    def process_getattr_node(self, node: Node) -> None:
        self.create_node_value(node)

    def process_output_node(self, node: Node) -> None:
        for out_node in node.all_input_nodes:
            if out_node not in self.node_to_value_ids:
                raise AssertionError(
                    "Cannot find input to output node in node_to_value_ids. This means "
                    "the output node is being serialized before its corresponding "
                    "internal node which is not allowed."
                )
            # Mutable buffers outputs are not included as an output to the
            # delegate call. Skip marking them as an output.
            if is_mutable_buffer_node(out_node, self.program):
                continue

            self.output_ids.append(self.node_to_value_ids[out_node])

    def process_node(self, node: Node, call_node_debug_hdl: int) -> None:
        if node.op == "placeholder":
            self.process_placeholder_node(node)
        elif node.op == "call_function":
            if node.target == operator.getitem:
                self.process_getitem_node(node)
            else:
                node.meta["debug_handle"] = call_node_debug_hdl
                self.process_call_function_node(node)
        elif node.op == "get_attr":
            self.process_getattr_node(node)
        elif node.op == "output":
            self.process_output_node(node)
        else:
            raise AssertionError(f"Unsupported node op: {node.op}")

    def build_graph(self) -> vk_graph_schema.VkGraph:
        call_node_debug_hdl = 0
        for node in self.program.graph_module.graph.nodes:
            self.process_node(node, call_node_debug_hdl)
            call_node_debug_hdl += 1

        logger.info("Operators included in this Vulkan partition: ")
        for op in self.seen_ops:
            logger.info(f"    {op.__name__}")

        return vk_graph_schema.VkGraph(
            version="0",
            chain=self.chain,
            values=self.values,
            input_ids=self.input_ids,
            output_ids=self.output_ids,
            constants=[],
            shaders=[],
        )
