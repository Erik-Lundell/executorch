/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __GNUC__
// Disable -Wdeprecated-declarations, as some builds use 'Werror'.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/merged_data_map.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/compiler.h>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace executorch_flatbuffer {
struct Chain;
struct ExecutionPlan;
struct EValue;
} // namespace executorch_flatbuffer

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

// Forward declare NamedData. This is a public header and must not include
// internal data types.
namespace deserialization {
struct NamedData;
} // namespace deserialization

// Forward declare Program to avoid a circular reference.
class Program;

// Forward declare internal types.
class BackendDelegate;
struct Chain;
class KernelRuntimeContext;
using OpFunction = void (*)(KernelRuntimeContext&, Span<EValue*>);
/// A list of pointers into the master values table that together compose the
/// argument list for a single instruction
using InstructionArgs = Span<EValue*>;
using deserialization::NamedData;

/**
 * An executable method of an executorch program. Maps to a python method like
 * `forward()` on the original nn.Module.
 */
class Method final {
 public:
  /**
   * Move ctor. Takes ownership of resources previously owned by `rhs`,
   * and leaves `rhs` in an uninitialized state.
   */
  Method(Method&& rhs) noexcept
      : step_state_(rhs.step_state_),
        program_(rhs.program_),
        memory_manager_(rhs.memory_manager_),
        temp_allocator_(rhs.temp_allocator_),
        serialization_plan_(rhs.serialization_plan_),
        event_tracer_(rhs.event_tracer_),
        n_value_(rhs.n_value_),
        values_(rhs.values_),
        n_delegate_(rhs.n_delegate_),
        delegates_(rhs.delegates_),
        n_chains_(rhs.n_chains_),
        chains_(rhs.chains_),
        merged_data_map_(std::move(rhs.merged_data_map_)),
        external_constants_(rhs.external_constants_),
        n_external_constants_(rhs.n_external_constants_),
        init_state_(rhs.init_state_) {
    // Required: clear out fields that the dtor looks at, so that we don't free
    // anything twice.
    rhs.n_value_ = 0;
    rhs.values_ = nullptr;
    rhs.n_delegate_ = 0;
    rhs.delegates_ = nullptr;

    rhs.merged_data_map_ = nullptr;
    rhs.n_external_constants_ = 0;
    rhs.external_constants_ = nullptr;

    // Helpful: Try to ensure that any other interactions with the old object
    // result in failures.
    rhs.init_state_ = InitializationState::Uninitialized;
    rhs.step_state_ = {};
    rhs.program_ = nullptr;
    rhs.memory_manager_ = nullptr;
    rhs.serialization_plan_ = nullptr;
    rhs.event_tracer_ = nullptr;
    rhs.n_chains_ = 0;
    rhs.chains_ = nullptr;
  }

  /**
   * Sets the internal input value to be equivalent to the to the provided
   * value.
   *
   * @param[in] input_evalue The evalue to copy into the method input. If the
   *     evalue is a tensor, the data is copied in most cases, so the tensor
   *     passed in here does not always need to outlive this call. But there is
   *     a case where the Method will keep a pointer to the tensor's data.
   *     Based on the memory plan of the method, the inputs may not have
   *     buffer space pre-allocated for them. In this case the executor will
   *     alias the memory of the tensors provided as inputs here rather then
   *     deepcopy the input into the memory planned arena.
   *
   * @param[in] input_idx Zero-based index of the input to set. Must be less
   *     than the value returned by inputs_size().
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  ET_NODISCARD Error set_input(const EValue& input_evalue, size_t input_idx);

  /**
   * Sets the values of all method inputs.
   *
   * See set_input() for a more detailed description of the behavior.
   *
   * @param[in] input_evalues The new values for all of the method inputs. The
   *     type of each element must match the type of corresponding input. If the
   *     value of an element is a tensor, attempts to allow dynamic shape, but
   *     the dtype must always agree.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  ET_NODISCARD Error
  set_inputs(const executorch::aten::ArrayRef<EValue>& input_evalues);

  /**
   * Sets the data buffer of the specified method output to the provided value.
   *
   * NOTE: Based on the memory plan of the method, the output tensors may not
   * have buffer space pre-allocated for them, in this case the executor will
   * point those tensors to the buffer provided here, so the user should take
   * care that the life span of this memory outlasts the executor forward.
   *
   * @param[in] buffer The block of memory to point the specified tensor at.
   *
   * @param[in] size the length of buffer in bytes, must be >= the nbytes of the
   * specified tensor.
   *
   * @param[in] output_idx The index of the output to set the data_ptr for. Must
   *     correspond to a tensor, and that tensor must not have had a buffer
   *     allocated by the memory plan.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  ET_NODISCARD Error
  set_output_data_ptr(void* buffer, size_t size, size_t output_idx);

  /**
   * Copies the method's outputs into the provided array.
   *
   * WARNING: The output contains shallow copies of internal tensor outputs.
   * Please do not mutate returned Tensor elements.
   *
   * TODO(T139259264): Add checks to detect output mutation, or deep-copy
   * outputs.
   *
   * @param[in] output_evalues The array to copy the outputs into. The first
   *     `outputs_size()` elements will be set to the corresponding output
   *     values. The rest of the array will be set to the EValue value None.
   * @param[in] length The size of the `output_evalues` array in elements. Must
   *     be greater than or equal to `outputs_size()`.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  ET_NODISCARD Error get_outputs(EValue* output_evalues, size_t length);

  /**
   * Copies the method's inputs into the provided array.
   *
   * WARNING: The input contains shallow copies of internal tensor inputs.
   * Please do not mutate returned Tensor elements.
   *
   * @param[in] input_evalues The array to copy the inputs into. The first
   *     `inputs_size()` elements will be set to the corresponding input
   *     values. The rest of the array will be set to the EValue value None.
   * @param[in] length The size of the `input_evalues` array in elements. Must
   *     be greater than or equal to `inputs_size()`.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  ET_NODISCARD Error get_inputs(EValue* input_evalues, size_t length);

  /**
   *
   * Retrieves the attribute tensor associated with the given name.
   *
   * @param[in] name The name of the attribute tensor to retrieve.
   *
   * @returns Result containing the attribute tensor on success, non-Ok on
   * failure.
   */
  ET_NODISCARD Result<executorch::aten::Tensor> get_attribute(
      std::string_view name);

  /**
   * Execute the method.
   *
   * NOTE: Will fail if the method has been partially executed using the
   * `step()` api.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  ET_NODISCARD Error execute();

  /**
   * EXPERIMENTAL: Advances/executes a single instruction in the method.
   *
   * @retval Error::Ok step succeeded
   * @retval non-Ok step failed
   * @retval Error::EndOfMethod method finished executing successfully
   */
  ET_EXPERIMENTAL ET_NODISCARD Error step();

  /// DEPRECATED: Use `step()` instead.
  ET_DEPRECATED ET_NODISCARD Error experimental_step();

  /**
   * EXPERIMENTAL: Resets execution state to the start of the Method. For use
   * with the `step()` API.
   *
   * @retval Error:Ok on success
   * @retval Error::InvalidState if called before step-based execution reached
   *     the end of the Method. This means it is not possible to recover a
   *     Method that failed mid-execution.
   */
  ET_EXPERIMENTAL ET_NODISCARD Error reset_execution();

  /// DEPRECATED: Use `reset_execution()` instead.
  ET_DEPRECATED ET_NODISCARD Error experimental_reset_execution();

  /**
   * Returns the MethodMeta that corresponds to the calling Method.
   */
  MethodMeta method_meta() const;

  /**
   * Returns the number of inputs the Method expects.
   */
  size_t inputs_size() const;

  /**
   * Returns the number of outputs the Method returns.
   */
  size_t outputs_size() const;

  /**
   * Retrieves the output at the specified index.
   */
  const EValue& get_output(size_t i) const;

  EventTracer* get_event_tracer();

  /// DEPRECATED: Use MethodMeta instead to access metadata, and set_input to
  /// update Method inputs.
  ET_DEPRECATED const EValue& get_input(size_t i) const;
  /// DEPRECATED: Use MethodMeta instead to access metadata, and set_input to
  /// update Method inputs.
  ET_DEPRECATED EValue& mutable_input(size_t i);
  /// DEPRECATED: Use MethodMeta instead to access metadata, and get_output to
  /// retrieve Method outputs.
  ET_DEPRECATED EValue& mutable_output(size_t i);

  ~Method();

 private:
  // Delete other rule-of-five methods.
  Method(const Method&) = delete;
  Method& operator=(const Method&) noexcept = delete;
  Method& operator=(Method&&) = delete;

  // Let Program call load().
  friend class Program;
  // Let Executor call the ctor and init().
  friend class Executor;

  enum class InitializationState : uint8_t {
    Uninitialized,
    Initialized,
    InitializationFailed,
  };

  /// Tracks what step in program execution we are on
  struct StepState {
    size_t chain_idx;
    size_t instr_idx;
  };

  Method(
      const Program* program,
      MemoryManager* memory_manager,
      EventTracer* event_tracer,
      MemoryAllocator* temp_allocator)
      : step_state_(),
        program_(program),
        memory_manager_(memory_manager),
        temp_allocator_(temp_allocator),
        serialization_plan_(nullptr),
        event_tracer_(event_tracer),
        n_value_(0),
        values_(nullptr),
        n_delegate_(0),
        delegates_(nullptr),
        n_chains_(0),
        chains_(nullptr),
        merged_data_map_(nullptr),
        external_constants_(nullptr),
        n_external_constants_(0),
        init_state_(InitializationState::Uninitialized) {}

  /// Static factory used by Program.
  ET_NODISCARD static Result<Method> load(
      executorch_flatbuffer::ExecutionPlan* s_plan,
      const Program* program,
      MemoryManager* memory_manager,
      EventTracer* event_tracer,
      const NamedDataMap* named_data_map);

  /**
   * Initialize the method from its serialized representation.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  ET_NODISCARD Error init(
      executorch_flatbuffer::ExecutionPlan* s_plan,
      const NamedDataMap* named_data_map);

  /// Returns true if the Method was successfully initialized.
  inline bool initialized() const {
    return init_state_ == InitializationState::Initialized;
  }

  const EValue& get_value(size_t i) const;
  EValue& mutable_value(size_t i);
  size_t get_input_index(size_t i) const;
  size_t get_output_index(size_t i) const;

  // Executes a single instruction using the state in step_state_
  ET_NODISCARD Error execute_instruction();

  StepState step_state_;
  const Program* program_;
  MemoryManager* memory_manager_;
  MemoryAllocator* temp_allocator_;
  executorch_flatbuffer::ExecutionPlan* serialization_plan_;
  EventTracer* event_tracer_;

  size_t n_value_;
  EValue* values_;

  size_t n_delegate_;
  BackendDelegate* delegates_;

  size_t n_chains_;
  Chain* chains_;

  internal::MergedDataMap* merged_data_map_;
  NamedData* external_constants_;
  size_t n_external_constants_ = 0;

  InitializationState init_state_;

  /**
   * Counts the number of tensors marked as EXTERNAL in the flatbuffer
   * for this method.
   */
  ET_NODISCARD Result<size_t> get_num_external_constants();

  /**
   * Parses the flatbuffer for constant tensors tagged as EXTERNAL.
   * Retrieves the external constants using the named_data_map and places them
   * into `external_constants_`. Updates `n_external_constants_` to count the
   * number of successfully-initialized external constants.
   * FreeableBuffers returned by the named_data_map are owned by the
   * method and are freed on method destruction.
   *
   * @param[in] named_data_map, to retrieve external constants from.
   * @returns Error::Ok on success, non-Ok on failure.
   */
  ET_NODISCARD Error
  parse_external_constants(const NamedDataMap* named_data_map);

  /**
   * Parses the elements of the values_ array. On error, n_value_ will be set to
   * the number of successfully-initialized entries so that ~Method doesn't try
   * to clean up uninitialized entries.
   */
  ET_NODISCARD Error parse_values(const NamedDataMap* named_data_map);

  ET_NODISCARD Error resolve_operator(
      int32_t op_index,
      OpFunction* kernels,
      size_t kernel_index,
      InstructionArgs args,
      size_t n_args);

  void log_outputs();
};

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::ET_RUNTIME_NAMESPACE::Method;
} // namespace executor
} // namespace torch

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
