//
// Created by Maxwell on 2020-09-29.
//

#include "status.h"

#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>

/*#include <llvm/ADT/DenseMap.h>*/

namespace {

  using namespace std::string_view_literals;

  class nvrtc_domain : public cu::status_domain{
  public:
    constexpr nvrtc_domain() noexcept
        : cu::status_domain(
              []()noexcept { return "nvRTC"sv; },
              [](int64_t Value) noexcept -> std::string_view {
                /*switch((nvrtcResult)Value){
                  case NVRTC_SUCCESS:
                    return "success";
                  case NVRTC_ERROR_OUT_OF_MEMORY:
                    return "out of memory";
                  case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
                    return "program creation failure";
                  case NVRTC_ERROR_INVALID_INPUT:
                    return "invalid input";
                  case NVRTC_ERROR_INVALID_PROGRAM:
                    return "invalid program";
                  case NVRTC_ERROR_INVALID_OPTION:
                    return "invalid option";
                  case NVRTC_ERROR_COMPILATION:
                    return "compilation error";
                  case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
                    return "builtin operation failure";
                  case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
                    return "no name expressions after compilation";
                  case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
                    return "no lowered named before compilation";
                  case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
                    return "name expression not valid";
                  case NVRTC_ERROR_INTERNAL_ERROR:
                    return "internal error";
                }*/
                return nvrtcGetErrorString((nvrtcResult)Value);
              },
              [](int64_t Value) noexcept -> cu::severity{
                switch((nvrtcResult)Value){
                  case NVRTC_SUCCESS:
                    return cu::severity::success;
                  case NVRTC_ERROR_OUT_OF_MEMORY:
                    return cu::severity::fatal;
                  case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
                  case NVRTC_ERROR_INVALID_INPUT:
                  case NVRTC_ERROR_INVALID_PROGRAM:
                  case NVRTC_ERROR_INVALID_OPTION:
                    return cu::severity::low;
                  case NVRTC_ERROR_COMPILATION:
                  case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
                  case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
                  case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
                  case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
                  case NVRTC_ERROR_INTERNAL_ERROR:
                    return cu::severity::high;
                }
              },
              [](int64_t Value) noexcept -> cu::generic_code{
                switch((nvrtcResult)Value){
                  case NVRTC_SUCCESS:
                    return cu::generic_code::success;
                  case NVRTC_ERROR_OUT_OF_MEMORY:
                    return cu::generic_code::not_enough_memory;

                  case NVRTC_ERROR_INVALID_INPUT:
                  case NVRTC_ERROR_INVALID_PROGRAM:
                  case NVRTC_ERROR_INVALID_OPTION:
                    return cu::generic_code::invalid_argument;
                  case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
                  case NVRTC_ERROR_INTERNAL_ERROR:
                    return cu::generic_code::state_not_recoverable;
                  case NVRTC_ERROR_COMPILATION:
                  case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
                  case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
                  case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
                  case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
                    return cu::generic_code::unknown;
                }
              }){}
  };
  class cuda_driver_domain : public cu::status_domain{
  public:
    constexpr cuda_driver_domain() noexcept
        : cu::status_domain(
              []()noexcept{ return "cuda driver"sv; },
              [](int64_t Value) noexcept -> std::string_view{
                const char* ErrorString;
                if(auto Error = cuGetErrorString((CUresult)Value, &ErrorString))
                  return "unknown cuda driver error";
                return ErrorString;
              },
              [](int64_t Value) noexcept -> cu::severity{
                switch((CUresult)Value){

                  case CUDA_SUCCESS:
                    return cu::severity::success;

                  case CUDA_ERROR_NOT_READY:
                  case CUDA_ERROR_TIMEOUT:
                    return cu::severity::info;

                  case CUDA_ERROR_PROFILER_ALREADY_STARTED:
                  case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
                  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
                  case CUDA_ERROR_ALREADY_ACQUIRED:
                  case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
                  case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
                  case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
                  case CUDA_ERROR_ARRAY_IS_MAPPED:
                  case CUDA_ERROR_ALREADY_MAPPED:
                  case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
                    return cu::severity::warning;

                  case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
                  case CUDA_ERROR_TOO_MANY_PEERS:
                  case CUDA_ERROR_UNSUPPORTED_LIMIT:
                  case CUDA_ERROR_NOT_INITIALIZED:
                  case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
                  case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
                  case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
                    return cu::severity::low;

                  case CUDA_ERROR_DEINITIALIZED:
                  case CUDA_ERROR_INVALID_DEVICE:
                  case CUDA_ERROR_INVALID_IMAGE:
                  case CUDA_ERROR_INVALID_CONTEXT:
                  case CUDA_ERROR_INVALID_PTX:
                  case CUDA_ERROR_INVALID_SOURCE:
                  case CUDA_ERROR_MAP_FAILED:
                  case CUDA_ERROR_UNMAP_FAILED:
                  case CUDA_ERROR_NO_BINARY_FOR_GPU:
                  case CUDA_ERROR_ECC_UNCORRECTABLE:
                  case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
                  case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
                  case CUDA_ERROR_OPERATING_SYSTEM:
                  case CUDA_ERROR_INVALID_HANDLE:
                  case CUDA_ERROR_ILLEGAL_STATE:
                  case CUDA_ERROR_NOT_FOUND:
                  case CUDA_ERROR_INVALID_VALUE:
                  case CUDA_ERROR_NOT_MAPPED:
                  case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
                  case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
                  case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
                  case CUDA_ERROR_NOT_PERMITTED:
                  case CUDA_ERROR_NOT_SUPPORTED:
                  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
                  case CUDA_ERROR_CONTEXT_IS_DESTROYED:
                    return cu::severity::high;

                  case CUDA_ERROR_PROFILER_DISABLED:
                  case CUDA_ERROR_OUT_OF_MEMORY:
                  case CUDA_ERROR_NO_DEVICE:
                  case CUDA_ERROR_HARDWARE_STACK_ERROR:
                  case CUDA_ERROR_ILLEGAL_INSTRUCTION:
                  case CUDA_ERROR_MISALIGNED_ADDRESS:
                  case CUDA_ERROR_INVALID_ADDRESS_SPACE:
                  case CUDA_ERROR_INVALID_PC:
                  case CUDA_ERROR_LAUNCH_FAILED:
                  case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
                  case CUDA_ERROR_NVLINK_UNCORRECTABLE:
                  case CUDA_ERROR_FILE_NOT_FOUND:
                  case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
                  case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
                  case CUDA_ERROR_ILLEGAL_ADDRESS:
                  case CUDA_ERROR_ASSERT:
                  case CUDA_ERROR_UNKNOWN:
                  case CUDA_ERROR_LAUNCH_TIMEOUT:
                  case CUDA_ERROR_SYSTEM_NOT_READY:
                  case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
                  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
                  case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
                  case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
                  case CUDA_ERROR_STREAM_CAPTURE_MERGE:
                  case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
                  case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
                  case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
                  case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
                  case CUDA_ERROR_CAPTURED_EVENT:
                  case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
                  case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
                    return cu::severity::fatal;
                  }
              },
              [](int64_t Value) noexcept -> cu::generic_code{
                switch((CUresult)Value){
                  case CUDA_SUCCESS:
                    return cu::generic_code::success;

                  case CUDA_ERROR_INVALID_VALUE:
                  case CUDA_ERROR_UNSUPPORTED_LIMIT:
                  case CUDA_ERROR_INVALID_HANDLE:
                  case CUDA_ERROR_INVALID_PTX:
                  case CUDA_ERROR_INVALID_SOURCE:
                  case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
                    return cu::generic_code::invalid_argument;

                  case CUDA_ERROR_OUT_OF_MEMORY:
                    return cu::generic_code::not_enough_memory;

                  case CUDA_ERROR_NOT_INITIALIZED:
                  case CUDA_ERROR_DEINITIALIZED:
                  case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
                    return cu::generic_code::not_initialized;

                  case CUDA_ERROR_INVALID_DEVICE:
                    return cu::generic_code::no_such_device;

                  case CUDA_ERROR_NO_DEVICE:
                  case CUDA_ERROR_ASSERT:
                  case CUDA_ERROR_HARDWARE_STACK_ERROR:
                  case CUDA_ERROR_ILLEGAL_INSTRUCTION:
                  case CUDA_ERROR_MISALIGNED_ADDRESS:
                  case CUDA_ERROR_INVALID_PC:
                  case CUDA_ERROR_LAUNCH_FAILED:
                  case CUDA_ERROR_INVALID_ADDRESS_SPACE:
                  case CUDA_ERROR_LAUNCH_TIMEOUT:
                    return cu::generic_code::state_not_recoverable;

                  case CUDA_ERROR_INVALID_IMAGE:
                    return cu::generic_code::executable_format_error;

                  case CUDA_ERROR_INVALID_CONTEXT:
                  case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
                  case CUDA_ERROR_ILLEGAL_STATE:
                  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
                  case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
                    return cu::generic_code::invalid_state;

                  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
                  case CUDA_ERROR_ALREADY_MAPPED:
                  case CUDA_ERROR_ALREADY_ACQUIRED:
                  case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
                  case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
                  case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
                    return cu::generic_code::resource_already_acquired;

                  case CUDA_ERROR_ARRAY_IS_MAPPED:
                    return cu::generic_code::still_in_use;

                  case CUDA_ERROR_NOT_MAPPED:
                  case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
                  case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
                  case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
                    return cu::generic_code::resource_not_acquired;


                  case CUDA_ERROR_ECC_UNCORRECTABLE:
                    return cu::generic_code::memory_corrupted;

                  case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
                    return cu::generic_code::device_or_resource_busy;
                  case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
                    return cu::generic_code::cross_device_link;

                  case CUDA_ERROR_NVLINK_UNCORRECTABLE:
                    return cu::generic_code::no_link;

                  case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
                  case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
                    return cu::generic_code::no_such_library;

                  case CUDA_ERROR_FILE_NOT_FOUND:
                    return cu::generic_code::no_such_file_or_directory;

                  case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
                    return cu::generic_code::library_corrupted;

                  case CUDA_ERROR_OPERATING_SYSTEM:
                    return (cu::generic_code)errno;

                  case CUDA_ERROR_NOT_FOUND:
                    return cu::generic_code::no_such_resource;

                  case CUDA_ERROR_NOT_READY:
                    return cu::generic_code::not_ready;

                  case CUDA_ERROR_ILLEGAL_ADDRESS:
                    return cu::generic_code::bad_address;

                  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
                    return cu::generic_code::no_space_on_device;


                  case CUDA_ERROR_TIMEOUT:
                    return cu::generic_code::timed_out;

                  case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
                    return cu::generic_code::not_connected;

                  case CUDA_ERROR_CONTEXT_IS_DESTROYED:
                    return cu::generic_code::resource_destroyed;

                  case CUDA_ERROR_TOO_MANY_PEERS:
                    return cu::generic_code::too_many_users;

                  case CUDA_ERROR_NOT_PERMITTED:
                    return cu::generic_code::permission_denied;

                  case CUDA_ERROR_NOT_SUPPORTED:
                  case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
                    return cu::generic_code::not_supported;

                  case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
                  case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
                    return cu::generic_code::incompatible_version;


                  case CUDA_ERROR_CAPTURED_EVENT:
                  case CUDA_ERROR_STREAM_CAPTURE_MERGE:
                  case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
                  case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
                  case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
                  case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
                    return cu::generic_code::operation_not_permitted;

                  case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
                    return cu::generic_code::resource_not_owned;

                  default:
                    return cu::generic_code::unknown;
                }
              }
              ){}
  };
  class cuda_runtime_domain : public cu::status_domain{
  public:
    constexpr cuda_runtime_domain() noexcept
        : cu::status_domain(
              []() noexcept { return "cuda runtime"sv;},
              [](int64_t Value) noexcept {
                return std::string_view(cudaGetErrorString((cudaError_t)Value));
              },
              [](int64_t Value) noexcept -> cu::severity{},
              [](int64_t Value) noexcept -> cu::generic_code{}
              ){}
  };

  inline constexpr nvrtc_domain nvrtc_domain_v{};
  inline constexpr cuda_driver_domain driver_domain_v{};
  inline constexpr cuda_runtime_domain runtime_domain_v{};
}

template <>
struct cu::status_enum<nvrtcResult>{
  static constexpr const nvrtc_domain& domain() noexcept{
    return nvrtc_domain_v;
  }
};
template <>
struct cu::status_enum<cudaError>{
  static constexpr const cuda_runtime_domain& domain() noexcept{
    return runtime_domain_v;
  }
};
template <>
struct cu::status_enum<CUresult>{
  static constexpr const cuda_driver_domain& domain() noexcept{
    return driver_domain_v;
  }
};





class cu::status_handler::action{
  class resolve_action;
  class callback_action;
  class propagate_action;
  class suppress_action;
  class ignore_action;
  class cancel_action;
  class retry_action;
  class kill_action;
  class terminate_action;


};
class cu::status_handler::action::resolve_action : public cu::status_handler::action{

};
class cu::status_handler::action::callback_action : public cu::status_handler::action{

};
class cu::status_handler::action::propagate_action : public cu::status_handler::action{

};
class cu::status_handler::action::suppress_action : public cu::status_handler::action{

};
class cu::status_handler::action::ignore_action : public cu::status_handler::action{

};
class cu::status_handler::action::cancel_action : public cu::status_handler::action{

};
class cu::status_handler::action::retry_action : public cu::status_handler::action{

};
class cu::status_handler::action::kill_action : public cu::status_handler::action{

};
class cu::status_handler::action::terminate_action : public cu::status_handler::action{

};
