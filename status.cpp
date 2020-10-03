//
// Created by Maxwell on 2020-09-29.
//

#include "include/util/status.h"

#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>

/*#include <llvm/ADT/DenseMap.h>*/

namespace {

  using namespace std::string_view_literals;

  class generic_domain : public cu::status_domain{
  public:
    constexpr generic_domain() noexcept
        : cu::status_domain(
              []()noexcept{ return "generic"sv; },
              [](int64_t Value)noexcept->std::string_view{
                switch((cu::generic_code)Value){
                  case cu::generic_code::success: return "success";
                  case cu::generic_code::address_family_not_supported: return "address family not supported"; // EAFNOSUPPORT
                  case cu::generic_code::address_in_use: return "address in use"; // EADDRINUSE
                  case cu::generic_code::address_not_available: return "address not available"; // EADDRNOTAVAIL
                  case cu::generic_code::already_connected: return "already connected"; // EISCONN
                  case cu::generic_code::argument_list_too_long: return "argument list too long"; // E2BIG
                  case cu::generic_code::argument_out_of_domain: return "argument out of domain"; // EDOM
                  case cu::generic_code::bad_address: return "bad address"; // EFAULT
                  case cu::generic_code::bad_file_descriptor: return "bad file descriptor"; // EBADF
                  case cu::generic_code::bad_message: return "bad message"; // EBADMSG
                  case cu::generic_code::broken_pipe: return "broken pipe"; // EPIPE
                  case cu::generic_code::connection_aborted: return "connection aborted"; // ECONNABORTED
                  case cu::generic_code::connection_already_in_progress: return "connection already in progress"; // EALREADY
                  case cu::generic_code::connection_refused: return "connection refused"; // ECONNREFUSED
                  case cu::generic_code::connection_reset: return "connection reset"; // ECONNRESET
                  case cu::generic_code::cross_device_link: return "cross device link"; // EXDEV
                  case cu::generic_code::destination_address_required: return "destination address required"; // EDESTADDRREQ
                  case cu::generic_code::device_or_resource_busy: return "device or resource busy"; // EBUSY
                  case cu::generic_code::directory_not_empty: return "directory not empty"; // ENOTEMPTY
                  case cu::generic_code::executable_format_error: return "executable format error"; // ENOEXEC
                  case cu::generic_code::file_exists: return "file exists"; // EEXIST
                  case cu::generic_code::file_too_large: return "file too large"; // EFBIG
                  case cu::generic_code::filename_too_long: return "filename too long"; // ENAMETOOLONG
                  case cu::generic_code::function_not_supported: return "function not supported"; // ENOSYS
                  case cu::generic_code::host_unreachable: return "host unreachable"; // EHOSTUNREACH
                  case cu::generic_code::identifier_removed: return "identifier removed"; // EIDRM
                  case cu::generic_code::illegal_byte_sequence: return "illegal byte sequence"; // EILSEQ
                  case cu::generic_code::inappropriate_io_control_operation: return "inappropriate io control operation"; // ENOTTY
                  case cu::generic_code::interrupted: return "interrupted"; // EINTR
                  case cu::generic_code::invalid_argument: return "invalid argument"; // EINVAL
                  case cu::generic_code::invalid_seek: return "invalid seek"; // ESPIPE
                  case cu::generic_code::io_error: return "io error"; // EIO
                  case cu::generic_code::is_a_directory: return "is a directory"; // EISDIR
                  case cu::generic_code::message_size: return "message size"; // EMSGSIZE
                  case cu::generic_code::network_down: return "network down"; // ENETDOWN
                  case cu::generic_code::network_reset: return "network reset"; // ENETRESET
                  case cu::generic_code::network_unreachable: return "network unreachable"; // ENETUNREACH
                  case cu::generic_code::no_buffer_space: return "no buffer space"; // ENOBUFS
                  case cu::generic_code::no_child_process: return "no child process"; // ECHILD
                  case cu::generic_code::no_link: return "no link"; // ENOLINK
                  case cu::generic_code::no_lock_available: return "no lock available"; // ENOLCK
                  case cu::generic_code::no_message_available: return "no message available"; // ENODATA
                  case cu::generic_code::no_message: return "no message"; // ENOMSG
                  case cu::generic_code::no_protocol_option: return "no protocol option"; // ENOPROTOOPT
                  case cu::generic_code::no_space_on_device: return "no space on device"; // ENOSPC
                  case cu::generic_code::no_stream_resources: return "no stream resources"; // ENOSR
                  case cu::generic_code::no_such_device_or_address: return "no such device or address"; // ENXIO
                  case cu::generic_code::no_such_device: return "no such device"; // ENODEV
                  case cu::generic_code::no_such_file_or_directory: return "no such file or directory"; // ENOENT
                  case cu::generic_code::no_such_process: return "no such process"; // ESRCH
                  case cu::generic_code::not_a_directory: return "not a directory"; // ENOTDIR
                  case cu::generic_code::not_a_socket: return "not a socket"; // ENOTSOCK
                  case cu::generic_code::not_a_stream: return "not a stream"; // ENOSTR
                  case cu::generic_code::not_connected: return "not connected"; // ENOTCONN
                  case cu::generic_code::not_enough_memory: return "not enough memory"; // ENOMEM
                  case cu::generic_code::not_supported: return "not supported"; // ENOTSUP
                  case cu::generic_code::operation_canceled: return "operation canceled"; // ECANCELED
                  case cu::generic_code::operation_in_progress: return "operation in progress"; // EINPROGRESS
                  case cu::generic_code::operation_not_permitted: return "operation not permitted"; // EPERM
                  case cu::generic_code::operation_not_supported: return "operation not supported"; // EOPNOTSUPP
                  case cu::generic_code::operation_would_block: return "operation would block"; // EWOULDBLOCK
                  case cu::generic_code::owner_dead: return "owner dead"; // EOWNERDEAD
                  case cu::generic_code::permission_denied: return "permission denied"; // EACCES
                  case cu::generic_code::protocol_error: return "protocol error"; // EPROTO
                  case cu::generic_code::protocol_not_supported: return "protocol not supported"; // EPROTONOSUPPORT
                  case cu::generic_code::read_only_file_system: return "read only file system"; // EROFS
                  case cu::generic_code::resource_deadlock_would_occur: return "resource deadlock would occur"; // EDEADLK
                  case cu::generic_code::resource_unavailable_try_again: return "resource unavailable try again"; // EAGAIN
                  case cu::generic_code::result_out_of_range: return "result out of range"; // ERANGE
                  case cu::generic_code::state_not_recoverable: return "state not recoverable"; // ENOTRECOVERABLE
                  case cu::generic_code::stream_timeout: return "stream timeout"; // ETIME
                  case cu::generic_code::text_file_busy: return "text file busy"; // ETXTBSY
                  case cu::generic_code::timed_out: return "timed out"; // ETIMEDOUT
                  case cu::generic_code::too_many_files_open_in_system: return "too many files open in system"; // ENFILE
                  case cu::generic_code::too_many_files_open: return "too many files open"; // EMFILE
                  case cu::generic_code::too_many_links: return "too many links"; // EMLINK
                  case cu::generic_code::too_many_symbolic_link_levels: return "too many symbolic link levels"; // ELOOP
                  case cu::generic_code::too_many_users: return "too many users";  // EUSERS
                  case cu::generic_code::value_too_large: return "value too large"; // EOVERFLOW
                  case cu::generic_code::wrong_protocol_type: return "wrong protocol type"; // EPROTOTYPE
                  case cu::generic_code::not_initialized: return "not initialized";
                  case cu::generic_code::invalid_state: return "invalid state";
                  case cu::generic_code::resource_already_acquired: return "resource already acquired";
                  case cu::generic_code::resource_not_acquired: return "resource not acquired";
                  case cu::generic_code::resource_destroyed: return "resource destroyed";
                  case cu::generic_code::no_such_resource: return "no such resource";
                  case cu::generic_code::resource_not_owned: return "resource not owned";
                  case cu::generic_code::still_in_use: return "still in use";
                  case cu::generic_code::not_ready: return "not ready";
                  case cu::generic_code::incompatible_version: return "incompatible version";
                  case cu::generic_code::memory_corrupted: return "memory corrupted";
                  case cu::generic_code::no_such_library: return "no such library";
                  case cu::generic_code::library_corrupted: return "library corrupted";
                  default:
                    return "unknown";
                }

              },
              [](int64_t Value)noexcept->cu::severity{
                switch((cu::generic_code)Value){

                  case cu::generic_code::success:
                    return cu::severity::success;

                  case cu::generic_code::not_ready:
                  case cu::generic_code::timed_out:
                    return cu::severity::info;


                  case cu::generic_code::unknown:
                  case cu::generic_code::address_family_not_supported:
                  case cu::generic_code::address_in_use:
                  case cu::generic_code::address_not_available:
                  case cu::generic_code::already_connected:
                  case cu::generic_code::argument_list_too_long:
                  case cu::generic_code::argument_out_of_domain:
                  case cu::generic_code::bad_address:
                  case cu::generic_code::bad_file_descriptor:
                  case cu::generic_code::bad_message:
                  case cu::generic_code::broken_pipe:
                  case cu::generic_code::connection_aborted:
                  case cu::generic_code::connection_already_in_progress:
                  case cu::generic_code::connection_refused:
                  case cu::generic_code::connection_reset:
                  case cu::generic_code::cross_device_link:
                  case cu::generic_code::destination_address_required:
                  case cu::generic_code::device_or_resource_busy:
                  case cu::generic_code::directory_not_empty:
                  case cu::generic_code::executable_format_error:
                  case cu::generic_code::file_exists:
                  case cu::generic_code::file_too_large:
                  case cu::generic_code::filename_too_long:
                  case cu::generic_code::function_not_supported:
                  case cu::generic_code::host_unreachable:
                  case cu::generic_code::identifier_removed:
                  case cu::generic_code::illegal_byte_sequence:
                  case cu::generic_code::inappropriate_io_control_operation:
                  case cu::generic_code::interrupted:
                  case cu::generic_code::invalid_argument:
                  case cu::generic_code::invalid_seek:
                  case cu::generic_code::io_error:
                  case cu::generic_code::is_a_directory:
                  case cu::generic_code::message_size:
                  case cu::generic_code::network_down:
                  case cu::generic_code::network_reset:
                  case cu::generic_code::network_unreachable:
                  case cu::generic_code::no_buffer_space:
                  case cu::generic_code::no_child_process:
                  case cu::generic_code::no_link:
                  case cu::generic_code::no_lock_available:
                  case cu::generic_code::no_message_available:
                  case cu::generic_code::no_message:
                  case cu::generic_code::no_protocol_option:
                  case cu::generic_code::no_space_on_device:
                  case cu::generic_code::no_stream_resources:
                  case cu::generic_code::no_such_device_or_address:
                  case cu::generic_code::no_such_device:
                  case cu::generic_code::no_such_file_or_directory:
                  case cu::generic_code::no_such_process:
                  case cu::generic_code::not_a_directory:
                  case cu::generic_code::not_a_socket:
                  case cu::generic_code::not_a_stream:
                  case cu::generic_code::not_connected:
                  case cu::generic_code::not_enough_memory:
                  case cu::generic_code::not_supported:
                  case cu::generic_code::operation_canceled:
                  case cu::generic_code::operation_in_progress:
                  case cu::generic_code::operation_not_permitted:
                  case cu::generic_code::operation_not_supported:
                  case cu::generic_code::operation_would_block:
                  case cu::generic_code::owner_dead:
                  case cu::generic_code::permission_denied:
                  case cu::generic_code::protocol_error:
                  case cu::generic_code::protocol_not_supported:
                  case cu::generic_code::read_only_file_system:
                  case cu::generic_code::resource_deadlock_would_occur:
                  case cu::generic_code::resource_unavailable_try_again:
                  case cu::generic_code::result_out_of_range:
                  case cu::generic_code::state_not_recoverable:
                  case cu::generic_code::stream_timeout:
                  case cu::generic_code::text_file_busy:
                  case cu::generic_code::too_many_files_open_in_system:
                  case cu::generic_code::too_many_files_open:
                  case cu::generic_code::too_many_links:
                  case cu::generic_code::too_many_symbolic_link_levels:
                  case cu::generic_code::too_many_users:
                  case cu::generic_code::value_too_large:
                  case cu::generic_code::wrong_protocol_type:
                  case cu::generic_code::not_initialized:
                  case cu::generic_code::invalid_state:
                  case cu::generic_code::resource_already_acquired:
                  case cu::generic_code::resource_not_acquired:
                  case cu::generic_code::resource_destroyed:
                  case cu::generic_code::no_such_resource:
                  case cu::generic_code::resource_not_owned:
                  case cu::generic_code::still_in_use:

                  case cu::generic_code::incompatible_version:
                  case cu::generic_code::memory_corrupted:
                  case cu::generic_code::no_such_library:
                  case cu::generic_code::library_corrupted:
                  default:
                    return cu::severity::fatal;
                }
              },
              [](int64_t Value)noexcept->cu::generic_code{
                return (cu::generic_code)Value;
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
              [](int64_t Value) noexcept -> cu::severity{
                switch((cudaError_t)Value){

                  case cudaSuccess:
                    return cu::severity::success;


                  case cudaErrorNotReady:
                  case cudaErrorTimeout:
                    return cu::severity::info;



                  case cudaErrorProfilerAlreadyStarted:
                  case cudaErrorProfilerAlreadyStopped:
                  case cudaErrorAlreadyMapped:
                  case cudaErrorAlreadyAcquired:
                  case cudaErrorHostMemoryAlreadyRegistered:
                    return cu::severity::warning;



                  case cudaErrorProfilerNotInitialized:
                  case cudaErrorInitializationError:
                  case cudaErrorDevicesUnavailable:
                    return cu::severity::low;



                  case cudaErrorInvalidConfiguration:
                  case cudaErrorCudartUnloading:
                  case cudaErrorInvalidValue:
                  case cudaErrorInvalidPitchValue:
                  case cudaErrorInvalidSymbol:
                  case cudaErrorInvalidHostPointer:
                  case cudaErrorInvalidDevicePointer:
                  case cudaErrorInvalidTexture:
                  case cudaErrorInvalidTextureBinding:
                  case cudaErrorInvalidChannelDescriptor:
                  case cudaErrorInvalidMemcpyDirection:
                  case cudaErrorInvalidSurface:
                  case cudaErrorMissingConfiguration:
                  case cudaErrorLaunchMaxDepthExceeded:
                  case cudaErrorIllegalAddress:
                    return cu::severity::high;



                  case cudaErrorUnknown:
                  case cudaErrorApiFailureBase:
                  case cudaErrorMemoryAllocation:
                  case cudaErrorProfilerDisabled:
                  case cudaErrorInvalidFilterSetting:
                  case cudaErrorInvalidNormSetting:
                  case cudaErrorInsufficientDriver:
                  case cudaErrorDuplicateVariableName:
                  case cudaErrorDuplicateTextureName:
                  case cudaErrorDuplicateSurfaceName:
                  case cudaErrorLaunchFileScopedTex:
                  case cudaErrorLaunchFileScopedSurf:
                  case cudaErrorSystemNotReady:
                  case cudaErrorSystemDriverMismatch:

                  case cudaErrorStreamCaptureUnsupported:
                  case cudaErrorStreamCaptureInvalidated:
                  case cudaErrorStreamCaptureMerge:
                  case cudaErrorStreamCaptureUnmatched:
                  case cudaErrorStreamCaptureUnjoined:
                  case cudaErrorStreamCaptureIsolation:
                  case cudaErrorStreamCaptureImplicit:
                  case cudaErrorCapturedEvent:
                  case cudaErrorStreamCaptureWrongThread:

                  case cudaErrorECCUncorrectable:

                  case cudaErrorHardwareStackError:
                  case cudaErrorIllegalInstruction:
                  case cudaErrorMisalignedAddress:
                  case cudaErrorInvalidAddressSpace:
                  case cudaErrorInvalidPc:
                  case cudaErrorLaunchFailure:

                    return cu::severity::fatal;



                  case cudaErrorAddressOfConstant:
                  case cudaErrorTextureFetchFailed:
                  case cudaErrorTextureNotBound:
                  case cudaErrorSynchronizationError:
                  case cudaErrorMixedDeviceExecution:
                  case cudaErrorNotYetImplemented:
                  case cudaErrorMemoryValueTooLarge:
                  case cudaErrorIncompatibleDriverContext:
                  case cudaErrorPriorLaunchFailure:
                    break;



                    break;
                  case cudaErrorSyncDepthExceeded:
                    break;
                  case cudaErrorLaunchPendingCountExceeded:
                    break;
                  case cudaErrorInvalidDeviceFunction:
                    break;
                  case cudaErrorNoDevice:
                    break;
                  case cudaErrorInvalidDevice:
                    break;
                  case cudaErrorStartupFailure:
                    break;
                  case cudaErrorInvalidKernelImage:
                    break;
                  case cudaErrorDeviceUninitialized:
                    break;
                  case cudaErrorMapBufferObjectFailed:
                    break;
                  case cudaErrorUnmapBufferObjectFailed:
                    break;
                  case cudaErrorArrayIsMapped:
                  case cudaErrorNoKernelImageForDevice:

                  case cudaErrorNotMapped:
                  case cudaErrorNotMappedAsArray:
                  case cudaErrorNotMappedAsPointer:
                    break;

                    break;
                  case cudaErrorUnsupportedLimit:
                    break;
                  case cudaErrorDeviceAlreadyInUse:
                    break;
                  case cudaErrorPeerAccessUnsupported:
                    break;
                  case cudaErrorInvalidPtx:
                    break;
                  case cudaErrorInvalidGraphicsContext:
                    break;
                  case cudaErrorNvlinkUncorrectable:
                    break;
                  case cudaErrorJitCompilerNotFound:
                    break;
                  case cudaErrorInvalidSource:
                    break;
                  case cudaErrorFileNotFound:
                    break;
                  case cudaErrorSharedObjectSymbolNotFound:
                    break;
                  case cudaErrorSharedObjectInitFailed:
                    break;
                  case cudaErrorOperatingSystem:
                    break;
                  case cudaErrorInvalidResourceHandle:
                    break;
                  case cudaErrorIllegalState:
                    break;
                  case cudaErrorSymbolNotFound:
                    break;

                  case cudaErrorLaunchOutOfResources:
                    break;
                  case cudaErrorLaunchTimeout:
                    break;
                  case cudaErrorLaunchIncompatibleTexturing:
                    break;
                  case cudaErrorPeerAccessAlreadyEnabled:
                    break;
                  case cudaErrorPeerAccessNotEnabled:
                    break;
                  case cudaErrorSetOnActiveProcess:
                    break;
                  case cudaErrorContextIsDestroyed:
                    break;
                  case cudaErrorAssert:
                    break;
                  case cudaErrorTooManyPeers:
                    break;

                    break;
                  case cudaErrorHostMemoryNotRegistered:
                    break;

                    break;
                  case cudaErrorCooperativeLaunchTooLarge:
                    break;
                  case cudaErrorNotPermitted:
                    break;
                  case cudaErrorNotSupported:
                    break;

                    break;
                  case cudaErrorCompatNotSupportedOnDevice:
                  case cudaErrorGraphExecUpdateFailure:
                    break;

                }
              },
              [](int64_t Value) noexcept -> cu::generic_code{}
              ){}
  };


  inline constexpr cuda_driver_domain driver_domain_v{};
  inline constexpr cuda_runtime_domain runtime_domain_v{};
  inline constexpr generic_domain generic_domain_v{};
}

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

const cu::status_domain& cu::status_enum<cu::generic_code>::domain() noexcept {
  return generic_domain_v;
}





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
