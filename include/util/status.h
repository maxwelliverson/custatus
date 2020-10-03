//
// Created by maxwell on 2020-09-27.
//

#ifndef CUDA_FUNCTIONS_STATUS_H
#define CUDA_FUNCTIONS_STATUS_H

#include "contracts.h"

#include <string_view>
#include <system_error>

namespace cu{
  template <typename Sig>
  class dual_function;
  template <typename Ret, typename ...Args>
  class dual_function<Ret(Args...)>{
    Ret(*DevicePtr)(Args...);
    Ret(*HostPtr)(Args...);
  public:

  };

  enum class scope{
    block,
    thread,
    process
  };
  enum class severity{
    success,
    info,
    warning,
    low,
    high,
    fatal
  };
  enum class generic_code{
    success = 0,
    unknown = -1,
    address_family_not_supported       = 102, // EAFNOSUPPORT
    address_in_use                     = 100, // EADDRINUSE
    address_not_available              = 101, // EADDRNOTAVAIL
    already_connected                  = 113, // EISCONN
    argument_list_too_long             = 7, // E2BIG
    argument_out_of_domain             = 33, // EDOM
    bad_address                        = 14, // EFAULT
    bad_file_descriptor                = 9, // EBADF
    bad_message                        = 104, // EBADMSG
    broken_pipe                        = 32, // EPIPE
    connection_aborted                 = 106, // ECONNABORTED
    connection_already_in_progress     = 103, // EALREADY
    connection_refused                 = 107, // ECONNREFUSED
    connection_reset                   = 108, // ECONNRESET
    cross_device_link                  = 18, // EXDEV
    destination_address_required       = 109, // EDESTADDRREQ
    device_or_resource_busy            = 16, // EBUSY
    directory_not_empty                = 41, // ENOTEMPTY
    executable_format_error            = 8, // ENOEXEC
    file_exists                        = 17, // EEXIST
    file_too_large                     = 27, // EFBIG
    filename_too_long                  = 38, // ENAMETOOLONG
    function_not_supported             = 40, // ENOSYS
    host_unreachable                   = 110, // EHOSTUNREACH
    identifier_removed                 = 111, // EIDRM
    illegal_byte_sequence              = 42, // EILSEQ
    inappropriate_io_control_operation = 25, // ENOTTY
    interrupted                        = 4, // EINTR
    invalid_argument                   = 22, // EINVAL
    invalid_seek                       = 29, // ESPIPE
    io_error                           = 5, // EIO
    is_a_directory                     = 21, // EISDIR
    message_size                       = 115, // EMSGSIZE
    network_down                       = 116, // ENETDOWN
    network_reset                      = 117, // ENETRESET
    network_unreachable                = 118, // ENETUNREACH
    no_buffer_space                    = 119, // ENOBUFS
    no_child_process                   = 10, // ECHILD
    no_link                            = 121, // ENOLINK
    no_lock_available                  = 39, // ENOLCK
    no_message_available               = 120, // ENODATA
    no_message                         = 122, // ENOMSG
    no_protocol_option                 = 123, // ENOPROTOOPT
    no_space_on_device                 = 28, // ENOSPC
    no_stream_resources                = 124, // ENOSR
    no_such_device_or_address          = 6, // ENXIO
    no_such_device                     = 19, // ENODEV
    no_such_file_or_directory          = 2, // ENOENT
    no_such_process                    = 3, // ESRCH
    not_a_directory                    = 20, // ENOTDIR
    not_a_socket                       = 128, // ENOTSOCK
    not_a_stream                       = 125, // ENOSTR
    not_connected                      = 126, // ENOTCONN
    not_enough_memory                  = 12, // ENOMEM
    not_supported                      = 129, // ENOTSUP
    operation_canceled                 = 105, // ECANCELED
    operation_in_progress              = 112, // EINPROGRESS
    operation_not_permitted            = 1, // EPERM
    operation_not_supported            = 130, // EOPNOTSUPP
    operation_would_block              = 140, // EWOULDBLOCK
    owner_dead                         = 133, // EOWNERDEAD
    permission_denied                  = 13, // EACCES
    protocol_error                     = 134, // EPROTO
    protocol_not_supported             = 135, // EPROTONOSUPPORT
    read_only_file_system              = 30, // EROFS
    resource_deadlock_would_occur      = 36, // EDEADLK
    resource_unavailable_try_again     = 11, // EAGAIN
    result_out_of_range                = 34, // ERANGE
    state_not_recoverable              = 127, // ENOTRECOVERABLE
    stream_timeout                     = 137, // ETIME
    text_file_busy                     = 139, // ETXTBSY
    timed_out                          = 138, // ETIMEDOUT
    too_many_files_open_in_system      = 23, // ENFILE
    too_many_files_open                = 24, // EMFILE
    too_many_links                     = 31, // EMLINK
    too_many_symbolic_link_levels      = 114, // ELOOP
    too_many_users                     = 87,  // EUSERS
    value_too_large                    = 132, // EOVERFLOW
    wrong_protocol_type                = 136, // EPROTOTYPE

    not_initialized                    = 141,
    invalid_state,

    resource_already_acquired,
    resource_not_acquired,
    resource_destroyed,
    no_such_resource,
    resource_not_owned,

    still_in_use,
    not_ready,

    incompatible_version,
    memory_corrupted,

    no_such_library,
    library_corrupted
  };

  inline constexpr bool operator==(generic_code A, generic_code B) noexcept{
    if((int)A == -1 || (int)B == -1)
      return false;
    return (int)A == (int)B;
  }


  class status_code;
  class status_domain;

  template <typename T>
  struct status_enum;
  template <typename T>
  concept status_code_enum = requires(const status_domain* Domain){
    { Domain = std::addressof(status_enum<T>::domain()) } noexcept;
  };

  class status_domain{
    inline constexpr static uint64_t hash_string_view(std::string_view StringView) noexcept{
      uint64_t Val = 14695981039346656037ULL;
      for(char C : StringView){
        Val ^= static_cast<uint64_t>(C);
        Val *= 1099511628211ULL;
      }
      return Val;
    }
  protected:
    friend class status_code;

    using name_fn_t = std::string_view(*const)() noexcept;
    using message_fn_t = std::string_view(*const)(int64_t) noexcept;
    using severity_fn_t = severity(*const)(int64_t) noexcept;
    using generic_code_fn_t = generic_code(*const)(int64_t) noexcept;

    const uint64_t Id;
    name_fn_t Name;
    message_fn_t Message;
    severity_fn_t Severity;
    generic_code_fn_t GenericCode;

    constexpr status_domain(name_fn_t Name, message_fn_t Message, severity_fn_t Severity, generic_code_fn_t GenericCode) noexcept
        : Id(hash_string_view(Name())),
          Name(Name),
          Message(Message),
          Severity(Severity),
          GenericCode(GenericCode){}
  };
  class status_code{
    int64_t Value;
    const status_domain* Domain;
  public:
    template <status_code_enum Enum>
    constexpr status_code(Enum Value) noexcept
        : Value(static_cast<int64_t>(Value)),
          Domain(std::addressof(status_enum<Enum>::domain())){}
    constexpr status_code(int64_t Value, const status_domain& Domain) noexcept
        : Value(Value),
          Domain(&Domain){}

    [[nodiscard]] std::string_view domain() const noexcept{
      return Domain->Name();
    }
    [[nodiscard]] std::string_view message() const noexcept{
      return Domain->Message(Value);
    }
    [[nodiscard]] enum severity severity() const noexcept{
      return Domain->Severity(Value);
    }
    [[nodiscard]] generic_code generic() const noexcept{
      return Domain->GenericCode(Value);
    }

    friend bool operator==(const status_code& A, const status_code& B) noexcept{
      if(A.Domain->Id == B.Domain->Id)
        return A.Value == B.Value;
      return A.generic() == B.generic();
    }
    friend std::partial_ordering operator<=>(const status_code& A, const status_code& B) noexcept{
      if(A.Domain->Id == B.Domain->Id)
        return A.Value <=> B.Value;
      return std::partial_ordering::unordered;
    }
  };
  class status{
    status_code Code;
    uint32_t StackFrames = 0;
    bool Checked = false;
  public:
    template <status_code_enum Enum>
    status(Enum E) noexcept : Code(E){}
    ~status();
  };
  template <typename T>
  class outcome{
    union{
      std::aligned_storage_t<sizeof(T), alignof(T)> Storage{};
      T Value;
    };
    status Status;
  public:

  };


  enum class handler_action{
    resolve,
    callback,
    propagate,
    suppress,
    ignore,
    cancel,
    retry,
    kill,
    terminate
  };
  inline constexpr static handler_action action_resolve = handler_action::resolve;
  inline constexpr static handler_action action_ignore = handler_action::ignore;
  inline constexpr static handler_action action_callback = handler_action::callback;
  inline constexpr static handler_action action_suppress = handler_action::suppress;
  inline constexpr static handler_action action_propagate = handler_action::propagate;
  inline constexpr static handler_action action_cancel = handler_action::cancel;
  inline constexpr static handler_action action_retry = handler_action::retry;
  inline constexpr static handler_action action_kill = handler_action::kill;
  inline constexpr static handler_action action_terminate = handler_action::terminate;

  class status_handler{
    class action;
    class interface;
  public:
  };

  void register_handler(handler_action Action, status_code Code) noexcept;
  void register_handler(handler_action Action, generic_code Code) noexcept;
  void register_handler(handler_action Action, severity Severity) noexcept;


  /*class handler_action{
    class interface;
  public:
    enum class kind{
      resolve,
      callback,
      propagate,
      suppress,
      ignore,
      cancel,
      retry,
      kill,
      terminate
    };
  };*/
}

template <>
struct cu::status_enum<cu::generic_code>{
  static const cu::status_domain& domain() noexcept;
};




#endif//CUDA_FUNCTIONS_STATUS_H
