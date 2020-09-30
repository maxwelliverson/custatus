//
// Created by maxwell on 2020-09-27.
//

#ifndef CUDA_FUNCTIONS_STATUS_H
#define CUDA_FUNCTIONS_STATUS_H

#include <string_view>
#include <system_error>

namespace cu{
  /*class string_view{

    inline static constexpr __host__ __device__ size_t strlen(const char* Str) noexcept{
#if defined(__CUDACC__)
      unsigned Len = 0;
      while(Str[Len])
        ++Len;
      return Len;
#else
      return std::char_traits<char>::length(Str);
#endif
    }

    const char* Data;
    size_t Size;
  public:

    // types
    using traits_type		= std::char_traits<char>;
    using value_type		= char;
    using pointer		= value_type*;
    using const_pointer	= const value_type*;
    using reference		= value_type&;
    using const_reference	= const value_type&;
    using const_iterator	= const value_type*;
    using iterator		= const_iterator;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator	= const_reverse_iterator;
    using size_type		= size_t;
    using difference_type	= ptrdiff_t;
    static constexpr size_type npos = size_type(-1);

    // [string.view.cons], construction and assignment

    __host__ __device__ constexpr string_view() noexcept : Size{0}, Data{nullptr}{ }
    __host__ __device__ constexpr string_view(const string_view&) noexcept = default;
    [[gnu::__nonnull__]] __host__ __device__ constexpr string_view(const char* str) noexcept
        : Size{strlen(str)},
          Data{str}
    { }

    __host__ __device__ constexpr
    string_view(const char* str, size_type len) noexcept
        : Size{len}, Data{str}
    { }

    __host__ __device__ constexpr string_view& operator=(const string_view&) noexcept = default;

    // [string.view.iterators], iterator support

    [[nodiscard]] __host__ __device__ constexpr const_iterator begin() const noexcept
    { return this->Data; }
    [[nodiscard]] __host__ __device__ constexpr const_iterator end() const noexcept
    { return this->Data + this->Size; }
    [[nodiscard]] __host__ __device__ constexpr const_iterator cbegin() const noexcept
    { return this->Data; }
    [[nodiscard]] __host__ __device__ constexpr const_iterator cend() const noexcept
    { return this->Data + this->Size; }

    [[nodiscard]] __host__ __device__ constexpr const_reverse_iterator rbegin() const noexcept
    { return const_reverse_iterator(this->end()); }
    [[nodiscard]] __host__ __device__ constexpr const_reverse_iterator rend() const noexcept
    { return const_reverse_iterator(this->begin()); }
    [[nodiscard]] __host__ __device__ constexpr const_reverse_iterator crbegin() const noexcept
    { return const_reverse_iterator(this->end()); }
    [[nodiscard]] __host__ __device__ constexpr const_reverse_iterator crend() const noexcept
    { return const_reverse_iterator(this->begin()); }

    // [string.view.capacity], capacity

    [[nodiscard]] __host__ __device__ constexpr size_type size() const noexcept
    { return this->Size; }
    [[nodiscard]] __host__ __device__ constexpr size_type length() const noexcept
    { return Size; }

    [[nodiscard]] __host__ __device__ size_type max_size() const noexcept
    {
      return (npos - sizeof(size_type) - sizeof(void*))
             / sizeof(value_type) / 4;
    }

    [[nodiscard]] __host__ __device__ constexpr bool empty() const noexcept
    { return this->Size == 0; }

    // [string.view.access], element access

    [[nodiscard]] __host__ __device__ constexpr const_reference operator[](size_type pos) const noexcept
    {
      // TODO: Assert to restore in a way compatible with the constexpr.
      // __glibcxx_assert(pos < this->Size);
      return *(this->Data + pos);
    }

    [[nodiscard]] __host__ __device__ constexpr const_reference at(size_type pos) const
    {
      if (pos >= Size)
        __throw_out_of_range_fmt(__N("string_view::at: pos "
                                     "(which is %zu) >= this->size() "
                                     "(which is %zu)"), pos, this->size());
      return *(this->Data + pos);
    }
    [[nodiscard]] __host__ __device__ constexpr const_reference front() const noexcept
    {
      // TODO: Assert to restore in a way compatible with the constexpr.
      // __glibcxx_assert(this->Size > 0);
      return *this->Data;
    }

    [[nodiscard]] __host__ __device__ constexpr const_reference back() const noexcept
    {
      // TODO: Assert to restore in a way compatible with the constexpr.
      // __glibcxx_assert(this->Size > 0);
      return *(this->Data + this->Size - 1);
    }

    [[nodiscard]] __host__ __device__ constexpr const_pointer data() const noexcept
    { return this->Data; }

    // [string.view.modifiers], modifiers:

    __host__ __device__ constexpr void remove_prefix(size_type n) noexcept
    {
      assert(this->Size >= n);
      this->Data += n;
      this->Size -= n;
    }
    __host__ __device__ constexpr void remove_suffix(size_type n) noexcept
    { this->Size -= n; }
    __host__ __device__ constexpr void swap(string_view& sv) noexcept
    {
      auto tmp = *this;
      *this = sv;
      sv = tmp;
    }

    // [string.view.ops], string operations:

    __host__ __device__ constexpr size_type copy(char* str, size_type n, size_type pos = 0) const
    {
      //__glibcxx_requires_string_len(str, n);
      //pos = std::__sv_check(size(), pos, "string_view::copy");
      const size_type rlen = std::min(n, Size - pos);
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 2777. string_view::copy should use char_traits::copy
      traits_type::copy(str, data() + pos, rlen);
      return rlen;
    }

    [[nodiscard]] constexpr string_view substr(size_type pos = 0, size_type n = npos) const noexcept(false)
    {
      pos = std::__sv_check(size(), pos, "string_view::substr");
      const size_type rlen = std::min(n, Size - pos);
      return string_view{Data + pos, rlen};
    }

    [[nodiscard]] __host__ __device__ constexpr int compare(string_view str) const noexcept
    {
      const size_type rlen = std::min(this->Size, str.Size);
      int ret = traits_type::compare(this->Data, str.Data, rlen);
      if (ret == 0)
        ret = _s_compare(this->Size, str.Size);
      return ret;
    }
    [[nodiscard]] __host__ __device__ constexpr int compare(size_type pos1, size_type n1, string_view str) const
    { return this->substr(pos1, n1).compare(str); }
    [[nodiscard]] __host__ __device__ constexpr int compare(size_type pos1, size_type n1, string_view str, size_type pos2, size_type n2) const
    {
      return this->substr(pos1, n1).compare(str.substr(pos2, n2));
    }
    [[gnu::__nonnull__, nodiscard]] __host__ __device__ constexpr int compare(const char* str) const noexcept
    { return this->compare(string_view{str}); }
    [[gnu::__nonnull__, nodiscard]] __host__ __device__ constexpr int compare(size_type pos1, size_type n1, const char* str) const
    { return this->substr(pos1, n1).compare(string_view{str}); }
    [[nodiscard]] __host__ __device__ constexpr int compare(size_type pos1, size_type n1, const char* str, size_type n2) const noexcept(false)
    {
      return this->substr(pos1, n1)
                 .compare(string_view(str, n2));
    }

    [[nodiscard]] __host__ __device__ constexpr bool starts_with(string_view x) const noexcept
    { return this->substr(0, x.size()) == x; }
    [[nodiscard]] __host__ __device__ constexpr bool starts_with(char x) const noexcept
    { return !this->empty() && traits_type::eq(this->front(), x); }
    [[nodiscard]] __host__ __device__ constexpr bool starts_with(const char* x) const noexcept
    { return this->starts_with(string_view(x)); }

    [[nodiscard]] __host__ __device__ constexpr bool ends_with(string_view x) const noexcept
    {
      return this->size() >= x.size() && this->compare(this->size() - x.size(), npos, x) == 0;
    }
    [[nodiscard]] __host__ __device__ constexpr bool ends_with(char x) const noexcept
    { return !this->empty() && traits_type::eq(this->back(), x); }
    [[nodiscard]] __host__ __device__ constexpr bool ends_with(const char* x) const noexcept
    { return this->ends_with(string_view(x)); }

    [[nodiscard]] __host__ __device__ constexpr size_type find(string_view str, size_type pos = 0) const noexcept
    { return this->find(str.Data, pos, str.Size); }
    [[nodiscard]] __host__ __device__ constexpr size_type find(char c, size_type pos = 0) const noexcept;
    [[nodiscard]] __host__ __device__ constexpr size_type find(const char* str, size_type pos, size_type n) const noexcept;
    [[gnu::__nonnull__]] constexpr size_type find(const char* str, size_type pos = 0) const noexcept
    { return this->find(str, pos, traits_type::length(str)); }

    [[nodiscard]] __host__ __device__ constexpr size_type rfind(string_view str, size_type pos = npos) const noexcept
    { return this->rfind(str.Data, pos, str.Size); }
    [[nodiscard]] __host__ __device__ constexpr size_type rfind(char c, size_type pos = npos) const noexcept;
    [[nodiscard]] __host__ __device__ constexpr size_type rfind(const char* str, size_type pos, size_type n) const noexcept;
    [[gnu::__nonnull__]] __host__ __device__  constexpr size_type rfind(const char* str, size_type pos = npos) const noexcept
    { return this->rfind(str, pos, traits_type::length(str)); }

    [[nodiscard]] __host__ __device__ constexpr size_type find_first_of(string_view str, size_type pos = 0) const noexcept
    { return this->find_first_of(str.Data, pos, str.Size); }
    [[nodiscard]] __host__ __device__ constexpr size_type find_first_of(char c, size_type pos = 0) const noexcept
    { return this->find(c, pos); }
    [[nodiscard]] __host__ __device__ constexpr size_type find_first_of(const char* str, size_type pos, size_type n) const noexcept;
    [[gnu::__nonnull__]] __host__ __device__  constexpr size_type find_first_of(const char* str, size_type pos = 0) const noexcept
    { return this->find_first_of(str, pos, traits_type::length(str)); }

    [[nodiscard]] __host__ __device__ constexpr size_type find_last_of(string_view str, size_type pos = npos) const noexcept
    { return this->find_last_of(str.Data, pos, str.Size); }
    [[nodiscard]] __host__ __device__ constexpr size_type find_last_of(char c, size_type pos=npos) const noexcept
    { return this->rfind(c, pos); }
    [[nodiscard]] __host__ __device__ constexpr size_type find_last_of(const char* str, size_type pos, size_type n) const noexcept;
    [[gnu::__nonnull__]] __host__ __device__ constexpr size_type find_last_of(const char* str, size_type pos = npos) const noexcept
    { return this->find_last_of(str, pos, traits_type::length(str)); }

    [[nodiscard]] __host__ __device__ constexpr size_type find_first_not_of(string_view str, size_type pos = 0) const noexcept
    { return this->find_first_not_of(str.Data, pos, str.Size); }
    [[nodiscard]] __host__ __device__ constexpr size_type find_first_not_of(char c, size_type pos = 0) const noexcept;
    [[nodiscard]] __host__ __device__ constexpr size_type find_first_not_of(const char* str, size_type pos, size_type n) const noexcept;
    [[gnu::__nonnull__]] __host__ __device__ constexpr size_type find_first_not_of(const char* str, size_type pos = 0) const noexcept
    {
      return this->find_first_not_of(str, pos, traits_type::length(str));
    }

    [[nodiscard]] __host__ __device__ constexpr size_type find_last_not_of(string_view str, size_type pos = npos) const noexcept
    { return this->find_last_not_of(str.Data, pos, str.Size); }
    [[nodiscard]] __host__ __device__ constexpr size_type find_last_not_of(char c, size_type pos = npos) const noexcept;
    [[nodiscard]] __host__ __device__ constexpr size_type find_last_not_of(const char* str, size_type pos, size_type n) const noexcept;
    [[gnu::__nonnull__]] __host__ __device__ constexpr size_type find_last_not_of(const char* str, size_type pos = npos) const noexcept
    {
      return this->find_last_not_of(str, pos, traits_type::length(str));
    }

  private:
    __host__ __device__ static constexpr int _s_compare(size_type n1, size_type n2) noexcept
    {
      const difference_type diff = n1 - n2;
      if (diff > std::numeric_limits<int>::max())
        return std::numeric_limits<int>::max();
      if (diff < std::numeric_limits<int>::min())
        return std::numeric_limits<int>::max();
      return static_cast<int>(diff);
    }
  };

  inline static constexpr string_view get_device_function_address = R"src(
    extern "C" __global__ void fetch_function_ptr_of${_FnName}(${_RetType}(**fn_ptr)(${_Args...})){
      *fn_ptr = &${_FnName};
    }
  )src";*/
  template <typename Sig>
  class dual_function;
  template <typename Ret, typename ...Args>
  class dual_function<Ret(Args...)>{
    Ret(*DevicePtr)(Args...);
    Ret(*HostPtr)(Args...);
  public:

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
    value_too_large                    = 132, // EOVERFLOW
    wrong_protocol_type                = 136 // EPROTOTYPE
  };

  class status_code;

  template <typename T>
  struct is_status_code_enum : std::false_type {};
  template <typename T>
  concept status_code_enum = is_status_code_enum<T>::value;

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
        : Id(hash_string_view(Name())), Name(Name), Message(Message), Severity(Severity), GenericCode(GenericCode){}
  };
  class status_code{
    int64_t Value;
    const status_domain* Domain;
  public:
    constexpr status_code(int64_t Value, const status_domain& Domain) noexcept : Value(Value), Domain(&Domain){}

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
    status(Enum E){}
    ~status();
  };
  template <typename T>
  class outcome{

  };


  class status_handler{
    class action;
  };

  class handler_action{
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
  };

}

#endif//CUDA_FUNCTIONS_STATUS_H
