//
// Created by maxwell on 2020-09-27.
//

#ifndef CUDA_FUNCTIONS_STATUS_H
#define CUDA_FUNCTIONS_STATUS_H

#include <nvrtc.h>
#include <string_view>

namespace cu{

  class string_view{

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

    /*[[nodiscard]] __host__ __device__ constexpr const_reverse_iterator rbegin() const noexcept
    { return const_reverse_iterator(this->end()); }
    [[nodiscard]] __host__ __device__ constexpr const_reverse_iterator rend() const noexcept
    { return const_reverse_iterator(this->begin()); }
    [[nodiscard]] __host__ __device__ constexpr const_reverse_iterator crbegin() const noexcept
    { return const_reverse_iterator(this->end()); }
    [[nodiscard]] __host__ __device__ constexpr const_reverse_iterator crend() const noexcept
    { return const_reverse_iterator(this->begin()); }*/

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
      /*if (pos >= Size)
        __throw_out_of_range_fmt(__N("string_view::at: pos "
                                     "(which is %zu) >= this->size() "
                                     "(which is %zu)"), pos, this->size());*/
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

    /*__host__ __device__ constexpr size_type copy(char* str, size_type n, size_type pos = 0) const
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

    *//*[[nodiscard]] __host__ __device__ constexpr int compare(string_view str) const noexcept
    {
      const size_type rlen = std::min(this->Size, str.Size);
      int ret = traits_type::compare(this->Data, str.Data, rlen);
      if (ret == 0)
        ret = _s_compare(this->Size, str.Size);
      return ret;
    }*//*
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
    { return this->substr(0, x.size()) == x; }*/
    [[nodiscard]] __host__ __device__ constexpr bool starts_with(char x) const noexcept
    { return !this->empty() && traits_type::eq(this->front(), x); }
    /*[[nodiscard]] __host__ __device__ constexpr bool starts_with(const char* x) const noexcept
    { return this->starts_with(string_view(x)); }*/

    /*[[nodiscard]] __host__ __device__ constexpr bool ends_with(string_view x) const noexcept
    {
      return this->size() >= x.size() && this->compare(this->size() - x.size(), npos, x) == 0;
    }*/
    [[nodiscard]] __host__ __device__ constexpr bool ends_with(char x) const noexcept
    { return !this->empty() && traits_type::eq(this->back(), x); }
    /*[[nodiscard]] __host__ __device__ constexpr bool ends_with(const char* x) const noexcept
    { return this->ends_with(string_view(x)); }*/

   /* [[nodiscard]] __host__ __device__ constexpr size_type find(string_view str, size_type pos = 0) const noexcept
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
*/
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
  )src";


  template <typename Sig>
  class dual_function;
  template <typename Ret, typename ...Args>
  class dual_function<Ret(Args...)>{
    Ret(*DevicePtr)(Args...);
    Ret(*HostPtr)(Args...);
  public:

  };




  class status{

    bool Checked;
  };


  class status_handler{

  };

  class handler_action{};
  class resolve_action : public handler_action{};
  class propagate_action : public handler_action{};
  class suppress_action : public handler_action{};
  class ignore_action : public handler_action{};
  class cancel_action : public handler_action{};
  class retry_action : public handler_action{};
  class kill_action : public handler_action{};
  class terminate_action : public handler_action{};
}

#endif//CUDA_FUNCTIONS_STATUS_H
