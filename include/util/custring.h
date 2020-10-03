//
// Created by Maxwell on 2020-09-29.
//

#ifndef CUDA_FUNCTIONS_CUSTRING_H
#define CUDA_FUNCTIONS_CUSTRING_H

#include "contracts.h"

#include <type_traits>
#include <string>
#include <concepts>
#include <memory>
#include <system_error>
#include <iostream>
#include <memory_resource>
#include <llvm/Support/raw_ostream.h>


namespace cu{

  template <typename C>
  concept char_like = std::regular<C> && std::is_integral_v<C> && requires{typename std::char_traits<C>;};
  template <typename CS>
  concept cstring_like = std::is_pointer_v<CS> && char_like<std::remove_const_t<std::remove_pointer_t<CS>>>;

  namespace meta{
    template <size_t N>
    struct overload : overload<N-1>{};
    template <>
    struct overload<0>{};
  }
  namespace functor{
    class size_functor{
      template <typename Str>
      inline constexpr static auto private_call(const Str& S, meta::overload<3>) noexcept
          -> decltype(S.size()){
        return S.size();
      }
      template <typename Str>
      inline constexpr static auto private_call(const Str& S, meta::overload<2>) noexcept
          -> decltype(size(S)){
        return size(S);
      }
      template <typename Char, size_t N, typename = std::void_t<std::char_traits<Char>>>
      inline constexpr static size_t private_call(const Char(&S)[N], meta::overload<1>) noexcept{
        return N-1;
      }
      template <typename Char, typename = std::void_t<std::char_traits<Char>>>
      inline static size_t private_call(const Char* String, meta::overload<0>) noexcept{
        return std::char_traits<Char>::length(String);
      }
    public:
      template <typename Str>
      inline constexpr size_t operator()(const Str& S) const noexcept
          requires(requires{
            {size_functor::private_call(std::declval<const Str&>(), meta::overload<3>{})} -> std::convertible_to<size_t>; }
         ){
        return size_functor::private_call(S, meta::overload<3>{});
      }
    };
    class data_functor{
      template <typename Str>
      inline constexpr static auto private_call(const Str& S, meta::overload<2>) noexcept -> decltype(S.data()){
        return S.data();
      }
      template <typename Str>
      inline constexpr static auto private_call(const Str& S, meta::overload<1>) noexcept
          -> decltype(data(S)){
        return data(S);
      }
      template <typename Char, typename = std::void_t<std::char_traits<Char>>>
      inline static const Char* private_call(const Char* String, meta::overload<0>) noexcept{
        return String;
      }
    public:
      template <typename Str>
      inline constexpr auto operator()(const Str& S) const noexcept requires(requires{
        { data_functor::private_call(S, meta::overload<2>{}) } -> cstring_like;
      }){
        return data_functor::private_call(S, meta::overload<2>{});
      }
    };
    class compare_functor{
      template <typename Str>
      inline constexpr static std::partial_ordering private_call(const Str& S, decltype(data_functor{}(S)) CString, size_t Pos, size_t N, size_t CStrN, meta::overload<2>) noexcept requires(requires{
        { std::char_traits<std::remove_cvref_t<decltype(*data_functor{}(S))>>::compare(data_functor{}(S) + Pos, CString, std::min(N, CStrN)) } -> std::same_as<int>;
      }){
        if(int Value = std::char_traits<std::remove_cvref_t<decltype(*data_functor{}(S))>>::compare(data_functor{}(S) + Pos, CString, std::min(N, CStrN)))
          return Value <=> 0;
        return N <=> CStrN;
      }
      template <typename Str, cstring_like CStr>
      inline constexpr static auto private_call(const Str& S, CStr CString, size_t Pos, size_t N, size_t CStrN, meta::overload<1>) noexcept requires(requires{
        { S.compare(Pos, N, CString, CStrN) } -> std::same_as<int>;
      }){
        return S.compare(Pos, N, CString, CStrN) <=> 0;
      }
      /*template <typename Str, cstring_like CStr>
      inline constexpr static auto private_call(const Str& S, CStr CString, size_t Pos, size_t N,  size_t CStrN, meta::overload<0>) noexcept
          -> decltype(std::lexicographical_compare_three_way(data_functor{}(S) + Pos, data_functor{}(S) + Pos + N, CString, CString + CStrN)){
        auto SBegin = data_functor{}(S) + Pos;
        return std::lexicographical_compare_three_way(SBegin, SBegin + N, CString, CString + CStrN);
      }*/
      inline constexpr static std::partial_ordering private_call(...) noexcept{
        return std::partial_ordering::unordered;
      }
    public:
      template <typename Str, typename Char>
      inline constexpr std::partial_ordering operator()(const Str& String, const Char* CString, size_t Pos, size_t N, size_t CStrN) const noexcept /*requires(requires{
        { compare_functor::private_call(String, CString, Pos, N, CStrN, meta::overload<2>{}) } -> std::convertible_to<std::partial_ordering>;
      })*/{
        return compare_functor::private_call(String, CString, Pos, N, CStrN, meta::overload<2>{});
      }
    };
    class print_functor{
      template <typename Str>
      inline static void private_call(const Str& String, llvm::raw_ostream& OS, meta::overload<2>) noexcept requires(
          requires{
            OS << String;
          })
      {
        OS << String;
      }
      template <typename Str>
      inline static void private_call(const Str& String, std::basic_ostream<std::remove_cvref_t<decltype(*data_functor{}(String))>>& OS, meta::overload<2>) noexcept requires(
          requires{
            OS << String;
          })
      {
        OS << String;
      }
      template <typename Str>
      inline static void private_call(const Str& String, llvm::raw_ostream& OS, meta::overload<1>) noexcept{
        OS << std::basic_string_view{data_functor{}(String), size_functor{}(String)};
      }
      template <typename Str>
      inline static void private_call(const Str& String, std::basic_ostream<std::remove_cvref_t<decltype(*data_functor{}(String))>>& OS, meta::overload<1>) noexcept{
        OS << std::basic_string_view{data_functor{}(String), size_functor{}(String)};
      }
      template <typename Str, typename OStream>
      inline static void private_call(const Str&, OStream& OS, meta::overload<0>) noexcept{
        OS << "=== [ ERROR INVALID STRING ] ===";
      }
    public:
      template <typename Str, typename OStream>
      inline void operator()(const Str& String, OStream& OS) const noexcept requires(requires{
          print_functor::private_call(String, OS, meta::overload<2>{});
      }){
        print_functor::private_call(String, OS, meta::overload<2>{});
      }
    };
  }

  template <typename S>
  concept string_like = requires(const S& String){
    { functor::size_functor{}(String) } -> std::convertible_to<size_t>;
    { functor::data_functor{}(String) } -> cstring_like;
  };

  namespace detail{
    enum class char_type{
      Default,
      Signed,
      Unsigned,
      Wide,
      Utf8,
      Utf16,
      Utf32,
      Invalid
    };
    template <typename T>
    inline constexpr static char_type char_token = char_type::Invalid;
    template <>
    inline constexpr static char_type char_token<char> = char_type::Default;
    template <>
    inline constexpr static char_type char_token<wchar_t> = char_type::Wide;
    template <>
    inline constexpr static char_type char_token<char8_t> = char_type::Utf8;
    template <>
    inline constexpr static char_type char_token<char16_t> = char_type::Utf16;
    template <>
    inline constexpr static char_type char_token<char32_t> = char_type::Utf32;
    template <>
    inline constexpr static char_type char_token<signed char> = char_type::Signed;
    template <>
    inline constexpr static char_type char_token<unsigned char> = char_type::Unsigned;

    template <typename Functor, string_like Str, typename ...Args>
    auto apply(const Str& String, const void* CString, char_type Type, Args&& ...args) noexcept{
      switch(Type){
        case char_type::Default:
          return Functor{}(String, static_cast<const char*>(CString), std::forward<Args>(args)...);
        case char_type::Signed:
          return Functor{}(String, static_cast<const signed char*>(CString), std::forward<Args>(args)...);
        case char_type::Unsigned:
          return Functor{}(String, static_cast<const unsigned char*>(CString), std::forward<Args>(args)...);
        case char_type::Wide:
          return Functor{}(String, static_cast<const wchar_t*>(CString), std::forward<Args>(args)...);
        case char_type::Utf8:
          return Functor{}(String, static_cast<const char8_t*>(CString), std::forward<Args>(args)...);
        case char_type::Utf16:
          return Functor{}(String, static_cast<const char16_t*>(CString), std::forward<Args>(args)...);
        case char_type::Utf32:
          return Functor{}(String, static_cast<const char32_t*>(CString), std::forward<Args>(args)...);
        case char_type::Invalid:
          return Functor{}(String, CString, std::forward<Args>(args)...);
      }
      __assume(false);
    }

    template <typename T, typename AllocTmp>
    class allocator_base{
      using Alloc = typename std::allocator_traits<AllocTmp>::template rebind_alloc<T>;
      using traits = std::allocator_traits<Alloc>;
      mutable Alloc Allocator;

    protected:
      using allocator_type = Alloc;

      explicit allocator_base(const Alloc& Allocator) noexcept : Allocator(Allocator){}

      template <typename Arg>
      inline auto new_ptr(Arg&& arg) const{
        auto Ptr = traits::allocate(Allocator);
        traits::construct(Allocator, Ptr, traits::select_on_container_copy_construction(Allocator), std::forward<Arg>(arg));
        return Ptr;
      }
      template <typename Ptr>
      inline void delete_ptr(Ptr* Pointer) const noexcept{
        Alloc AllocCopy = std::move(Allocator);
        traits::destroy(AllocCopy, Pointer);
        traits::deallocate(AllocCopy, Pointer, 1);
      }
    };
    template <typename T, typename AllocTmp> requires(
        std::is_empty_v<typename std::allocator_traits<AllocTmp>::template rebind_alloc<T>> &&
        std::allocator_traits<AllocTmp>::template rebind_traits<T>::is_always_equal::value)
    class allocator_base<T, AllocTmp>{
      using Alloc = typename std::allocator_traits<AllocTmp>::template rebind_alloc<T>;
      using traits = std::allocator_traits<Alloc>;
    protected:
      using allocator_type = Alloc;

      explicit allocator_base(const Alloc&) noexcept{}

      template <typename Arg>
      inline auto new_ptr(Arg&& arg) const{
        Alloc Allocator{};
        auto Ptr = traits::allocate(Allocator, 1);
        traits::construct(Allocator, Ptr, traits::select_on_container_copy_construction(Allocator), std::forward<Arg>(arg));
        return Ptr;
      }
      template <typename Ptr>
      inline void delete_ptr(Ptr* Pointer) const noexcept{
        Alloc AllocCopy{};
        traits::destroy(AllocCopy, Pointer);
        traits::deallocate(AllocCopy, Pointer, 1);
      }
    };
  }
  /*class string{
    inline constexpr static functor::size_functor size_fn{};
    inline constexpr static functor::data_functor data_fn{};
    inline constexpr static functor::print_functor print_fn{};

    struct info{
      detail::char_type Type;
      const void* Data;
      size_t Size;
    };

    class string_base{
    protected:
      ~string_base() = default;
    public:
      virtual void destroy() noexcept;

      [[nodiscard]] virtual size_t size() const noexcept = 0;
      [[nodiscard]] virtual const void* data() const noexcept = 0;
      [[nodiscard]] virtual detail::char_type get_type() const noexcept = 0;
      [[nodiscard]] virtual string_base* clone() const = 0;

      [[nodiscard]] virtual info get_info() const noexcept = 0;

      virtual void print(llvm::raw_ostream& OS) const noexcept = 0;
      virtual void print(std::ostream& OS) const noexcept = 0;

      [[nodiscard]] virtual std::partial_ordering compare(const info& Other) const noexcept = 0;
    };
    template <string_like Str, typename Allocator>
    class typed_string :
        public string_base,
        detail::allocator_base<typed_string<Str, Allocator>, Allocator>{
      using allocator_base = detail::allocator_base<typed_string<Str, Allocator>, Allocator>;
      using allocator_type = typename allocator_base::allocator_type;
      Str String;

      using char_type = std::remove_cvref_t<decltype(*data_fn(String))>;

      template <size_t N, size_t ...I>
      explicit typed_string(const allocator_type& Alloc, const char_type(&String)[N], std::index_sequence<I...>) noexcept
          : allocator_base(Alloc), String{String[I]...}{}

    public:
      template <typename ...Args>
      explicit typed_string(const allocator_type& Alloc, Args&& ...args) noexcept(std::is_nothrow_constructible_v<Str, Args...>)
          : allocator_base(Alloc), String(std::forward<Args>(args)...){}

      template <size_t N>
      explicit typed_string(const allocator_type& Alloc, const char_type(&String)[N]) noexcept
          : typed_string(Alloc, String, std::make_index_sequence<N>{}){}

      void destroy() noexcept override{
        allocator_base::delete_ptr(this);
      }

      [[nodiscard]] size_t size() const noexcept override{
        return size_fn(String);
      }
      [[nodiscard]] const void* data() const noexcept override{
        return data_fn(String);
      }
      [[nodiscard]] detail::char_type get_type() const noexcept override{
        return detail::char_token<std::remove_cvref_t<decltype(*data_fn(String))>>;
      }
      [[nodiscard]] typed_string * clone() const override{
        return allocator_base::new_ptr(String);
      }

      [[nodiscard]] info get_info() const noexcept override{
        return info{
            .Type=detail::char_token<std::remove_cvref_t<decltype(*data_fn(String))>>,
            .Data=data_fn(String),
            .Size=size_fn(String)
        };
      }

      void print(llvm::raw_ostream& OS) const noexcept override{
        print_fn(String, OS);
      }
      void print(std::ostream& OS) const noexcept override{
        print_fn(String, OS);
      }

      [[nodiscard]] std::partial_ordering compare(const info& Other) const noexcept override{
        return detail::apply<functor::compare_functor>(String, Other.Data, Other.Type, 0, size_fn(String), Other.Size);
      }
    };

    string_base* StringPtr = nullptr;

    template <string_like Str, typename Allocator>
    inline static auto make(Str&& String, const Allocator& AllocTmp){
      using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<typed_string<std::remove_cvref_t<Str>, Allocator>>;
      using traits = std::allocator_traits<allocator_type>;
      allocator_type Alloc{std::move(AllocTmp)};
      auto* Ptr = traits::allocate(Alloc, 1);
      traits::construct(Alloc, Ptr, Alloc, std::forward<Str>(String));
      return Ptr;
    }
    template <string_like Str, typename StrAlloc, typename Allocator>
    inline static auto make(Str&& String, const StrAlloc& StrAllocTmp, const Allocator& AllocTmp){
      using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<typed_string<std::remove_cvref_t<Str>, Allocator>>;
      using traits = std::allocator_traits<allocator_type>;
      allocator_type Alloc{std::move(AllocTmp)};
      auto* Ptr = traits::allocate(Alloc, 1);
      traits::construct(Alloc, Ptr, Alloc, std::forward<Str>(String), StrAllocTmp);
      return Ptr;
    }
    inline static void destroy(string_base* String) noexcept{
      if(String)
        String->destroy();
    }

  public:
    string() = default;
    template <string_like Str, typename Allocator = std::allocator<std::byte>>
    string(Str&& String, const Allocator& Alloc = {})
        : StringPtr(make(std::forward<Str>(String), Alloc)){}
    template <string_like Str, typename StringAlloc, typename Allocator = std::allocator<std::byte>>
    string(std::allocator_arg_t, const StringAlloc& StrAlloc, Str&& String, const Allocator& Alloc = {})
        : StringPtr(make(std::forward<Str>(String), StrAlloc, Alloc)){}
    string(const string& Other) : StringPtr(Other.StringPtr->clone()){}
    string(string&& Other) noexcept : StringPtr(Other.StringPtr){
      Other.StringPtr = nullptr;
    }
    ~string(){
      destroy(StringPtr);
    }

    string& operator=(const string& String){
      if(this == &String)
        return *this;
      destroy(StringPtr);
      StringPtr = String.StringPtr->clone();
      return *this;
    }
    string& operator=(string&& String) noexcept{
      destroy(StringPtr);
      StringPtr = String.StringPtr;
      String.StringPtr = nullptr;
      return *this;
    }
    template <string_like Str>
    string& operator=(Str&& String){
      destroy(StringPtr);
      StringPtr = string::make(std::forward<Str>(String), std::allocator<void>{});
      return *this;
    }

    [[nodiscard]] bool empty() const noexcept{
      if(!StringPtr)
        return true;
      return !StringPtr->size();
    }
    [[nodiscard]] size_t size() const noexcept{
      if(!StringPtr)
        return 0;
      return StringPtr->size();
    }
    [[nodiscard]] const void* data() const noexcept{
      if(!StringPtr)
        return "\0";
      return StringPtr->data();
    }
    [[nodiscard]] std::partial_ordering compare(const string& Other) const noexcept{
      switch((!!StringPtr << 1) | (!!Other.StringPtr)){
        case 1:
          return std::partial_ordering::less;
        case 2:
          return std::partial_ordering::greater;
        case 3:
          return StringPtr->compare(Other.StringPtr->get_info());
        default:
          return std::partial_ordering::equivalent;
      }
    }

    friend bool operator==(const string& A, const string& B) noexcept{
      return (A <=> B) == std::partial_ordering::equivalent;
    }
    friend std::partial_ordering operator<=>(const string& A, const string& B) noexcept{
      return A.compare(B);
    }
    template <typename Char>
    friend std::basic_ostream<Char>& operator<<(std::basic_ostream<Char>& OS, const string& String){
      String.StringPtr->print(OS);
      return OS;
    }
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const string& String){
      String.StringPtr->print(OS);
      return OS;
    }
  };*/

  namespace detail{
    template <typename Alloc>
    struct allocator_like;
    template <typename Alloc> requires(std::is_class_v<Alloc>)
    struct allocator_like<Alloc>{
      using type = Alloc;
    };
    template <std::derived_from<std::pmr::memory_resource> Alloc>
    struct allocator_like<Alloc*>{
      using type = std::pmr::polymorphic_allocator<std::byte>;
    };

    template <typename T>
    concept allocator = requires{
        typename allocator_like<T>::type;
    } && requires(typename allocator_like<T>::type& A, size_t N){
      { std::allocator_traits<typename allocator_like<T>::type>::allocate(A, N) };
    };
  }


  template <typename Alloc = std::allocator<char>>
  class fixed_string : std::allocator_traits<Alloc>::template rebind_alloc<char>{
    using allocator = typename std::allocator_traits<Alloc>::template rebind_alloc<char>;
    using alloc_traits = std::allocator_traits<allocator>;

    inline allocator & alloc() noexcept{
      return static_cast<allocator &>(*this);
    }
    inline const allocator & alloc() const noexcept{
      return static_cast<const allocator &>(*this);
    }

    inline static char null_string[1] = "";

    uint32_t AliasingBits = 0;
    uint32_t Length = 0;
    typename alloc_traits::pointer Data = null_string;
  public:

    using allocator_type = allocator;
    using value_type = char;
    using pointer = typename alloc_traits::pointer;
    using const_pointer = typename alloc_traits::const_pointer;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using size_type = typename alloc_traits::size_type;



    fixed_string() = default;
    fixed_string(const fixed_string& Other) :
      allocator(Other.alloc()),
      AliasingBits(Other.AliasingBits),
      Length(Other.Length),
      Data(alloc_traits::allocate(alloc(), Length + 1)){
      std::memcpy(Data, Other.Data, Length + 1);
    }
    fixed_string(fixed_string&& Other) noexcept{
      swap(Other);
    }
    explicit fixed_string(size_t ReserveLength, char CharValue = ' ') noexcept
        : Length(ReserveLength),
          Data(alloc_traits::allocate(alloc(), Length + 1)){
      std::memset(Data, CharValue, Length);
      Data[Length] = '\0';
    }
    template <string_like Str>
    fixed_string(const Str& String)
        : Length(functor::size_functor{}(String)),
          Data(alloc_traits::allocate(alloc(), Length + 1)){
      std::memcpy(Data, functor::data_functor{}(String), Length);
      Data[Length] = '\0';
    }

    fixed_string(const allocator& Allocator) : allocator(Allocator){};
    fixed_string(const fixed_string& Other, const allocator& Allocator) :
        allocator(Allocator),
        AliasingBits(Other.AliasingBits),
        Length(Other.Length),
        Data(alloc_traits::allocate(alloc(), Length + 1)){
      std::memcpy(Data, Other.Data, Length + 1);
    }
    fixed_string(size_t ReserveLength, char CharValue, const allocator& Allocator) noexcept
        : allocator(Allocator),
          Length(ReserveLength),
          Data(alloc_traits::allocate(alloc(), Length + 1)){
      std::memset(Data, CharValue, Length);
      Data[Length] = '\0';
    }
    template <string_like Str>
    fixed_string(const Str& String, const allocator& Allocator)
        : allocator(Allocator),
          Length(functor::size_functor{}(String)),
          Data(alloc_traits::allocate(alloc(), Length + 1)){
      std::memcpy(Data, functor::data_functor{}(String), Length);
      Data[Length] = '\0';
    }


    fixed_string(const std::basic_string<char, std::char_traits<char>, allocator>& StdString) noexcept
        : allocator(StdString.get_allocator()),
          Length(StdString.size()),
          Data(alloc_traits::allocate(alloc(), Length + 1)){
      std::memcpy(Data, StdString.data(), Length);
      Data[Length] = '\0';
    }
    fixed_string(std::basic_string<char, std::char_traits<char>, allocator>&& StdString) noexcept
        : Alloc(std::move(StdString.get_allocator())),
          Length(StdString.size()),
          Data(alloc_traits::allocate(alloc(), Length + 1)){
      std::memcpy(Data, StdString.data(), Length);
      Data[Length] = '\0';
    }


    fixed_string& operator=(const fixed_string& Other){
      if(this != &Other){
        this->~fixed_string();
        ::new(this) fixed_string(Other);
      }
      return *this;
    }
    fixed_string& operator=(fixed_string&& Other) noexcept{
      ~fixed_string();
      ::new(this) fixed_string(std::move(Other));
      return *this;
    }

    ~fixed_string(){
      if(Length)
        alloc_traits::deallocate(alloc(), Data, Length + 1);
    }

    allocator& get_allocator() noexcept{
      return static_cast<allocator &>(*this);
    }
    const allocator& get_allocator() const noexcept{
      return static_cast<const allocator &>(*this);
    }




    [[nodiscard]] pointer data() const noexcept{
      return Data;
    }

    [[nodiscard]] size_type size() const noexcept{
      return Length;
    }
    [[nodiscard]] size_type length() const noexcept{
      return Length;
    }
    [[nodiscard]] bool empty() const noexcept{
      return !Data[0];
    }

    [[nodiscard]] char& front() noexcept{
      return *Data;
    }
    [[nodiscard]] const char& front() const noexcept{
      return *Data;
    }
    [[nodiscard]] char& back() noexcept{
      return Data[Length - 1];
    }
    [[nodiscard]] const char& back() const noexcept{
      return Data[Length - 1];
    }

    [[nodiscard]] char& operator[](size_type N) noexcept{
      CU_axiom(N < Length)
      return Data[N];
    }
    [[nodiscard]] const char& operator[](size_type N) const noexcept{
      CU_axiom(N < Length)
      return Data[N];
    }

    [[nodiscard]] iterator begin() noexcept{
      return Data;
    }
    [[nodiscard]] const_iterator begin() const noexcept{
      return Data;
    }
    [[nodiscard]] const_iterator cbegin() const noexcept{
      return Data;
    }

    [[nodiscard]] iterator end() noexcept{
      return Data + Length;
    }
    [[nodiscard]] const_iterator end() const noexcept{
      return Data + Length;
    }
    [[nodiscard]] const_iterator cend() const noexcept{
      return Data + Length;
    }

    void swap(fixed_string& Other) noexcept{
      if constexpr(!alloc_traits::propagates_on_container_swap::value)
      CU_axiom(alloc() == Other.alloc())
      std::swap(alloc(), Other.alloc());
      std::swap(reinterpret_cast<std::byte(&)[16]>(AliasingBits),
                reinterpret_cast<std::byte(&)[16]>(Other.AliasingBits));
    }

    template <string_like Str>
    [[nodiscard]] int compare(const Str& String) const noexcept{
      return this->compare(functor::data_functor{}(String), functor::size_functor{}(String));
    }
    [[nodiscard]] int compare(const char* CString, size_type Size) const noexcept{
      if(int result = std::char_traits<char>::compare(
          std::pointer_traits<pointer>::to_address(Data),
          CString,
          std::min(Length, Size)))
        return result;
      return int(Length <=> Size);
    }

    friend std::ostream& operator<<(std::ostream& OS, const fixed_string& Str){
      return OS.write(Str.Data, Str.Length);
    }
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const fixed_string& Str){
      return OS.write(Str.Data, Str.Length);
    }
    friend bool operator==(const fixed_string& A, const fixed_string& B) noexcept{
      return (A <=> B) == std::strong_ordering::equal;
    }
    friend std::strong_ordering operator<=>(const fixed_string& A, const fixed_string& B) noexcept{
      return std::lexicographical_compare_three_way(A.begin(), A.end(), B.begin(), B.end());
    }
  };


  template <typename Char, typename Traits, typename Allocator>
  fixed_string(const std::basic_string<Char, Traits, Allocator>&) -> fixed_string<Allocator>;
  template <typename Char, typename Traits, typename Allocator>
  fixed_string(std::basic_string<Char, Traits, Allocator>&&) -> fixed_string<Allocator>;
  template <string_like Str, detail::allocator Alloc>
  fixed_string(const Str&, const Alloc&) -> fixed_string<typename detail::allocator_like<Alloc>::type>;
  template <detail::allocator Alloc>
  fixed_string(const Alloc&) -> fixed_string<typename detail::allocator_like<Alloc>::type>;
  template <string_like Str>
  fixed_string(const Str&) -> fixed_string<>;
  template <detail::allocator Alloc>
  fixed_string(size_t, char, const Alloc&) -> fixed_string<typename detail::allocator_like<Alloc>::type>;
}

#endif//CUDA_FUNCTIONS_CUSTRING_H
