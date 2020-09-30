//
// Created by Maxwell on 2020-09-28.
//

#ifndef CUDA_FUNCTIONS_RTC_H
#define CUDA_FUNCTIONS_RTC_H

#include <string_view>
#include <span>
#include <system_error>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringMap.h>


/*#define _ptx_source_string_BEGIN_ R"ptx_src(
#define _ptx_source_string_END_ )ptx_src"
#define PTX_SOURCE_STRING_BEGIN _ptx_source_string_BEGIN_
#define PTX_SOURCE_STRING_END _ptx_source_string_END_
#define PTX_SOURCE_STRING(...) PTX_SOURCE_STRING_BEGIN #__VA_ARGS__ PTX_SOURCE_STRING_END*/
#define STRINGIFY_I(...) #__VA_ARGS__
#define STRINGIFY_ STRINGIFY_I
#define STRINGIFY(...) STRINGIFY_(__VA_ARGS__)
#define CONCAT_I(x, y) x##y
#define CONCAT_ CONCAT_I
#define CONCAT(x, y) CONCAT_(x, y)
#define EMBED_SOURCE_STRING(...) CONCAT(R, STRINGIFY(ptx_SRC(__VA_ARGS__)ptx_SRC))
#define INVAR(Name) ${Name}

inline constexpr static std::string_view get_function_ptr =
    EMBED_SOURCE_STRING(extern "C" __global__ void fetch_device_ptr_${_FnName}(${_RetType}(**fn_ptr)(${_Args...})){
      *fn_ptr = &${_FnName};
    });

inline constexpr static std::string_view some_source =
    EMBED_SOURCE_STRING(__global__ void add_fn(int* A, int* B, size_t N){
      unsigned I = threadIdx.x + blockDim.x * blockIdx.x;
      unsigned Stride = blockDim.x * gridDim.x;
      for(; I < N; I += Stride)
        A[I] += B[I];
    });

namespace cu::rtc{
  class source_code{
    class impl;
    class interface;
    class source_template;
    class complete_source;
    class empty_source;
  public:
    source_code();
    template <size_t N>
    source_code(const char(&CString)[N]) : source_code(llvm::StringRef(CString)){}
    source_code(llvm::StringRef String);
    source_code(std::string_view String);
    source_code(const source_code& Other);
    source_code(source_code&& Other) noexcept;
    ~source_code();

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] bool is_complete() const noexcept;
    [[nodiscard]] std::span<const llvm::StringRef> template_arguments() const;
    std::error_code substitute(llvm::StringRef Key, llvm::StringRef Substitution);

  private:
    std::unique_ptr<impl> Impl;

    void print(std::ostream& OS) const;
    void print(llvm::raw_ostream& OS) const;

    friend std::ostream& operator<<(std::ostream& OS, const source_code& SC){
      SC.print(OS);
      return OS;
    }
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const source_code& SC){
      SC.print(OS);
      return OS;
    }
  };
  class function;
  class global;
  class module;
  class ptx_source;

  class compiler{
    class impl;
    class interface;
  public:
    compiler();




  private:
    std::unique_ptr<impl> Impl;
  };
  class ptx_source{};

  class function{  };
  class global{};
  class module{};

}

#endif//CUDA_FUNCTIONS_RTC_H
