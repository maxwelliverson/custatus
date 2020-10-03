//
// Created by Maxwell on 2020-09-28.
//

#ifndef CUDA_FUNCTIONS_RTC_H
#define CUDA_FUNCTIONS_RTC_H

#include "util/custring.h"


#include <string_view>
#include <span>
#include <system_error>
#include <memory>
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


namespace cu{

}

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
  namespace code{
    class function{};
    class kernel{};
    class declaration{};
    class block{};
    class global{};
    class constant{};
  }
  class function;
  class global;
  class module;

  enum class arch{
    compute_35,
    compute_37,
    compute_50,
    compute_52,
    compute_53,
    compute_60,
    compute_61,
    compute_62,
    compute_70,
    compute_72,
    compute_75,
    compute_80,
    compute_86,
    sm_35,
    sm_37,
    sm_50,
    sm_52,
    sm_53,
    sm_60,
    sm_61,
    sm_62,
    sm_70,
    sm_72,
    sm_75,
    sm_80,
    sm_86,
  };
  enum class standard{
    cxx11,
    cxx14,
    cxx17
  };
  enum class default_execution_space{
    host,
    device
  };

  namespace options{
    enum compiler{
      builtin_move_forward,
      builtin_initializer_list,
      warnings,
      pointer_aliasing,
      device_as_default_execution_space,
      denormal_values,
      fma_instructions,
      precise_sqrt,
      precise_division,
      fast_math,
      extra_device_vectorization,
      linking,
      link_time_optimizations,
      debug_info_generation,
      line_info_generation,
    };
    enum linker{
      wall_time,
      log_info,
      log_errors,
      generate_debug_info,
      verbose,
      global_symbols
    };
  }

  struct header{
    llvm::SmallString<8> Name;
    fixed_string<> Body;
  };

  class compiler{
    class impl;
    class interface;
    //class options;
  public:

    using option = options::compiler;


    compiler();
    ~compiler();


    void enable(option Opt);
    void disable(option Opt);
    bool get(option Opt) const;
    void set(option Opt, bool IsEnabled);
    void set(default_execution_space ExecutionSpace);

    void target(standard Std);
    void target(arch Arch);
    void set_max_register_count(uint32_t MaxRegCount);

    void define(std::string_view PPMacro);
    void undef(std::string_view PPMacro);
    void search_path(std::string_view PPMacro);
    void include(std::string_view Header);

    module compile() const;

    fixed_string<> log() const;

  private:
    std::unique_ptr<impl> Impl;
  };
  class linker{
    class impl;
    class interface;
  public:
    linker();



  private:
    std::unique_ptr<impl> Impl;
  };

  class function{
    class impl;
  public:
  private:
    std::unique_ptr<impl> Impl;
  };
  class global{
    class impl;
  public:
  private:
    std::unique_ptr<impl> Impl;
  };
  class module{
    class impl;
    class ptx_source;
    class cubin;
    class fatbin;
  public:
  private:
    std::unique_ptr<impl> Impl;
  };
}

#endif//CUDA_FUNCTIONS_RTC_H
