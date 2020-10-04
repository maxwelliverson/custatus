//
// Created by Maxwell on 2020-09-28.
//

#include "include/rtc.h"
#include "include/util/status.h"

#include <algorithm>
#include <iostream>
#include <system_error>
#include <range/v3/action/unique.hpp>
#include <nvPTXCompiler.h>
#include <nvrtc.h>
#include <boost/outcome.hpp>
#include <llvm/ADT/DirectedGraph.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringExtras.h>
#include <fatbinary.h>
#include <optional>
#include <magic_enum.hpp>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning (disable : 4624)
#endif

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

  inline constexpr nvrtc_domain nvrtc_domain_v{};

  class source_template{};
  struct source_code_interface{

  };
}

template <>
struct cu::status_enum<nvrtcResult>{
  static constexpr const nvrtc_domain& domain() noexcept{
    return nvrtc_domain_v;
  }
};

struct cu::rtc::source_code::interface{
  bool (*const is_complete)();
  bool (*const is_empty)();
  std::span<const llvm::StringRef> (*const template_arguments)(const cu::rtc::source_code* Src);
  cu::rtc::source_code::impl*(*const clone)(const cu::rtc::source_code* Src);
  void(*const substitute)(cu::rtc::source_code* Src, std::error_code& Error, llvm::StringRef Key, llvm::StringRef Value);
  void(*const llvm_print)(const cu::rtc::source_code* Src, llvm::raw_ostream& OS);
  void(*const std_print)(const cu::rtc::source_code* Src, std::ostream& OS);
};

namespace{
  inline bool always_false() noexcept{
    return false;
  }
  inline bool always_true() noexcept{
    return true;
  }
  inline void invalid_op(cu::rtc::source_code* Src, std::error_code& Error, llvm::StringRef Key, llvm::StringRef Value) noexcept{
    Error = std::make_error_code(std::errc::operation_not_supported);
  }
}

class cu::rtc::source_code::impl{
public:
  const cu::rtc::source_code::interface* Interface;
  explicit impl(const cu::rtc::source_code::interface* Interface) noexcept : Interface(Interface){}
};
class cu::rtc::source_code::empty_source : public cu::rtc::source_code::impl{
  inline static std::span<const llvm::StringRef> template_arguments_impl(const cu::rtc::source_code*) noexcept{
    return {};
  }
  inline static cu::rtc::source_code::impl* clone_impl(const cu::rtc::source_code* Source) noexcept{
    return Source->Impl.get();
  }
  inline static void llvm_print_impl(const cu::rtc::source_code* Src, llvm::raw_ostream& OS){}
  inline static void std_print_impl(const cu::rtc::source_code* Src, std::ostream& OS){}
  inline constexpr static cu::rtc::source_code::interface EmptyInterface{
      .is_complete = always_false,
      .is_empty = always_true,
      .template_arguments = template_arguments_impl,
      .clone = clone_impl,
      .substitute = invalid_op,
      .llvm_print = llvm_print_impl,
      .std_print = std_print_impl
  };
  empty_source() noexcept : cu::rtc::source_code::impl(&EmptyInterface){}

public:
  inline static empty_source* get(){
    static empty_source Source{};
    return &Source;
  }
};
class cu::rtc::source_code::complete_source : public cu::rtc::source_code::impl{
  inline static std::span<const llvm::StringRef> template_arguments_impl(const cu::rtc::source_code*) noexcept{
    return {};
  }
  inline static cu::rtc::source_code::impl* clone_impl(const cu::rtc::source_code* Source) noexcept{
    return new complete_source(static_cast<const complete_source&>(*Source->Impl));
  }
  inline static void llvm_print_impl(const cu::rtc::source_code* Src, llvm::raw_ostream& OS){
    OS << static_cast<const complete_source&>(*Src->Impl.get()).Source;
  }
  inline static void std_print_impl(const cu::rtc::source_code* Src, std::ostream& OS){
    auto&& Source = static_cast<const complete_source&>(*Src->Impl.get()).Source;
    OS << std::string_view{Source.data(), Source.size()};
  }
  inline constexpr static cu::rtc::source_code::interface CompleteInterface{
    .is_complete = always_true,
    .is_empty = always_false,
    .template_arguments = template_arguments_impl,
    .clone = clone_impl,
    .substitute = invalid_op,
    .llvm_print = llvm_print_impl,
    .std_print = std_print_impl
  };

  llvm::SmallString<0> Source;

public:
  explicit complete_source(llvm::StringRef String) noexcept
      : cu::rtc::source_code::impl(&CompleteInterface),
        Source(String){}
};
class cu::rtc::source_code::source_template : public cu::rtc::source_code::impl{
  inline static std::span<const llvm::StringRef> template_arguments_impl(const cu::rtc::source_code* Src) noexcept{
    return static_cast<const cu::rtc::source_code::source_template*>(Src->Impl.get())->Params;
  }
  inline static cu::rtc::source_code::impl* clone_impl(const cu::rtc::source_code* Source) noexcept{
    return new source_template(*static_cast<const source_template*>(Source->Impl.get()));
  }
  inline static void llvm_print_impl(const cu::rtc::source_code* Src, llvm::raw_ostream& OS){
    for(auto&& Chunk : static_cast<const source_template*>(Src->Impl.get())->Chunks){
      if(!Chunk.Param.empty())
        OS << "${" << Chunk.Param << "}";
      OS << Chunk.Source;
    }
  }
  inline static void std_print_impl(const cu::rtc::source_code* Src, std::ostream& OS){
    for(auto&& Chunk : static_cast<const source_template*>(Src->Impl.get())->Chunks){
      if(!Chunk.Param.empty())
        OS << "${" << std::string_view{Chunk.Param.data(), Chunk.Param.size()} << "}";
      OS << std::string_view{Chunk.Source.data(), Chunk.Source.size()};
    }
  }

  inline static void substitute_impl(cu::rtc::source_code* Src, std::error_code& Error, llvm::StringRef Key, llvm::StringRef Value) noexcept{
    auto* This = static_cast<source_template*>(Src->Impl.get());
    if(Key.empty()){
      Error = std::make_error_code(std::errc::invalid_argument);
      return;
    }
    const auto* OldEnd = This->Chunks.end();
    auto* Begin = This->Chunks.begin();
    auto* End = This->Chunks.end();
    while(Begin != End){
      if(Key.equals(Begin->Param)){
        (Value + Begin->Source).toVector((Begin - 1)->Source);
        This->Chunks.erase(Begin);
        --End;
      } else {
        ++Begin;
      }
    }
    if(End == OldEnd){
      Error = std::make_error_code(std::errc::invalid_argument);
      return;
    }
    for(const auto& Param : This->Params){
      if(Key == Param) {
        This->Params.erase(&Param);
        break;
      }
    }
    if(This->Params.empty())
      Src->Impl.reset(new complete_source(This->Chunks[0].Source));
    Error = std::error_code();
  }

  inline constexpr static cu::rtc::source_code::interface TemplateInterface{
      .is_complete = always_false,
      .is_empty = always_false,
      .template_arguments = template_arguments_impl,
      .clone = clone_impl,
      .substitute = substitute_impl,
      .llvm_print = llvm_print_impl,
      .std_print = std_print_impl
  };

  struct SourceChunk{
    llvm::SmallString<8> Param;
    llvm::SmallString<8> Source;

    SourceChunk() = default;
    explicit SourceChunk(llvm::StringRef Str){
      if(Str[0] == '{') {
        auto&& [Front, Back] = llvm::getToken(Str.drop_front(), "}");
        Param = Front;
        Source = Back.drop_front();
      }
      else
        Source = Str;
    }
  };

  llvm::SmallVector<llvm::StringRef, 4> Params;
  llvm::SmallVector<SourceChunk, 4> Chunks;

public:
  explicit source_template(const llvm::StringRef String) noexcept : cu::rtc::source_code::impl(&TemplateInterface){
    llvm::StringRef Front, Back = String;
    while(!Back.empty()){
      std::tie(Front, Back) = getToken(Back, "$");
      Params.push_back(Chunks.emplace_back(Front).Param);
    }
    if(Params.size() > 1){
      llvm::sort(Params);
      auto it = ranges::unique(Params);
      Params.erase(it, Params.end());
    }
  }
};


class cu::rtc::compiler::impl{
  friend compiler;

  enum class cpp_op{
    define,
    undef,
    include_path,
    pre_include
  };
  class cpp_argument{
    union{
      uint32_t Operation;
      fixed_string<> String;
    };
  public:
    cpp_argument(cpp_op Op, std::string_view Argument) : String(Argument){
      Operation = (uint32_t)Op;
    }
    cpp_argument(const cpp_argument& Other) : String(Other.String){
      Operation = Other.Operation;
    }
    cpp_argument(cpp_argument&& Other) noexcept : String(std::move(Other.String)){
      Operation = Other.Operation;
    }
    ~cpp_argument(){
      this->String.~fixed_string<>();
    }
    [[nodiscard]] std::string_view string() const noexcept{
      return std::string_view(String.data(), String.size());
    }
    [[nodiscard]] cpp_op operation() const noexcept{
      return static_cast<cpp_op>(Operation);
    }
  };


  mutable cu::status_code Status;

  fixed_string<> Name;
  fixed_string<> Body;

  nvPTXCompilerHandle PtxToMC;
  nvrtcProgram CxxToPtx;
  llvm::SmallVector<header, 2> HeaderFiles;
  llvm::SmallVector<cpp_argument, 2> CppArguments;
  struct{
    uint32_t maxregcount = 0;
    options::standard std : 8 = options::standard::cxx17;
    options::arch gpu_architecture : 8 = options::arch::compute_52;
    uint32_t builtin_move_forward : 1 = true;
    uint32_t builtin_initializer_list : 1 = true;
    uint32_t disable_warnings : 1 = false;
    uint32_t restrict : 1 = false;
    uint32_t device_as_default_execution_space : 1 = true;
    uint32_t ftz : 1 = false;
    uint32_t prec_sqrt : 1 = true;
    uint32_t prec_div : 1 = true;
    uint32_t fmad : 1 = true;
    uint32_t use_fast_math : 1 = false;
    uint32_t extra_device_vectorization : 1 = false;
    uint32_t relocatable_device_code : 1 = false;
    uint32_t extensible_whole_program : 1 = false;
    uint32_t device_debug : 1 = false;
    uint32_t generate_line_info : 1 = false;
    uint32_t padding : 1 = false;
  } opt;
  static_assert(sizeof(opt) == sizeof(void*));

public:
  using option = ::cu::rtc::options::compiler;

  impl() noexcept
      : Status(cu::generic_code::success),
        PtxToMC(),
        CxxToPtx(),
        CppArguments(),
        opt(){}


  [[nodiscard]] bool get(option Opt) const{
    switch(Opt){
      case options::builtin_move_forward:
        return opt.builtin_move_forward;
      case options::builtin_initializer_list:
        return opt.builtin_initializer_list;
      case options::warnings:
        return !opt.disable_warnings;
      case options::pointer_aliasing:
        return !opt.restrict;
      case options::device_as_default_execution_space:
        return opt.device_as_default_execution_space;
      case options::denormal_values:
        return !opt.ftz;
      case options::fma_instructions:
        return opt.fmad;
      case options::precise_sqrt:
        return opt.prec_sqrt;
      case options::precise_division:
        return opt.prec_div;
      case options::fast_math:
        return opt.use_fast_math;
      case options::extra_device_vectorization:
        return opt.extra_device_vectorization;
      case options::linking:
        return opt.relocatable_device_code;
      case options::debug_info_generation:
        return opt.device_debug;
      case options::line_info_generation:
        return opt.generate_line_info;
      case options::link_time_optimizations:
        return opt.extensible_whole_program;
      default:
        Status = cu::generic_code::invalid_argument;
        return false;
    }
  }
  void set(option Opt, bool IsEnabled){
    switch(Opt){
      case options::builtin_move_forward:
        opt.builtin_move_forward = IsEnabled;
        break;
      case options::builtin_initializer_list:
        opt.builtin_initializer_list = IsEnabled;
        break;
      case options::warnings:
        opt.disable_warnings = !IsEnabled;
        break;
      case options::pointer_aliasing:
        opt.restrict = !IsEnabled;
        break;
      case options::device_as_default_execution_space:
        opt.device_as_default_execution_space = IsEnabled;
        break;
      case options::denormal_values:
        opt.ftz = !IsEnabled;
        break;
      case options::fma_instructions:
        opt.fmad = IsEnabled;
        break;
      case options::precise_sqrt:
        opt.prec_sqrt = IsEnabled;
        break;
      case options::precise_division:
        opt.prec_div = IsEnabled;
        break;
      case options::fast_math:
        opt.use_fast_math = IsEnabled;
        break;
      case options::extra_device_vectorization:
        opt.extra_device_vectorization = IsEnabled;
        break;
      case options::linking:
        opt.relocatable_device_code = IsEnabled;
        break;
      case options::debug_info_generation:
        opt.device_debug = IsEnabled;
        break;
      case options::line_info_generation:
        opt.generate_line_info = IsEnabled;
        break;
      case options::link_time_optimizations:
        opt.extensible_whole_program = IsEnabled;
      default:
        Status = generic_code::invalid_argument;
    }
  }
  void set(options::default_execution_space ExecutionSpace){
    opt.device_as_default_execution_space = (ExecutionSpace == options::default_execution_space::device);
  }


  void target(options::standard Std){
    opt.std = Std;
  }
  void target(options::arch Arch){
    opt.gpu_architecture = Arch;
  }
  void set_max_register_count(uint32_t MaxRegCount){
    opt.maxregcount = MaxRegCount;
  }


  void define(std::string_view PPMacro){
    CppArguments.emplace_back(cpp_op::define, PPMacro);
  }
  void undef(std::string_view PPMacro){
    CppArguments.emplace_back(cpp_op::undef, PPMacro);
  }
  void search_path(std::string_view Path){
    CppArguments.emplace_back(cpp_op::include_path, Path);
  }
  void include(std::string_view Header){
    CppArguments.emplace_back(cpp_op::pre_include, Header);
  }

  module compile(){
    auto HeaderCount = (int) HeaderFiles.size();
    auto HeaderNames = (const char**)alloca(HeaderCount * sizeof(void*));
    auto HeaderSources = (const char**)alloca(HeaderCount * sizeof(void*));
    size_t Index = 0;
    for(auto Header : HeaderFiles){
      HeaderNames[Index] = Header.Name.data();
      HeaderSources[Index] = Header.Body.data();
      Index += 1;
    }

    Status = nvrtcCreateProgram(&CxxToPtx,
                                Body.data(),
                                Name.data(),
                                HeaderCount,
                                HeaderSources,
                                HeaderNames);

    static_assert(magic_enum::detail::values_v<option>[0] == option::builtin_move_forward);

    llvm::SmallVector<const char*, 16> Options;
    for(auto Option : magic_enum::enum_values<option>()){
      if(auto String = get_option_string(Option))
        Options.push_back(String.value());
    }

    auto Hello = magic_enum::enum_values<option>();



    nvrtcCompileProgram(CxxToPtx, );


    return cu::rtc::module();
  }

private:
    inline std::optional<const char*> get_option_string(option Opt) const noexcept{
      switch(Opt){
        case options::builtin_move_forward:
          if(!opt.builtin_move_forward)
            return "--builtin-move-forward=false";
          break;
        case options::builtin_initializer_list:
          if(!opt.builtin_initializer_list)
            return "--builtin-initializer-list=false";
          break;
        case options::warnings:
          if(opt.disable_warnings)
            return "-w";
          break;
        case options::pointer_aliasing:
          if(opt.restrict)
            return "-restrict";
          break;
        case options::device_as_default_execution_space:
          if(opt.device_as_default_execution_space)
            return "-default-device";
          break;
        case options::denormal_values:
          if(opt.ftz)
            return "--ftz=true";
          return "--ftx=false";
        case options::fma_instructions:
          if(opt.fmad)
            return "--fmad=true";
          return "--fmad=false";
        case options::precise_sqrt:
          if(opt.prec_sqrt)
            return "--prec-sqrt=true";
          return "--prec-sqrt=false";
        case options::precise_division:
          if(opt.prec_div)
            return "--prec-div=true";
          return "--prec-div=false";
        case options::fast_math:
          if(opt.use_fast_math)
            return "-use_fast_math=true";
          break;
        case options::extra_device_vectorization:
          if(opt.extra_device_vectorization)
            return "-extra-device-vectorization";
          break;
        case options::linking:
          if(opt.relocatable_device_code)
            return "-dc";
          break;
        case options::debug_info_generation:
          if(opt.device_debug)
            return "-G";
          break;
        case options::line_info_generation:
          if(opt.generate_line_info)
            return "-lineinfo";
          break;
        case options::link_time_optimizations:
          if(opt.extensible_whole_program)
            return "-ewp";
          break;
      }
      return std::nullopt;
    }
};
class cu::rtc::linker::impl{

};

cu::rtc::compiler::compiler() : Impl(new impl()){}
cu::rtc::compiler::~compiler() = default;
void cu::rtc::compiler::enable(cu::rtc::compiler::option Opt) {
  Impl->set(Opt, true);
}
void cu::rtc::compiler::disable(cu::rtc::compiler::option Opt) {
  Impl->set(Opt, false);
}
bool cu::rtc::compiler::get(cu::rtc::compiler::option Opt) const {
  return Impl->get(Opt);
}
void cu::rtc::compiler::set(cu::rtc::compiler::option Opt, bool IsEnabled) {
  Impl->set(Opt, IsEnabled);
}
void cu::rtc::compiler::target(options::standard Std) {
  Impl->target(Std);
}
void cu::rtc::compiler::target(options::arch Arch) {
  Impl->target(Arch);
}
void cu::rtc::compiler::set_max_register_count(uint32_t MaxRegCount) {
  Impl->set_max_register_count(MaxRegCount);
}
void cu::rtc::compiler::define(std::string_view PPMacro) {
  Impl->define(PPMacro);
}
void cu::rtc::compiler::undef(std::string_view PPMacro) {
  Impl->undef(PPMacro);
}
void cu::rtc::compiler::search_path(std::string_view PPMacro) {
  Impl->search_path(PPMacro);
}
void cu::rtc::compiler::include(std::string_view Header) {
  Impl->include(Header);
}
void cu::rtc::compiler::set(options::default_execution_space ExecutionSpace) {
  Impl->set(ExecutionSpace);
}
cu::fixed_string<> cu::rtc::compiler::log() const {
  size_t LogSize = 8;
  Impl->Status = nvrtcGetProgramLogSize(Impl->CxxToPtx, &LogSize);
  if(Impl->Status != NVRTC_SUCCESS)
    return "";
  fixed_string Ret(LogSize);
  Impl->Status = nvrtcGetProgramLog(Impl->CxxToPtx, Ret.data());
  return std::move(Ret);
}
cu::rtc::module cu::rtc::compiler::compile() const {

}


[[nodiscard]] bool cu::rtc::source_code::empty() const noexcept{
  return Impl->Interface->is_empty();
}
bool cu::rtc::source_code::is_complete() const noexcept {
  return Impl->Interface->is_complete();
}
std::span<const llvm::StringRef> cu::rtc::source_code::template_arguments() const {
  return Impl->Interface->template_arguments(this);
}
cu::rtc::source_code::source_code() : Impl(empty_source::get()){}
cu::rtc::source_code::source_code(llvm::StringRef String)
    : Impl(!String.empty() ? (String.contains('$') ? (impl*)new source_template(String) : new complete_source(String)) : empty_source::get()){}
cu::rtc::source_code::source_code(std::string_view String) : source_code(llvm::StringRef(String.data(), String.size())) {}
cu::rtc::source_code::source_code(const cu::rtc::source_code &Other) : Impl(Other.Impl->Interface->clone(&Other)){}
cu::rtc::source_code::source_code(cu::rtc::source_code &&Other) noexcept : Impl(std::move(Other.Impl)){
  Impl.reset(empty_source::get());
}
cu::rtc::source_code::~source_code() {
  if(empty())
    auto _ = Impl.release();
}

void cu::rtc::source_code::print(std::ostream &OS) const {
  Impl->Interface->std_print(this, OS);
}
void cu::rtc::source_code::print(llvm::raw_ostream &OS) const {
  Impl->Interface->llvm_print(this, OS);
}
std::error_code cu::rtc::source_code::substitute(llvm::StringRef Key, llvm::StringRef Substitution) {
  std::error_code Err;
  Impl->Interface->substitute(this, Err, Key, Substitution);
  return Err;
}


