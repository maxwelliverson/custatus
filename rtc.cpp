//
// Created by Maxwell on 2020-09-28.
//

#include "rtc.h"

#include <algorithm>
#include <iostream>
#include <system_error>
#include <range/v3/action/unique.hpp>
#include <boost/outcome.hpp>
#include <llvm/ADT/DirectedGraph.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/ilist.h>

namespace {
  class source_template{};
  struct source_code_interface{

  };
}

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
  class st_graph_node;
  class st_graph_edge : public llvm::DGEdge<st_graph_node, st_graph_edge>{};
  class st_graph_node : public llvm::DGNode<st_graph_node, st_graph_edge>{
    llvm::SmallString<8> String;
  public:

  };
  class source_template_node : public llvm::ilist_node<source_template_node>{
    llvm::SmallString<32> String;
  public:
    explicit source_template_node(llvm::StringRef String) noexcept : String(String){}
  };

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
    Impl.release();
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
