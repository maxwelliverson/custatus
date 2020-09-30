//
// Created by Maxwell on 2020-09-29.
//

#include "custring.h"

#include <string>
#include <string_view>
#include <llvm/ADT/SmallString.h>

int main(){
  using cu::string;

  string Empty;

  assert(Empty.empty());

  Empty = "No longer empty...";

  assert(!Empty.empty());

  string Str = "Maxwell Iverson";
  string Utf8Str = u8"Maxwell Iverson";
  string WideStr = L"Maxwell Iverson";
  string stdStr = std::string("Maxwell Iverson");
  string llvmStr = llvm::SmallString<4>{"Maxwell Iverson"};

  size_t Size = Empty.size();
  Size = Str.size();
  Size = Utf8Str.size();
  Size = stdStr.size();
  Size = llvmStr.size();

  assert(!Str.empty());
  assert(!Utf8Str.empty());
  assert(!WideStr.empty());
  assert(!stdStr.empty());
  assert(!llvmStr.empty());

  //assert(Str == Utf8Str);
  //assert(Str == WideStr);
  assert(Str == llvmStr);
  assert(Str == stdStr);

  string OtherStr = "Maxwell James";
  string OtherWideStr = L"Maxwell James";

  std::cout << "Str: " << Str << "\n";
  //std::wcout << "WideStr: " << WideStr << "\n";
  std::cout << "stdStr: " << stdStr << "\n";
  llvm::outs() << "llvmStr: " << llvmStr << "\n";
  llvm::outs() << "OtherStr: " << OtherStr << "\n\n";



  assert(Str < OtherStr);
  assert((llvmStr <=> OtherWideStr) == std::partial_ordering::unordered);
}
