//
// Created by Maxwell on 2020-09-29.
//

#include "include/util/custring.h"

#include <string>
#include <string_view>
#include <llvm/ADT/SmallString.h>

#include <memory_resource>
#include <cstdlib>
#include <llvm/Support/FormatVariadic.h>

class logged_resource : public std::pmr::memory_resource{
  llvm::buffer_ostream OS{llvm::outs()};
  std::atomic_size_t  OpNum = 0;
  std::atomic_size_t TotalAllocated = 0;
  std::atomic_size_t TotalAllocations = 0;
  std::atomic_size_t TotalDeallocated = 0;
  std::atomic_size_t TotalDeallocations = 0;

  void* do_allocate(size_t Bytes, size_t Align) override{
    OS << llvm::formatv(" | #{0,-2}:  {1,-10} {2,6} bytes\n", OpNum++, "allocate", Bytes);
    TotalAllocated.fetch_add(Bytes, std::memory_order_relaxed);
    TotalAllocations.fetch_add(1, std::memory_order_relaxed);
    return _aligned_malloc(Bytes, Align);
  }
  void do_deallocate(void* Ptr, size_t Bytes, size_t Align) override{
    OS << llvm::formatv(" | #{0,-2}:  {1,-10} {2,6} bytes\n", OpNum++, "deallocate", Bytes);
    TotalDeallocated.fetch_add(Bytes, std::memory_order_relaxed);
    TotalDeallocations.fetch_add(1, std::memory_order_relaxed);
    return _aligned_free(Ptr);
  }
  [[nodiscard]] bool do_is_equal(const memory_resource& That) const noexcept override{
    return this == &That;
  }

public:

  logged_resource(){
    this->OS << llvm::formatv("===[ Allocator {0} Created   ]===\n |\n", this);
  }
  explicit logged_resource(llvm::raw_ostream& OS) : OS(OS){
    this->OS << llvm::formatv("===[ Allocator {0} Created   ]===\n |\n", this);
  }

  ~logged_resource() override {
    OS << " |\n |\n |\n";
    OS << llvm::formatv(" | total allocated:   {0,8} bytes\n"
                             " | total deallocated: {1,8} bytes\n"
                        " |\n"
                        " | total allocations:   {2}\n"
                        " | total deallocations: {3}\n |\n |\n |\n",
                            TotalAllocated.load(),
                            TotalDeallocated.load(),
                            TotalAllocations.load(),
                            TotalDeallocations.load());

    this->OS << llvm::formatv("===[ Allocator {0} Destroyed ]===\n\n", this);
  }
};




int main(){
  using cu::string, cu::fixed_string;

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


  logged_resource MemoryResource{};

  /*fixed_string FixStr("Hello my darling...", &MemoryResource);
  fixed_string fixStr2("Hello my baby...", &MemoryResource);
  fixed_string fixStr3("Hello my ragtime gaaaaAAlll", &MemoryResource);
  llvm::outs() << FixStr << "\n";
  llvm::outs() << fixStr2 << "\n";
  llvm::outs() << fixStr3 << "\n";
  fixStr2 = fixStr3;
  fixStr3 = FixStr;
  fixed_string fixStr4(640, ' ', &MemoryResource);

  FixStr = fixStr4;
*/
  std::pmr::string LoggedString("PmrString here!", &MemoryResource);
  LoggedString += " Some more stuff here lmao...";
  LoggedString += std::pmr::string("PmrStringA", &MemoryResource) + std::pmr::string("PmrStringB", &MemoryResource);

  std::cout << LoggedString << std::endl;

  /*llvm::outs() << FixStr << "\n";
  llvm::outs() << fixStr2 << "\n";
  llvm::outs() << fixStr3 << "\n\n\n";*/
}
