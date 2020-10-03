//
// Created by Maxwell on 2020-09-29.
//

#include "include/rtc.h"

#include <llvm/Support/raw_ostream.h>

int main(){
  using namespace cu::rtc;
  source_code Source = "";
  source_code CompleteSource = some_source;
  source_code TemplateSource = get_function_ptr;

  llvm::outs() << "Empty Source: " << Source << "\n\n";
  llvm::outs() << "Complete Source: " << CompleteSource << "\n\n";
  llvm::outs() << "Template Source: " << TemplateSource << "\n\n";

  llvm::outs() << "\nEmpty Args:  ";
  for(auto&& Arg : Source.template_arguments())
    llvm::outs() << Arg << ", ";
  llvm::outs() << "\b\b\n\nComplete Args:  ";
  for(auto&& Arg : CompleteSource.template_arguments())
    llvm::outs() << Arg << ", ";
  llvm::outs() << "\b\b\n\nTemplate Args:  ";
  for(auto&& Arg : TemplateSource.template_arguments())
    llvm::outs() << Arg << ", ";
  llvm::outs() << "\b\b\n\n";

  if(auto Error = Source.substitute("SomeKey", "SomeValue")){
    llvm::errs() << "Empty: " << Error.message() << "\n\n";
  }
  if(auto Error = CompleteSource.substitute("SomeKey", "SomeValue")){
    llvm::errs() << "Complete: " << Error.message() << "\n\n";
  }
  if(auto Error = TemplateSource.substitute("_FnName", "pthread_attr_getpriority_np")){
    llvm::errs() << "Complete: " << Error.message() << "\n\n";
  }

  llvm::outs() << "New Template:\n\t" << TemplateSource << "\n\n";
  llvm::outs().flush();
}