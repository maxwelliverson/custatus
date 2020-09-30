//
// Created by Maxwell on 2020-09-29.
//

#include "custring.h"
#include <llvm/ADT/StringRef.h>

void cu::string::string_base::destroy() noexcept {
  delete this;
}