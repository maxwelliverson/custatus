//
// Created by maxwe on 2020-10-03.
//

#ifndef CUDA_FUNCTIONS_ALLOCATOR_H
#define CUDA_FUNCTIONS_ALLOCATOR_H

#include <cstddef>

namespace opal{
  namespace containers{
    template <typename T>
    class vector;
    template <typename T, size_t N>
    class static_vector;

    class string;
  }
  namespace utils{
    class version;
    class logger;
  }
  namespace memory{
    class resource{};
    class allocator{};
  }
  namespace status{
    class code{};
    class error{};
    class handler{};
    template <typename T>
    class maybe;
  }
  namespace reflect{}
  namespace test{}
  namespace async{
    class token{};
    class message{};
    class mailbox{};
  }
}



#endif//CUDA_FUNCTIONS_ALLOCATOR_H
