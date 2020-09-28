//
// Created by maxwell on 2020-09-27.
//

#ifndef CUDA_FUNCTIONS_EVENT_H
#define CUDA_FUNCTIONS_EVENT_H

#include <cuda.h>
#include <cassert>

#define UNWRAP_I(arg) arg
#define UNWRAP_ UNWRAP_I
#define UNWRAP_FIRST(...) UNWRAP_ __VA_ARGS__
#define TAKE_FIRST_I(arg) arg DISCARD_TAIL_
#define TAKE_FIRST_ TAKE_FIRST_I
#define TAKE_FIRST(...) TAKE_FIRST_ __VA_ARGS__
#define DISCARD_TAIL(...)
#define DISCARD_TAIL_ DISCARD_TAIL

#define STRINGIFY_I(...) #__VA_ARGS__
#define STRINGIFY_ STRINGIFY_I
#define STRINGIFY(...) STRINGIFY_(__VA_ARGS__)



#if defined(NDEBUG)
#define cuda_assert(...) __VA_ARGS__;
#else
#define cuda_assert(...) if(auto _err = UNWRAP_FIRST(__VA_ARGS__)){ \
    const char* Name, *String;                  \
    assert(!cuGetErrorName(_err, &Name));             \
    assert(!cuGetErrorString(_err, &String));         \
    std::cerr << Name << ": " << String << "\nRaised in call to " << STRINGIFY(TAKE_FIRST(__VA_ARGS__)) << std::endl; \
    abort();                                          \
  }
#endif

#if __cplusplus <= 201703ull
namespace std{
  template <typename = void>
  struct coroutine_handle;
}
#else
#include <coroutine>
#endif

namespace cu{

  void sync(CUstream Stream);


  class event{
    CUevent Handle;
  public:
    explicit event(CUstream Stream) : Handle(){
      cuEventCreate(&Handle, CU_EVENT_DISABLE_TIMING);
      cuEventRecord(Handle, Stream);
    }
    event(event&& Other) noexcept : Handle(Other.Handle){
      Other.Handle = nullptr;
    }
    ~event(){
      if(Handle)
        cuEventDestroy(Handle);
    }

    [[nodiscard]] bool ready() const noexcept{
      return cuEventQuery(Handle) == CUDA_SUCCESS;
    }


#if __cplusplus > 201703ull
    /*awaitable operator co_await() const noexcept{
      return awaitable{.Handle = Handle};
    }*/

#endif
  };
}


#endif//CUDA_FUNCTIONS_EVENT_H
