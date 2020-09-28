//
// Created by maxwell on 2020-09-27.
//

#include "event.h"

#include <coroutine>
#include <iostream>
#include <chrono>
#include <thread>
/*

namespace std::experimental{
  using std::coroutine_handle, std::coroutine_traits;
}

*/

namespace {

  class timed_test{
    void(*CB)();
    std::chrono::microseconds Interval;
    cu::event Event;
  public:
    timed_test(cu::event Event, std::chrono::microseconds Interval, void(*CB)()) noexcept :
      CB(CB), Interval(Interval), Event(std::move(Event)){}

    [[nodiscard]] bool await_ready() const noexcept{
      return Event.ready();
    }
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> Handle) const noexcept{
      while(!Event.ready()){
        std::this_thread::sleep_for(Interval);
        CB();
      }
      return Handle;
    }
    void await_resume() const noexcept{

    }
  };

  class task{
  public:

    struct promise_type{
      task get_return_object() noexcept {
        return task(this);
      }
      [[nodiscard]] std::suspend_never initial_suspend() const noexcept{
        return {};
      }
      void return_void() const noexcept{}
      void unhandled_exception() const noexcept{}
      std::suspend_never final_suspend() const noexcept{
        return {};
      }
    };

  private:
    explicit task(promise_type* Prom) noexcept : Handle(std::coroutine_handle<promise_type>::from_promise(*Prom)){}

    std::coroutine_handle<promise_type> Handle;
  };



  inline task await_task(cu::event Event, std::chrono::microseconds Interval, void(*CB)()){
    co_await timed_test(std::move(Event), Interval, CB);
  }
}






void cu::sync(CUstream Stream) {
  using namespace std::chrono_literals;
  auto Task = await_task(event(Stream), 1000us, []{ puts("."); });
}