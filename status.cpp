//
// Created by Maxwell on 2020-09-29.
//

#include "status.h"

#include <llvm/ADT/DenseMap.h>

namespace {
  class resolve_action;
  class callback_action;
  class propagate_action;
  class suppress_action;
  class ignore_action;
  class cancel_action;
  class retry_action;
  class kill_action;
  class terminate_action;
}



class cu::status_handler::action{
  friend ::resolve_action;
  class callback;
  class propagate;
  class suppress;
  class ignore;
  class cancel;
  class retry;
  class kill;
  class terminate;
};

namespace {}

class cu::handler_action::interface{};


class status_handler{
  class action;
};
class handler_action{
  class interface;
public:
  enum class kind{
    resolve,
    callback,
    propagate,
    suppress,
    ignore,
    cancel,
    retry,
    kill,
    terminate
  };
};