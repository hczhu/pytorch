#pragma once

#include <memory>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/autograd/context/context.h>

namespace torch::distributed::autograd {

// Forward declarations.
class DistAutogradContext;

struct TORCH_API DistAccumulateGrad : public torch::autograd::Node {
  explicit DistAccumulateGrad(
      std::shared_ptr<DistAutogradContext> autogradContext);

  variable_list apply(variable_list&& grads) override;

 private:
  std::shared_ptr<DistAutogradContext> autogradContext_;
};

} // namespace torch::distributed::autograd
