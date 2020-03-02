#pragma once

#include <memory>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { 
namespace distributed::autograd {
class DistAutogradContext;
}  // namespace distributed

namespace autograd {

struct TORCH_API AccumulateGrad : public Node {
  using DistAutogradContextPtr =
      std::shared_ptr<distributed::autograd::DistAutogradContext>;

  explicit AccumulateGrad(Variable variable_);

  variable_list apply(variable_list&& grads) override;

  void setDistAutogradContext(const DistAutogradContextPtr& ctx);

  Variable variable;

  // The node is being evaluated under distributed autograd if
  // this member is not null.
  DistAutogradContextPtr distAutogradCtx;
};

}} // namespace torch::autograd
