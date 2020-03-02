#include <torch/csrc/distributed/autograd/functions/dist_accumulate_grad.h>

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/variable.h>

namespace torch::distributed::autograd {

DistAccumulateGrad::DistAccumulateGrad(
    AccumulateGrad&& accumulateGrad,
    std::shared_ptr<DistAutogradContext> autogradContext)
    : Node(accumulateGrad.sequence_nr()),
      variable_(std::move(accumulateGrad.variable)),
      autogradContext_(std::move(autogradContext)) {
  move_from(std::move(accumulateGrad));
  // Replace the variable's reference to AccumulateGrad.
  torch::autograd::impl::set_grad_accumulator(
      variable_, this->shared_from_this());
}

variable_list DistAccumulateGrad::apply(variable_list&& grads) {
  // TODO: Rebase to https://github.com/pytorch/pytorch/pull/33214
  // XXX: this method is not thread-safe!
  torch::autograd::check_input_variables("DistAccumulateGrad", grads, 1, 0);

  if (!grads[0].defined())
    return {};
  if (variable_.grad_fn())
    throw std::logic_error(
        "leaf variable has been moved into the graph interior");
  if (!variable_.requires_grad())
    return {};

  auto new_grad = std::move(grads[0]);
  for (auto& hook : torch::autograd::impl::hooks(variable_)) {
    new_grad = (*hook)({new_grad})[0];
  }

  autogradContext_->accumulateGrad(variable_, new_grad);
  return variable_list();
}

} // namespace torch::distributed::autograd
