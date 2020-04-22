#include <queue>

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/functions/dist_accumulate_grad.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::AccumulateGrad;
using torch::autograd::edge_list;
using torch::autograd::Engine;
using torch::autograd::FutureVariableList;
using torch::autograd::GraphRoot;
using torch::autograd::GraphTask;
using torch::autograd::Node;
using torch::autograd::ReadyQueue;
using torch::autograd::validate_outputs;
using torch::autograd::variable_list;

static constexpr char* kNumBackwardPasses = "num_current_backward_passes";
static constexpr char* kNumAutogradContexts = "num_autograd_contexts";

// This hook does 3 things:
//   1. Call pre hooks of the original AccumulateGrad to modify the input grad.
//   2. Accumuate the gard to RPC context.
//   3. Call post hooks of the original AccumulateGrad.
class DistAccumulateGradCaptureHook
    : public GraphTask::ExecInfo::Capture::GradCaptureHook {
 public:
  DistAccumulateGradCaptureHook(
      std::shared_ptr<AccumulateGrad> accumulateGrad,
      ContextPtr autogradContext)
      : accumulateGrad_(std::move(accumulateGrad)),
        autogradContext_(std::move(autogradContext)) {}

  at::Tensor operator()(const at::Tensor& grad) override {
    variable_list inputGrads = {grad};
    // It's intended that pre/post hooks are still called even if the grad is
    // undenfined here.
    for (const auto& hook : accumulateGrad_->pre_hooks()) {
      inputGrads = (*hook)(inputGrads);
    }

    // It is possible that the grad is not defined since a separate
    // invocation of the autograd engine on the same node might actually
    // compute this gradient.
    if (inputGrads[0].defined()) {
      // There are 3 internal references to 'inputGrads[0]' at this moment:
      //   1. 'inputGrads[0]' in this function.
      //   2. 'graph_task->captured_vars_' on the callsite in the local engine.
      //   3. 'InputBuffer& inputs' on the callsite as the inputs of the
      //   function node.
      autogradContext_->accumulateGrad(
          accumulateGrad_->variable, inputGrads[0], 3 /* num_expected_refs */);
    }

    const variable_list kEmptyOuput;
    for (const auto& hook : accumulateGrad_->post_hooks()) {
      (*hook)(kEmptyOuput, inputGrads);
    }
    return inputGrads[0];
  }

 private:
  std::shared_ptr<AccumulateGrad> accumulateGrad_;
  ContextPtr autogradContext_;
};

DistEngine::DistEngine()
    : initializedContextIds_(), engine_(Engine::get_default_engine()) {}

DistEngine& DistEngine::getInstance() {
  // Leaky singleton to avoid module destructor race.
  static DistEngine* engine = new DistEngine();
  return *engine;
}

void DistEngine::validateRootsAndRetrieveEdges(
    const variable_list& roots,
    edge_list& rootEdges,
    variable_list& grads) {
  TORCH_CHECK(!roots.empty(), "No tensors provided for gradient computation.");
  TORCH_INTERNAL_ASSERT(rootEdges.empty());
  TORCH_INTERNAL_ASSERT(grads.empty());

  // Verify roots are all scalar and require gradients.
  for (const auto& root : roots) {
    TORCH_CHECK(
        root.requires_grad(), "requires_grad not set on: ", root.name());
    TORCH_CHECK(
        root.numel() == 1,
        root.name(),
        " is not a scalar, all roots need to be scalar");
    TORCH_CHECK(
        root.grad_fn(),
        root.name(),
        " does not have a valid gradient function.");

    // Compute the root edges and generate the appropriate gradients.
    rootEdges.push_back(torch::autograd::impl::gradient_edge(root));
    grads.push_back(at::ones_like(root, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
    LOG(ERROR) << "hcz: root edge from a scalar output: " << rootEdges.back().function->name() << " seq: " << rootEdges.back().function->sequence_nr();
  }

  // Validate rootEdges and grads.
  validate_outputs(
      rootEdges, grads, [](const std::string& msg) { return msg; });
}

// NB: this function modifies the autograd graph. More specifically it replaces
// all AccumulateGrad nodes by DistAccumulateGrad.
void DistEngine::computeDependencies(
    const ContextPtr& autogradContext,
    const edge_list& rootEdges,
    const variable_list& grads,
    const std::shared_ptr<Node>& graphRoot,
    edge_list& outputEdges,
    bool retainGraph) {
  TORCH_INTERNAL_ASSERT(graphRoot, "graphRoot is null!");
  LOG(ERROR) << "hcz: computeDependencies for graph root: @" << graphRoot.get() << " for " << rootEdges.size() << " root edges.";

  // Build a CPU ready queue that is used by the graphTask in local
  // autograd engine, since Distributed Autograd Engine calls
  // Engine::execute_with_graph_task in async mode instead of
  // Engine::execute, we allocate our own CPU ReadyQueue for
  // each GraphTask.
  // NB: We must allocate a separate ready queue for each GraphTask,
  // because the async mode of autograd engine loop through the
  // GraphTask's ready queue, so a single ready_queue cannot be
  // shared by different GraphTasks.
  auto cpu_ready_queue = std::make_shared<ReadyQueue>();

  // Build the graph task and graph root.
  auto graphTask = std::make_shared<GraphTask>(
      /* keep_graph */ retainGraph,
      /* create_graph */ false,
      /* depth */ 0,
      /* cpu_ready_queue */ cpu_ready_queue,
      /* exit_on_error */ true);

  // Run BFS to traverse the graph locally. The roots of the graph are
  // GraphRoot and all send functions for this autograd context.
  std::unordered_set<Node*> seen;
  std::queue<Node*> queue;
  queue.push(static_cast<Node*>(graphRoot.get()));

  auto sendFunctions = autogradContext->sendFunctions();

  // Add all the send functions to the queue as roots.
  for (const auto& mapEntry : sendFunctions) {
    // Increment 'outstanding_tasks_' for GraphTask for each send_function
    // since we want the local autograd engine to wait for all of them.
    graphTask->outstanding_tasks_++;
    queue.push(mapEntry.second.get());
  }

  edge_list recvBackwardEdges;
  // Traverse the graph.
  auto& dependencies = graphTask->dependencies_;
  // If a node is in this map, it should be replaced by its corresponding value
  // in the graph, i.e. all edges pointing to the key should point to the value.
  std::unordered_map<Node*, std::shared_ptr<DistAccumulateGrad>>
      accumulate_grad_replacements;
  while (!queue.empty()) {
    auto fn = queue.front();
    LOG(ERROR) << "hcz: processing function @" << fn->sequence_nr() << " in computeDependencies()";
    queue.pop();
    std::string line;
    for (int index = 0; index < fn->num_outputs(); ++index) {
      const auto& edge = fn->next_edge(index);
      std::string line = fn->name();
      if (auto nextFn = edge.function.get()) {
        line += "\n    ----> " + nextFn->name() + " @" + std::to_string(nextFn->sequence_nr());
        dependencies[nextFn] += 1;
        const bool wasInserted = seen.insert(nextFn).second;
        if (wasInserted) {
          // Seeing this function for the first time.
          queue.push(nextFn);

          if (nextFn->next_edges().empty()) {
            if (auto ptr = dynamic_cast<AccumulateGrad*>(nextFn)) {
              LOG(ERROR) << "AccumulateGrad has " << nextFn->num_inputs()
                         << " inputs and " << edge.function.use_count()
                         << " shared_ptr use_count for variable: "
                         << ptr->variable.toString();
            }
            TORCH_INTERNAL_ASSERT(
                dynamic_cast<AccumulateGrad*>(nextFn) ||
                dynamic_cast<RecvRpcBackward*>(nextFn));
            // We have found a leaf node which should be either AccumulateGrad
            // or RecvRpcBackward. Record the function
            // to ensure we don't execute it and instead accumulate the grads on
            // the autograd context. These functions would be passed in as the
            // 'outputs' parameter of the vanilla autograd engine.

            // We don't accumulate any grads in the context for RecvRpcBackward.
            // RecvRpcBackward is added as an output edge to indicate it is a
            // leaf node and this helps in properly computing dependencies for
            // the local autograd graph. Putting RecvRpcBackward in
            // 'outputEdges' means that this function needs to be executed
            // (inline with our assumption for FAST mode that all send/recv
            // functions are valid in the backward pass), and as a result all of
            //  its ancestors need to be executed as well.
            if (dynamic_cast<RecvRpcBackward*>(nextFn)) {
              recvBackwardEdges.emplace_back(edge);
            } else if (
                auto accumulateGradFn = dynamic_cast<AccumulateGrad*>(nextFn)) {
              LOG(ERROR) << "hcz: AccumulateGrad @"
                         << accumulateGradFn->sequence_nr() << " on variable @"
                         << accumulateGradFn->variable.unsafeGetTensorImpl();

              TORCH_INTERNAL_ASSERT(
                  !accumulateGradFn->variable.defined() ||
                      nextFn ==
                          torch::autograd::impl::try_get_grad_accumulator(
                              accumulateGradFn->variable)
                              .get(),
                  "The AccumulateGrad point in the variable should be the "
                  "same as the one in autograph");
              // Replace an AccumulateGrad node by DistAccumulateGrad,
              // because the former accumulates grads to the variable's '.grad'
              // without considering the context id. That may cause data race
              // on the '.grad', since multiple context ids may have grads for
              // the same '.grad'.
              auto distAccumulateGradFn = std::make_shared<DistAccumulateGrad>(
                  // The pointer to 'AccumulateGrad' in
                  // 'accumulateGradFn->variable' will be replaced by a pointer
                  // to 'distAccumulateGradFn'.
                  std::move(*accumulateGradFn),
                  autogradContext);
              LOG(ERROR) << "hcz: created a DistAccumulateGrad @" << distAccumulateGradFn.get() << " for AccumulateGrad @" << nextFn;
              accumulate_grad_replacements[nextFn] = distAccumulateGradFn;
              fn->next_edge(index).function = std::move(distAccumulateGradFn);
              TORCH_INTERNAL_ASSERT(edge == fn->next_edge(index));
            }
            outputEdges.emplace_back(edge);
          }
        } else {
          auto itr = accumulate_grad_replacements.find(nextFn);
          if (itr != accumulate_grad_replacements.end()) {
            LOG(ERROR) << "hcz: Replaced a AccumulateGrad @" << nextFn << " by DistAccumulateGrad @" << itr->second.get();
            fn->next_edge(index).function = itr->second;
          }
        }
      }
    }
    LOG(ERROR) << "hcz: Edges from node " << fn->name() << " @"
               << std::to_string(fn->sequence_nr()) << ": " << line;
  }
  for (const auto& [node, distAccumulateGrad] : accumulate_grad_replacements) {
    TORCH_INTERNAL_ASSERT(
        dependencies.count(distAccumulateGrad.get()) == 0,
        "DistAccumulateGrad nodes shouldn't be in dependencies");
    auto itr = dependencies.find(node);
    TORCH_INTERNAL_ASSERT(
        itr != dependencies.end(),
        "Replaced AccumulateGrad nodes should be in dependencies.");
    dependencies[distAccumulateGrad.get()] = itr->second;
    dependencies.erase(itr);
  }
  LOG(ERROR) << "hcz: BFS in computeDependencies() is done.";

  // hcz: AccumulateGrad nodes are marked at not-needed and the local autograd 
  // engine doesn't call 'applly()' at all. That's why the post hook installed
  // by DistributedDataParallel is not called.
  // Now lets compute which functions need to be executed. The algorithm is as
  // follows:
  // 1. Create a dummy GraphRoot which points to all 'send' functions for this
  //    context and the original graphRoot. Run 'init_to_execute' with the
  //    outputEdges and the dummy GraphRoot. This ensures we mark
  //    appropriate functions as needed if they are reachable only from a
  //    specific 'send' function locally and not necessarily from the provided
  //    roots.
  // 2. For all edges in 'outputEdges' which point to 'RecvRpcBackward', mark
  //    those functions as needed for execution. The reason for this is that
  //    'init_to_execute', will mark these as not needed. But 'RecvRpcBackward'
  //    is unique in the sense that we use it as a leaf node in graph to compute
  //    needed execution accurately, but unlike AccumulateGrad, we do need to
  //    execute this function.
  outputEdges.clear();
  if (!outputEdges.empty()) {
    // Compute 'needed execution' starting from all 'send' functions and the
    // original graphRoot.
    edge_list edges;
    // Create some dummy edges (input_nr not important for init_to_execute).
    for (const auto& mapEntry : sendFunctions) {
      edges.emplace_back(mapEntry.second, 0);
    }

    // Add the original graphRoot as an edge.
    edges.emplace_back(graphRoot, 0);

    // Create a dummy GraphRoot and run init_to_execute with it.
    GraphRoot dummyRoot(edges, {});
    // hcz: 'outputEdges' point to nodes/functions which have no edge.
    // Either AccumulateGrad or RecvRpcBackward for now.
    graphTask->init_to_execute(dummyRoot, outputEdges);
    for (auto& mapEntry : graphTask->exec_info_) {
      auto& execInfo = mapEntry.second;
      if (!execInfo.captures_) {
        continue;
      }
      auto fn = mapEntry.first;
      // There may be nodes other than 'AccumulateGrad', e.g. RecvRPCBackward,
      // to be captured.
      if (auto accumulateGradFn = dynamic_cast<AccumulateGrad*>(fn)) {
        for (auto& capture : *execInfo.captures_) {
          capture.hooks_.push_back(
              std::make_unique<DistAccumulateGradCaptureHook>(
                  std::dynamic_pointer_cast<AccumulateGrad>(
                      accumulateGradFn->shared_from_this()),
                  autogradContext));
        }
      }
    }

    // Mark all 'RecvRPCBackward' as needing execution.
    for (const auto& recvBackwardEdge : recvBackwardEdges) {
      graphTask->exec_info_[recvBackwardEdge.function.get()].needed_ = true;
    }
    // Mark all output nodes as needed.
    for (const auto& outputEdge : outputEdges) {
      const auto func = outputEdge.function.get();
      graphTask->exec_info_[func].needed_ = true;
    }
  }

  // Run the whole autograd graph
  TORCH_INTERNAL_ASSERT(graphTask->exec_info_.empty());
  // Let autograd context take ownership of the GraphTask.
  autogradContext->setGraphTask(std::move(graphTask));
}

std::shared_ptr<rpc::FutureMessage> DistEngine::runEngineAndAccumulateGradients(
    const ContextPtr& autogradContext,
    const std::shared_ptr<Node>& graphRoot,
    const edge_list& outputEdges) {
  // Cleanup previous state for outstanding RPCs. Outstanding RPCs could be
  // lingering if we're running backward multiple times and some of the
  // passes ran into errors.
  autogradContext->clearOutstandingRpcs();

  LOG(ERROR) << "DistEngine::runEngineAndAccumulateGradients() with root @"
             << graphRoot.get() << " name: " << graphRoot->name();
  auto futureGrads = engine_.execute_with_graph_task(
      autogradContext->retrieveGraphTask(), graphRoot, /*async_mode=*/true);

  // Build a future that waits for the callbacks to execute (since callbacks
  // execute after the original future is completed). This ensures we return a
  // future that waits for all gradient accumulation to finish.
  auto accumulateGradFuture = std::make_shared<rpc::FutureMessage>();

  futureGrads->addCallback([autogradContext, outputEdges, accumulateGradFuture](
                               const FutureVariableList& futureGrads) {
    if (futureGrads.hasError()) {
      // Don't accumulate gradients if we receive an error.
      // We must add the node information here since DistEngine::execute
      // waits on accumulateGradFuture and will throw an exception once we
      // set the error below.
      std::string errorMsg = c10::str(
          "Error on Node ",
          DistAutogradContainer::getInstance().getWorkerId(),
          ": ",
          futureGrads.error()->what());
      accumulateGradFuture->setError(errorMsg);
      return;
    }

    try {
      const variable_list& grads = futureGrads.constValue();
      TORCH_INTERNAL_ASSERT(grads.size() == outputEdges.size());
      accumulateGradFuture->markCompleted(rpc::Message());
    } catch (std::exception& e) {
      accumulateGradFuture->setErrorIfNeeded(e.what());
    }
  });

  return accumulateGradFuture;
}

std::shared_ptr<rpc::FutureMessage> DistEngine::executeSendFunctionAsync(
    const ContextPtr& autogradContext,
    const std::shared_ptr<Node>& sendFunction,
    bool retainGraph) {
  std::unique_lock<std::mutex> lock(initializedContextIdsLock_);
  if (initializedContextIds_.find(autogradContext->contextId()) ==
      initializedContextIds_.end()) {
    edge_list outputEdges;
    // Pass in a dummy graphRoot since all send functions are the roots.
    auto dummyRoot = std::make_shared<GraphRoot>(edge_list(), variable_list());
    computeDependencies(
        autogradContext, {}, {}, dummyRoot, outputEdges, retainGraph);

    // Mark the autograd context id as initialized and unlock.
    initializedContextIds_.insert(autogradContext->contextId());
    lock.unlock();

    // Enqueue the current send function.
    auto graphTask = autogradContext->retrieveGraphTask();
    engine_.enqueue_blocked_task_on_cpu(torch::autograd::NodeTask(
        graphTask, sendFunction, torch::autograd::InputBuffer(0)));

    // Run the autograd engine.
    auto accumulateGradFuture = runEngineAndAccumulateGradients(
        autogradContext, dummyRoot, outputEdges);

    // Build the 'uber' future that waits for everything.
    auto callbackFuture = std::make_shared<rpc::FutureMessage>();

    accumulateGradFuture->addCallback(
        [autogradContext,
         callbackFuture](const rpc::FutureMessage& accumulateGradFuture) {
          try {
            if (accumulateGradFuture.hasError()) {
              // Perform cleanup at the end of the backward pass (before we mark
              // the future as completed).
              DistEngine::getInstance().cleanupBackwardPass(autogradContext);

              // Skip any further processing on errors.
              callbackFuture->setError(accumulateGradFuture.error()->what());
              return;
            }

            // Wait for all RPCs after the autograd engine is done.
            auto rpcFuture =
                autogradContext->clearAndWaitForOutstandingRpcsAsync();
            rpcFuture->addCallback([callbackFuture, autogradContext](
                                       const rpc::FutureMessage& rpcFuture) {
              try {
                // Perform cleanup at the end of the backward pass (before
                // we mark the future as completed).
                DistEngine::getInstance().cleanupBackwardPass(autogradContext);
              } catch (std::exception& e) {
                callbackFuture->setErrorIfNeeded(e.what());
                return;
              }

              // Finally mark the 'uber' future as completed.
              if (!rpcFuture.hasError()) {
                callbackFuture->markCompleted(rpc::Message());
              } else {
                callbackFuture->setError(rpcFuture.error()->what());
              }
            });
          } catch (std::exception& e) {
            callbackFuture->setErrorIfNeeded(e.what());
          }
        });

    // Return the future which waits for all async processing to be done.
    return callbackFuture;
  } else {
    lock.unlock();
    auto graphTask = autogradContext->retrieveGraphTask();
    engine_.enqueue_blocked_task_on_cpu(torch::autograd::NodeTask(
        graphTask, sendFunction, torch::autograd::InputBuffer(0)));
    return std::make_shared<rpc::FutureMessage>(rpc::Message());
  }
}

// hcz: The entry point of call dist_autograd.backward(context_id, [loss]) in Python
void DistEngine::execute(
    int64_t contextId,
    const variable_list& roots,
    bool retainGraph) {
  LOG(ERROR) << "Running DistEngine::execute() with " << roots.size() << " roots.";
  // Retrieve the context for the given context_id. This will throw if the
  // context_id is invalid.
  auto autogradContext =
      DistAutogradContainer::getInstance().retrieveContext(contextId);

  // Perform initial pre-processing.
  edge_list rootEdges;
  variable_list grads;
  validateRootsAndRetrieveEdges(roots, rootEdges, grads);

  std::shared_ptr<Node> graphRoot =
      std::make_shared<GraphRoot>(rootEdges, grads);
  edge_list outputEdges;
  // Compute dependencies locally, starting from all roots and all 'send'
  // functions.
  {
    std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
    // Context should not have been initialized already.
    TORCH_INTERNAL_ASSERT(
        initializedContextIds_.find(autogradContext->contextId()) ==
        initializedContextIds_.end());

    LOG(ERROR) << "hcz: computing dependencies and set up the graphTask in the context.";
    computeDependencies(
        autogradContext, rootEdges, grads, graphRoot, outputEdges, retainGraph);

    // Mark the autograd context id as initialized.
    initializedContextIds_.insert(autogradContext->contextId());
  }

  BackwardPassCleanupGuard guard(autogradContext);

  // This needs to be blocking and as a result we wait for the future to
  // complete.
  runEngineAndAccumulateGradients(autogradContext, graphRoot, outputEdges)
      ->wait();

  // Wait for all of the outstanding rpcs to complete.
  autogradContext->clearAndWaitForOutstandingRpcsAsync()->wait();
}

void DistEngine::cleanupBackwardPass(const ContextPtr& autogradContext) {
  // Validate only the GraphTask is holding a reference to the Future
  // which holds gradients for the backward pass. This ensures that
  // after 'resetGraphTask' is called below, there are no remaining
  // references left to the gradients for the backward pass.
  //
  // This ensures our 'use_count' checks in
  // AccumulateGrad::accumulateGrad are correct and we're
  // not leaking any references to the gradients anywhere else.
  const auto& futureGrads =
      autogradContext->retrieveGraphTask()->future_result_;
  TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1);

  // Reset the graph task once we're done with all processing.
  autogradContext->resetGraphTask();

  // Clear any outstanding rpcs.
  autogradContext->clearOutstandingRpcs();

  // Clear the context id once we're done with the autograd engine
  // processing.
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  initializedContextIds_.erase(autogradContext->contextId());
}

size_t DistEngine::numBackwardPasses() const {
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  return initializedContextIds_.size();
}

std::unordered_map<std::string, std::string> DistEngine::getDebugInfo() const {
  std::unordered_map<std::string, std::string> debugInfo;
  auto& DistAutogradContainer = DistAutogradContainer::getInstance();
  debugInfo[kNumBackwardPasses] = std::to_string(numBackwardPasses());
  // fill in all cpu queue size information for each graph task of the
  // context_id in initializedContextIds_
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  for (auto context_id : initializedContextIds_) {
    std::shared_ptr<torch::autograd::GraphTask> graph_task =
        DistAutogradContainer.retrieveContext(context_id)->retrieveGraphTask();
    std::string kGraphTaskCPUQueueSize =
        "context_id: " + std::to_string(context_id) +
        " graph_task_cpu_queue_size";
    debugInfo[kGraphTaskCPUQueueSize] =
        std::to_string(engine_.ready_queue_size(graph_task, at::kCPU));
  }
  debugInfo[kNumAutogradContexts] = std::to_string(
      DistAutogradContainer::getInstance().numAutogradContexts());
  return debugInfo;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
