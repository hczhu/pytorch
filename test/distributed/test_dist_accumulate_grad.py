#!/usr/bin/env python3

from typing import Callable, NamedTuple
import enum
import logging
import os
from functools import partial
from itertools import chain

from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.dist_utils import dist_init
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.distributed_c10d as dist_c10d
import torch.nn as nn
import torch.autograd as autograd

import logging
import os

WORLD_SIZE = 2
MASTER_RANK = 0
WORKER_RANK = 1

def init_logger():
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if "debug" in os.environ else logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
    return logger


gLogger = init_logger()


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_sync(
        rref.owner(), _call_method, args=args_tup, kwargs=kwargs
    )


def _remote_method_async(method, rref, *args, **kwargs):
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_async(
        rref.owner(), _call_method, args=args_tup, kwargs=kwargs
    )


def get_linear():
    d_in = 1
    d_out = 1
    l = nn.Linear(d_in, d_out, bias=False)
    w = torch.ones((d_out, d_in))
    w.requires_grad_()
    l.weight.data = w
    return l


class TestDistAccumulateGrad(MultiProcessTestCase):
    rpc_backend = rpc.backend_registry.BackendType.PROCESS_GROUP
    rpc_backend_options = None

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    def worker_name(self):
        return "worker1"

    def setUp(self):
        super(TestDistAccumulateGrad, self).setUp()

        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        self._spawn_processes()

    def tearDown(self):
        super(TestDistAccumulateGrad, self).tearDown()

    @dist_init
    def _worker_process(self):
        gLogger.info(f"Running the worker process...")

    @dist_init
    def _master_process(self, run_test: Callable):
        gLogger.info(f"Running the master process...")
        run_test()

    def _do_test(self, run_test: Callable):
        if self.rank == MASTER_RANK:
            self._master_process(run_test)
        elif self.rank == WORKER_RANK:
            self._worker_process()
        else:
            raise RuntimeError(f"Unknow process rank: {self.rank}")

    def _test_single_rpc(self):
        with dist_autograd.context() as context_id:
            a_rre = rpc.remote(self.worker_name(), get_linear)

            b = torch.tensor([2.0])
            b.requires_grad_()

            c = torch.tensor([3.0])
            c.requires_grad_()

            x = _remote_method(torch.nn.Linear.forward, a_rre, b)
            y = b * c

            z = x + y

            gLogger.info(f"Running dist backward.")
            dist_autograd.backward(context_id, [z])
            tensor_to_grad = dist_autograd.get_gradients(context_id)
            gLogger.info(f"Got tensor to grad map: {tensor_to_grad}")
            
    def test_single_rpc(self):
        self._do_test(self._test_single_rpc)
             

if __name__ == "__main__":
    run_tests()
