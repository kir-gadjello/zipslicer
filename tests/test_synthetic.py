# Copyright (c) 2023- Kirill Gadjello.
# See LICENSE for details (basically it uses part of PyTorch sourcecode and is licensed under the same conditions)

import os
import sys
import torch
import random
from test_checkpoint_readonly import __test_incremental_load

sys.path.append("./zipslicer")

import zipslicer

seed = int(os.environ.get("ZIPSLICER_TEST_SEED", "1337"))


def test_basic():
    FNAME = "test_basic.pth"
    torch.manual_seed(seed)

    sdict = dict(
        a=torch.randn(10, 20, 3, dtype=torch.float32),
        longer_name=torch.randn(10, 20, 3, dtype=torch.bfloat16),
    )

    torch.save(sdict, FNAME)
    __test_incremental_load(ckpt=FNAME)
    os.unlink(FNAME)


def test_various_dtypes():
    FNAME = "test_various_dtypes.pth"
    torch.manual_seed(seed)
    random.seed(seed)

    sdict = dict()
    for dtype in zipslicer.dtype_sizes.keys():
        key = ".".join(str(random.randint(0, 2**16)) for _ in range(6))
        # TODO: quantized tensor support
        if "q" not in str(dtype):
            t = (
                torch.randn(
                    random.randint(1, 16),
                    random.randint(1, 16),
                    random.randint(1, 16),
                    dtype=torch.float32,
                )
                * 200.0
            ).to(dtype)

            sdict[key] = t

    torch.save(sdict, FNAME)
    __test_incremental_load(ckpt=FNAME)
    os.unlink(FNAME)


def test_nn_sdict():
    FNAME = "test_nn_sdict.pth"
    torch.manual_seed(seed)

    network = torch.nn.ModuleList(
        [torch.nn.Linear(1000, 2000), torch.nn.Linear(2000, 2000)]
    )

    sdict = network.state_dict()

    torch.save(sdict, FNAME)
    __test_incremental_load(ckpt=FNAME)
    os.unlink(FNAME)


def test_nn_sdict_w_extra_state():
    FNAME = "test_nn_sdict_w_extra_state.pth"
    torch.manual_seed(seed)

    class CustomLinear(torch.nn.Linear):
        def get_extra_state(self):
            return dict(
                a=random.randint(1, 2**64 - 1), b="this is extra state", c=[1, 2, 3]
            )

    network = torch.nn.ModuleList([CustomLinear(1000, 2000), CustomLinear(2000, 2000)])

    sdict = network.state_dict()

    torch.save(sdict, FNAME)
    __test_incremental_load(ckpt=FNAME)
    os.unlink(FNAME)


def test_nn_pickle_raises():
    FNAME = "test_nn_pickle.pth"
    torch.manual_seed(seed)

    network = torch.nn.ModuleList(
        [torch.nn.Linear(1000, 2000), torch.nn.Linear(2000, 2000)]
    )

    torch.save(network, FNAME)

    try:
        zipslicer.load(FNAME)
    except Exception as e:
        assert (
            "Error at zipslicer.load bootstrap stage, your torch pickle checkpoint is likely too complex for the lightweight loader to interpret. Make sure your network was saved as a state_dict, instead of general-purpose network pickle"
            in str(e)
        )

    os.unlink(FNAME)
