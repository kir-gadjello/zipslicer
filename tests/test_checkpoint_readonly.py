# Copyright (c) 2023- Kirill Gadjello.
# See LICENSE for details (basically it uses part of PyTorch sourcecode and is licensed under the same conditions)

# Run it like this (from zipslicer repo root directory):
# python ./tests/test_checkpoint_readonly.py 'path_to_your_checkpoint.pth'

import os
import sys
import time
import torch
import random

sys.path.append("./zipslicer")

import zipslicer

cgreen = "\033[92m"
cyellow = "\033[93m"
creset = "\033[0m"
ok_green = f"{cgreen}[OK]{creset}"


def __test_incremental_load(ckpt=None, seed=1337):
    random.seed(int(os.environ.get("ZIPSLICER_TEST_SEED", seed)))

    print_note = False
    if ckpt is None:
        if len(sys.argv) <= 1:
            print(
                "Usage:\n\tpython ./tests/test_checkpoint_readonly.py 'path_to_your_checkpoint.pth'"
            )
            sys.exit(-1)
        ckpt = sys.argv[1]
        print_note = True

    assert os.path.isfile(ckpt)
    if print_note:
        print(f'Using "{cyellow}{ckpt}{creset}" in {cgreen}readonly{creset} mode')
        print("=" * (os.get_terminal_size().columns))
        print(
            "Note: this test loads two copies of the checkpoint, one using standard torch.load and the other using zipslicer. You need enough CPU RAM to fit both, or you risk unresponsive behavior and massive swapping from your machine."
        )
        print("=" * (os.get_terminal_size().columns))

    sdict = torch.load(ckpt, map_location="cpu")
    skeys = sdict.keys()
    lazy_sdict = zipslicer.load(
        ckpt, map_location="cpu", debug=os.environ.get("ZIPSLICER_DEBUG") == "1"
    )
    lazy_keys = lazy_sdict.keys()

    print("Checking basic key correspondence")
    for k in skeys:
        assert k in lazy_keys

    for k in lazy_keys:
        assert k in skeys
    print(f"{ok_green}: {len(skeys)} keys total")

    print("Checking tensor metadata correspondence")
    for k, v in sdict.items():
        meta = lazy_sdict.get_meta(k)
        if k.endswith("._extra_state") and not isinstance(v, torch.Tensor):
            assert meta is None
            continue

        assert meta.shape == v.shape
        assert meta.size() == v.size()
        assert meta.dtype == v.dtype
    print(f"{ok_green}: {len(skeys)} keys total")

    test_keys = list(skeys)

    if os.environ.get("ZIPSLICER_TEST_SUBSET"):
        ratio = float(os.environ.get("ZIPSLICER_TEST_SUBSET"))
        random.shuffle(test_keys)
        N = int(len(test_keys) * ratio)
        test_keys = test_keys[:N]
        print(f"Using randomized key subset of length {N} for testing")

    N = len(test_keys)
    for i, k in enumerate(test_keys):
        print(f"[{i+1}/{N}] Checking key: {k}", end=" ")
        t0 = time.time_ns()
        T = sdict[k]
        LT = lazy_sdict[k]

        if k.endswith("._extra_state") and not isinstance(T, torch.Tensor):
            assert T == LT
        else:
            assert T.dtype == LT.dtype
            assert T.shape == LT.shape
            assert torch.allclose(T, LT)

        dt = time.time_ns() - t0
        print(f"{ok_green} in {round(dt/1e6, 2)}ms")

    del sdict
    del lazy_sdict


if __name__ == "__main__":
    __test_incremental_load()
    print(f"{ok_green} All tests passed successfully")
