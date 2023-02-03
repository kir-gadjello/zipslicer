# Copyright (c) 2023- Kirill Gadjello.
# See LICENSE for details (basically it uses parts of PyTorch sourcecode and is licensed under the same conditions)

import os
from functools import reduce
import types
from collections import OrderedDict
import pickle
import zipfile
import struct

import torch

from . import custom_load
from . import weights_only_unpickler


def create_storage(buf):
    if hasattr(torch, "UntypedStorage"):
        return torch.UntypedStorage.from_buffer(buf, dtype=torch.uint8)
    elif hasattr(torch, "_UntypedStorage"):
        # fallback for older torch versions which don't have UntypedStorage
        return torch._UntypedStorage.from_buffer(buf, dtype=torch.uint8)
    else:
        # fallback for older torch versions which don't have _UntypedStorage
        return torch.ByteStorage.from_buffer(buf)


def get_typed_storage(backing_storage, dtype):
    # TODO: Upstream pytorch might change the semantics here eventually
    if hasattr(torch, "TypedStorage"):
        return torch.storage.TypedStorage(wrap_storage=backing_storage, dtype=dtype)
    else:
        legacy_storage_types = {
            torch.bfloat16: torch.BFloat16Storage,
            torch.half: torch.HalfStorage,
            torch.float: torch.FloatStorage,
            torch.double: torch.DoubleStorage,
            torch.int8: torch.CharStorage,
            torch.uint8: torch.ByteStorage,
            torch.short: torch.ShortStorage,
            torch.int: torch.IntStorage,
            torch.long: torch.LongStorage,
            torch.bool: torch.BoolStorage,
        }
        if dtype not in legacy_storage_types:
            raise Exception(
                "Failed to create Torch Storage object, your Torch version is probably too old"
            )
        return legacy_storage_types[dtype](wrap_storage=backing_storage)


# ZIP "local file header" structure, magic number, size, and indices
# (section V.A in the format document)
structFileHeader = "<4s2B4HL2L2H"
stringFileHeader = b"PK\003\004"
_FH_SIGNATURE = 0
_FH_EXTRA_FIELD_LENGTH = 11
_FH_FILENAME_LENGTH = 10
_FH_GENERAL_PURPOSE_FLAG_BITS = 3

sizeFileHeader = struct.calcsize(structFileHeader)


def skip_header(zef_file, zinfo):
    zef_file.seek(zinfo.header_offset)
    fheader = zef_file.read(sizeFileHeader)
    if len(fheader) != sizeFileHeader:
        raise Exception("Truncated file header")
    fheader = struct.unpack(structFileHeader, fheader)
    if fheader[_FH_SIGNATURE] != stringFileHeader:
        raise Exception("Bad magic number for file header")

    fname = zef_file.read(fheader[_FH_FILENAME_LENGTH])
    if fheader[_FH_EXTRA_FIELD_LENGTH]:
        zef_file.read(fheader[_FH_EXTRA_FIELD_LENGTH])

    if fheader[_FH_GENERAL_PURPOSE_FLAG_BITS] & 0x800:
        # UTF-8 filename
        fname_str = fname.decode("utf-8")
    else:
        fname_str = fname.decode("cp437")

    if fname_str != zinfo.orig_filename:
        raise Exception(
            "File name in directory %r and header %r differ."
            % (zinfo.orig_filename, fname)
        )

    return True


dtype_sizes = {
    torch.float64: 8,
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    # torch.quint8: 1, # TODO
    # torch.qint8: 1,
}

dtype_by_name = {
    "torch.float64": torch.float64,
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.int16": torch.int16,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.bool": torch.bool,
    # "torch.quint8": torch.quint8,
    # "torch.qint8": torch.qint8,
}


def load_tensor_partial(
    zipfile,
    fh,
    offset_index,
    dtype,
    numel,
    key,
    location,
    offset,
    use_uncompressed=True,
):
    name = f"data/{key}"
    znames = list(filter(lambda x: x.endswith(name), zipfile.namelist()))
    assert len(znames) == 1
    zname = znames[0]

    dsize = dtype_sizes[dtype]
    bbuffer = None

    if use_uncompressed:
        try:
            zero_offset = None

            # fast path
            if offset_index.get(zname) is not None:
                zero_offset = offset_index.get(zname)
                data_offset = zero_offset + offset
                fh.seek(data_offset)
                bbuffer = fh.read(dsize * numel)
            else:
                info = zipfile.getinfo(zname)
                is_uncompressed = (
                    info.compress_size == info.file_size
                ) and info.compress_type == 0

                if is_uncompressed:
                    fh.seek(info.header_offset)
                    success = skip_header(fh, info)

                    if success:
                        zero_offset = fh.tell()
                        offset_index[zname] = zero_offset
                        data_offset = zero_offset + offset
                        fh.seek(data_offset)
                        bbuffer = fh.read(dsize * numel)

                        assert len(bbuffer) == dsize * numel
        except Exception as e:
            print(f"[ZIPSLICER]: Exception during attempt at fast seek: {e}")

    # fallback uses python-native zipfile seek which becomes slow for large checkpoints
    if bbuffer is None:
        print("[ZIPSLICER]: fast torch storage seek failed, executing fallback")
        with zipfile.open(zname, "r") as zf:
            zf.seek(offset)
            bbuffer = zf.read(dsize * numel)

    storage = create_storage(bbuffer)
    return get_typed_storage(storage, dtype)


# This class is meant to behave in a functionally similar manner to
# the conventional PyTorch state_dict objects (instances of Python's OrderedDict).
# It should be noted that as the docs say in https://peps.python.org/pep-0372/
# "subclassing dict is a non-trivial task", and this implementation doesn't try
# to be perfect. But it should work for pytorch checkpoint access, see tests.
# Note that creation of new keys is not yet supported.
# Caching isn't meant to be enabled just yet, but tcache member is used
# for storage of some k-v pairs some apps may assign to state_dict object.
# Currently, this class is meant to be created inside zipslicer.load method
class LazyStateDict(OrderedDict):
    """
    A Lazy state_dict produced by zipslicer https://github.com:kir-gadjello/zipslicer
    This object should be used to access PyTorch checkpoints in an incremental way.
    If caching is disabled (by default) this object doesn't force the tensors and
    Torch Storage objects it created to stay in RAM - you can delete them safely.
    Updating and deleting the values in-place (without saving) is supported.
    Creation of new keys isn't supported just yet.

    Special methods: there is a 'get_meta' method for accessing torch tensor shapes
    for the available keys without loading the whole tensor from disk.
    """

    def __init__(
        self,
        tensors=None,
        extras=None,  # TODO: we might have to handle extra data
        untie_weights=False,  # TODO
        map_location="cpu",
        zipfile=None,
        fh=None,
        debug=False,
        dtype=None,
        cache_tensors=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__lazy = True
        self.tensors = tensors
        self.zipfile = zipfile
        self.fh = fh
        self.offset_index = {}
        self.map_location = map_location
        self.untie_weights = untie_weights
        self.cache_tensors = cache_tensors
        self.debug = debug
        self.dtype = dtype
        self.tcache = {}

        for k in self.keys():
            self.validate_tensor_ref(k)

    def __del__(self):
        if self.zipfile is not None:
            if hasattr(self.zipfile, "close"):
                self.zipfile.close()
            del self.zipfile
        if self.fh is not None:
            if hasattr(self.fh, "close"):
                self.fh.close()
            del self.fh

    def __len__(self):
        return len(self.tensors.keys())

    def __setitem__(self, key, value):
        # Not supporting adding new keys for now
        if key not in self.tensors:
            raise KeyError(key)

        self.tcache[key] = value

    def __delitem__(self, key):
        if key not in self.tcache and key not in self.tensors:
            raise KeyError(key)

        if key in self.tcache:
            del self.tcache[key]
        if key in self.tensors:
            del self.tensors[key]

    def __getitem__(self, k):
        if k in self.tcache:
            return self.tcache[k]
        elif k in self.tensors:
            ret = self.reform_tensor(k)
            if self.cache_tensors:
                self.tcache[k] = ret
            return ret
        else:
            raise KeyError(k)

    def __contains__(self, k):
        return k in self.tcache.keys() or k in self.tensors.keys()

    def keys(self):
        # Fast path, cache is disabled and no keys were assigned
        if len(self.tcache) == 0:
            return self.tensors.keys()
        else:
            # A naive implementation, but should suffice
            return list(set(self.tcache.keys()) + set(self.tensors.keys()))

    def values(self):
        raise Exception(
            "LazyStateDict isn't meant for loading all values at once due to RAM constraints, reconsider your usage pattern"
        )

    def items(self):
        for k in self.keys():
            yield (k, self.__getitem__(k))

    def get_meta(self, k):
        if k not in self.tensors:
            raise KeyError(k)

        if k.endswith("._extra_state"):
            return None

        dtype = self.tensors[k]["args"][0]["dtype"]
        dtype = dtype_by_name[dtype]
        size = torch.Size(self.tensors[k]["args"][2])

        return types.SimpleNamespace(
            shape=size,
            size=lambda: size,
            dtype=dtype,
        )

    def validate_tensor_ref(self, k):
        if k not in self.tensors:
            raise KeyError(k)

        # Allow free-form ._extra_state for now: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state
        if k.endswith("._extra_state"):
            assert isinstance(self.tensors[k], dict)
            return True

        ref = self.tensors[k]
        assert ref["type"] == "stub_obj"
        assert ref["fn"] == "_rebuild_tensor_v2"
        storage_args = ref["args"][0]

        dtype = None
        try:
            dtype = dtype_by_name[storage_args["dtype"]]
        except Exception as e:
            print("Couldn't load tensor:", e)
            return None

        assert isinstance(dtype, torch.dtype)

        return True

    def reform_tensor(self, k):
        if k not in self.tensors:
            raise KeyError(k)

        if k.endswith("._extra_state"):
            return self.tensors[k]

        ref = self.tensors[k]
        storage_args = ref["args"][0]
        rebuild_tensor_args = ref["args"][1:]
        dtype = eval(storage_args["dtype"])

        (
            storage_offset,
            size,
            stride,
            requires_grad,
            backward_hooks,
        ) = rebuild_tensor_args

        assert dtype in dtype_sizes

        # TODO stride correctness checks

        dsize = dtype_sizes[dtype]
        numel = (
            reduce(lambda x, y: x * y, size) if size is not None and len(size) else 1
        )

        storage = load_tensor_partial(
            self.zipfile,
            self.fh,
            self.offset_index,
            dtype=dtype,
            numel=numel,
            key=storage_args["key"],
            location=self.map_location,
            offset=storage_offset * dsize,
        )

        ret = torch._utils._rebuild_tensor_v2(
            storage, 0, size, stride, requires_grad, backward_hooks
        )

        if self.dtype is not None and ret.dtype != dtype:
            ret = ret.to(dtype)

        return ret


def load(
    ckpt,
    map_location="cpu",
    dtype=None,
    cache_tensors=False,
    debug=False,
):
    """
    Should behave similarly to torch.load, but operates incrementally.
    Loads accessed tensors on the fly.
    ckpt should be a valid file path for now
    cache_tensors should be False
    """
    assert map_location == "cpu"
    assert os.path.isfile(ckpt)

    tensors_meta = None
    with custom_load._open_zipfile_reader(open(ckpt, "rb")) as zf:
        try:
            tensors_meta = custom_load._custom_load(
                zf, torch.device(map_location), weights_only_unpickler
            )
        except Exception as e:
            raise pickle.UnpicklingError(
                f"Error at zipslicer.load bootstrap stage, your torch pickle checkpoint is likely too complex for the lightweight loader to interpret. Make sure your network was saved as a state_dict, instead of general-purpose network pickle. Exception was: {e}"
            )

    zipfile_h = zipfile.ZipFile(ckpt, "r", allowZip64=True)

    return LazyStateDict(
        tensors=tensors_meta,
        fh=open(ckpt, "rb"),
        zipfile=zipfile_h,
        map_location=map_location,
        debug=debug,
        dtype=dtype,
        cache_tensors=cache_tensors,
    )
