# zipslicer 📁✂️

A library for incremental loading of large PyTorch checkpoints

[Read a blogpost introduction by yours truly](https://kir-gadjello.github.io/zipslicer)

# Synopsis
```
import torch
import zipslicer

# Could be a private custom recurrent sentient transformer
# instead of a garden variety resnet
my_complicated_network = torch.hub.load(
    "pytorch/vision:v0.10.0", "resnet18", pretrained=True
)
s_dict = my_complicated_network.state_dict()
torch.save(s_dict, "my_network_checkpoint_v123.pth")
del my_complicated_network

# Later, on a smaller unrelated machine you load a "LazyStateDict"
# Which is just like a regular state dict, but it loads tensors only when it has to
lazy_s_dict = zipslicer.load("my_network_checkpoint_v123.pth")
layer3_tensors = {}
for k in lazy_s_dict.keys():
    if k.startswith("layer3"):
        layer3_tensors[k] = lazy_s_dict[k]
# Now you have layer3's tensors and you can analyze them without breaking your RAM.
# Or you can load the layers' classes in sequence and compute the whole network's output.
# But we will just print the tensors:
print(layer3_tensors)
```

Run this example and unit-tests:

`python examples/example_resnet18.py`

`pytest-3 -o log_cli=true --capture=tee-sys -p no:asyncio tests/test_synthetic.py`

Test your checkpoint for compatibility:

`python tests/test_checkpoint_readonly.py your_magnificent_checkpoint.pth`

If it's all green, it will work

# Pre-Requisites

* Modern enough install of PyTorch - use included safe test to check for compatibility of `zipslicer` with your PyTorch, your checkpoint. This is a pure Python library, so specific CPU architecture shouldn't matter.
* A checkpoint produced by saving your model's state dict via vanilla torch.save(...) - default settings should suffice, as Torch doesn't use ZIP compression.
* An application that can take advantage of incrementally-loaded checkpoint - i.e. if your app just loads all `state_dict.items()` right away it doesn't make much sense to use this library. Make sure your code reads `state_dict.keys()` (and `state_dict.get_meta(k)` if necessary) and uses these intelligently to work on a subset of `state_dict[k]` tensors at a time.

# Install

Generally, copying the `zipslicer/zipslicer` directory into your project's source tree is enough.

If you are a fan of official ceremony-driven install processes for executable modules of dubious provenance, soon there will be a possibility of installing this zipslicer via pip:

`pip install zipslicer`

# Notes
* Some rare tensor types (i.e: pytorch quantized tensors - not to be confused with integer tensors) are not yet supported, if it bothers you, share your experience in issues.
* Perhaps more importantly, <u>general-purpose pickles are not supported</u> - the design of this library doesn't allow you to save and load whole neural network class instances. Usually this isn't necessary, and [pytorch official documentation recommends you use `state_dict` for model serialization](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict). We support `state_dict`s.
* We say "Hi" to [HF `safetensors` project](https://github.com/huggingface/safetensors), but note that in comparison to theirs, our approach doesn't require checkpoint conversion. In fact, both approaches could be complementary, as you will need to load tensors from the pytorch checkpoint to convert it to `safetensors`