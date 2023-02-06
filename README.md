# *ZIPSLICER* 📁✂️
[![Lint and Test Python package](https://github.com/kir-gadjello/zipslicer/actions/workflows/python-test.yml/badge.svg)](https://github.com/kir-gadjello/zipslicer/actions/workflows/python-test.yml)
[![Published to PyPI](https://github.com/kir-gadjello/zipslicer/actions/workflows/pypi-deploy.yml/badge.svg)](https://github.com/kir-gadjello/zipslicer/actions/workflows/pypi-deploy.yml)

A library for incremental loading of large PyTorch checkpoints<br>
[Read a blogpost introduction by yours truly](https://kir-gadjello.github.io/zipslicer)

## Synopsis
```python
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
# Or you can instantiate the layers' classes in sequence and compute the whole
# network's output for a given input by threading the activations through them.
# But we will just print the tensors instead:
print(layer3_tensors)
```

Run this example and unit-tests:

`python examples/example_resnet18.py`

`pytest -o log_cli=true --capture=tee-sys -p no:asyncio`

Test your checkpoint for compatibility:

`python tests/test_checkpoint_readonly.py your_magnificent_checkpoint.pth`

If it's all green, it will work.

## Prerequisites
* Supported python and torch versions: `python-3.10 + torch-(1.11,1.12,stable)` `python-3.11 + torch:stable`
* Generally, `zipslicer` should work with modern enough install of PyTorch - use [included safe test](https://github.com/kir-gadjello/zipslicer/blob/main/tests/test_checkpoint_readonly.py) to check for compatibility of `zipslicer` with your PyTorch and your checkpoint. This is a pure Python library, so specific CPU architecture shouldn't matter.
* A checkpoint produced by saving your model's `state_dict` via vanilla torch.save(...) - default settings should suffice, as Torch doesn't use ZIP compression.
* An application that can take advantage of incrementally-loaded checkpoint - i.e. if your app just loads all `state_dict.items()` in a loop right away it doesn't make much sense to use this library. Make sure your code reads `state_dict.keys()` (and `state_dict.get_meta(k)` if necessary) and uses these intelligently to work on a subset of `state_dict[k]` tensors at a time. For general inspiration you might read [this (HF)](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/model#transformers.modeling_utils.load_sharded_checkpoint) and [this (arxiv)](https://arxiv.org/abs/2104.07857). With some additional engineering it should be possible to run Large Language Models like [BLOOM-176B](https://huggingface.co/bigscience/bloom) or [FLAN-T5-XXL](https://huggingface.co/google/flan-t5-xxl) on a single mid-range GPU at home - if you are willing to wait for a night's worth of time. In the large batch regime this might even make some practical sense, for example to process a set of documents into embeddings.

## Install

Generally, copying the `zipslicer/zipslicer` directory into your project's source tree is enough.

If you are a fan of official ceremony-driven install processes for executable modules of dubious provenance, soon there will be a possibility of installing this boutique software module via pip: `pip install zipslicer`

## Notes
* This library is only for reading pytorch tensors from checkpoints. We leave writing for future work.
* Writing to loaded `state_dict` is frowned upon, but it *will* work - though you should avoid doing this while iterating over keys for now and expecting the keys to reflect this update.
* Perhaps more importantly, **general-purpose pickles are not supported** - the design of this library doesn't allow you to load whole neural network class instances. Usually this isn't necessary, and [pytorch official documentation recommends you to use `state_dict` for model serialization](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict). We support `state_dict`'s.
* Some rare tensor types (i.e: pytorch quantized tensors - not to be confused with integer tensors which work fine) are not yet supported. If this bothers you, share your experience in issues.
* We say "Hi" to [HF `safetensors` project](https://github.com/huggingface/safetensors), but note that in comparison to theirs, our approach doesn't require checkpoint conversion which takes significant time and storage. In fact, both approaches could be complementary, as you will have to load tensors from the pytorch checkpoint somehow to convert it to `safetensors` - and the default loading mechanism is constrained by available RAM.

## Prospective features we are considering
If you are interested in some of these features, consider creating an issue:
* Effective loading of tensor slices - to implement tensor parallelism in sharded deployments
* Accessing the source checkpoint over a network
* Writing to a checkpoint in-place
* Incremental conversion to other checkpoint formats
