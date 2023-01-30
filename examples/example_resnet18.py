import torch

try:
    import zipslicer
except Exception:
    import sys

    sys.path.append("./zipslicer")
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

# Cleanup after our experiment
import os
os.unlink("my_network_checkpoint_v123.pth")
