## SyncNAS: Synchronization-aware NAS for an Efficient Collaborative Inference on Mobile Platforms

An anonymized private repo for our work under review.

### Requirements
- Install python libraries:
  ```
  pip3 install -r requirements.txt
  ```
- Set the environment variable PYTHONPATH
  ```
  export PYTHONPATH=/path/to/SyncNAS
  ```
- ImageNet2012 datasets (For validation)

### How to use
- To simply load PyTorch model :
```
#~/SyncNAS/

from torch_modules import TorchBranchedModel
from utils import load_params

#base_model: mobilenet_v2
model = TorchBranchedModel('model_configs/syncnas_mobilenet_v2_100.json')
model.load_state_dict(load_params('syncnas_mobilenet_v2_100.pth'))
```

- To evaluate on ImageNet :
```
$ python3 eval.py --base_model mobilenet_v2 --path 'your/path/to/imagenet'  

    # --base_model: base model that is being adapted -> available: ['mobilenet_v2', 'mnasnet_b1', 'fbnet_c']
    # -j: number of workers (default: 4)
    # -b: batch_size (default: 128)
```
