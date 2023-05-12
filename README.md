# Synchronization-aware NAS for an Efficient Collaborative Inference on Mobile Platforms

## Evaluation of Pre-trained Versions of Our models

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
model = TorchBranchedModel('model_configs/syncnas_mobilenet_v2_100.json')	#base_model: mobilenet_v2    
model.load_state_dict(load_params('syncnas_mobilenet_v2_100.pth'))
```

- To evaluate on ImageNet :
```
$ python3 eval.py --base_model mobilenet_v2 --path 'your/path/to/imagenet'  
    # --base_model: base model that is being adapted -> available: ['mobilenet_v2', 'mnasnet_b1', 'fbnet_c']
    # -j: number of workers (default: 4)
    # -b: batch_size (default: 128)
```

<br>

## Monte Carlo Tree Search
- Algorithm 1 in our paper corresponds to SyncNAS/src/local_worker.py
- Algorithm 2 in our paper corresponds to SyncNAS/src/mcts.py

<br>

## Appendix
### A. Lightweight Model Design Trend
- The recent lightweight CNNs consist of multiple inverted residual (MBConv) blocks, following the design convention inspired by MobileNetV2 [1] due to its computational efficiency.
  - *stages*: The conventional criteria that a number of MBConv blocks are grouped together
  - *Exp*: The expanded channel size of each MBConv layer
  - *Out*: The output channel size of each MBConv layer
  - *Stride*: The stride size of a depthwise convolution in each MBConv layer
  - *Kernel*: The kernel size of a depthwise convolution in each MBConv layer
- Note that $*$ mark indicates that squeeze-and-excitation is applied. 
- We omitted other details, such as nonlinearities, to highlight the general structure.

**\* Each bracket in stages specifies MBConv Block in the format of (*Exp*-*Out*,*Stride*,*Kernel*)**

|         Model         | Stem (Out) |    Stage 1   |                             Stage 2                            |                                       Stage 3                                      |                                            Stage 4                                           |                                            Stage 5                                           |                                    Stage 6                                    |                   Stage 7                   |
|:---------------------:|:----------:|:------------:|:--------------------------------------------------------------:|:----------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------:|:-------------------------------------------:|
|    MobileNetV2 [1]    |    (32)    |  (32-16,1,3) |                  (96-24,2,3) <br> (144-24,1,3)                 |                  (144-32,2,3) <br> (192-32,1,3) <br> (192-32,1,3)                  |              (192-64,2,3) <br> (384-64,1,3) <br> (384-64,1,3) <br> (384-64,1,3)              |                       (384-96,1,3) <br> (576-96,1,3) <br> (576-96,1,3)                       |              (576-160,2,3) <br> (960-160,1,3) <br> (960-160,1,3)              |  (960-320,1,3)                              |
|     MnasNet-A1 [2]    |    (32)    |  (32-16,1,3) |                  (96-24,2,3) <br> (144-24,1,3)                 |                   (72-40,2,5) <br> (120-40,1,5) <br> (120-40,1,5)                  |              (240-80,2,3) <br> (480-80,1,3) <br> (480-80,1,3) <br> (480-80,1,3)              |                               (480-112,1,3) <br> (672-112,1,3)                               |              (672-160,2,5) <br> (960-160,1,5) <br> (960-160,1,5)              |  (960-320,1,3)                              |
|     MnasNet-B1 [2]    |    (32)    |  (32-16,1,3) |          (48-24,2,3) <br> (72-24,1,3) <br> (72-24,1,3)         |                   (72-40,2,5) <br> (120-40,1,5) <br> (120-40,1,5)                  |                       (240-80,2,5) <br> (480-80,1,5) <br> (480-80,1,5)                       |                                (480-96,1,3) <br> (576-96,1,3)                                |   (576-192,2,5) <br> (1152-192,1,5) <br> (1152-192,1,5) <br> (1152-192,1,5)   |  (1152-320,1,3)                             |
|      FBNet-B [3]      |    (16)    |  (16-16,1,3) | (96-24,2,3) <br> (24-24,1,5) <br> (24,24,1,3) <br> (24,24,1,3) |          (144-32,2,5) <br> (96-32,1,5) <br> (192-32,1,3) <br> (192-32,1,5)         |                        (192-64,2,5) <br> (64-64,1,5) <br> (192-64,1,5)                       |            (384-112,1,5) <br> (112-112,1,3) <br> (112-112,1,5) <br> (336-112,1,5)            |    (672-184,2,5) <br> (184-184,1,5) <br> (1104-184,1,5) <br> (1104-184,1,5)   |  (1104-352,1,3)                             |
|      FBNet-C [3]      |    (16)    |  (16-16,1,3) |          (96-24,2,3) <br> (24-24,1,5) <br> (24,24,1,3)         |          (144-32,2,5) <br> (96-32,1,5) <br> (192-32,1,5) <br> (192-32,1,3)         |              (192-64,2,5) <br> (192-64,1,5) <br> (384-64,1,5) <br> (384-64,1,5)              |            (384-112,1,5) <br> (672-112,1,5) <br> (672-112,1,5) <br> (336-112,1,5)            |   (672-184,2,5) <br> (1104-184,1,5) <br> (1104-184,1,5) <br> (1104-184,1,5)   |  (1104-352,1,3)                             |
|    Proxyless-R [4]    |    (32)    |  (32-16,1,3) |                  (48-32,2,5) <br> (96-32,1,3)                  |          (96-40,2,7) <br> (120-40,1,3) <br> (120-40,1,5) <br> (120-40,1,5)         |              (240-80,2,7) <br> (240-80,1,5) <br> (240-80,1,5) <br> (240-80,1,5)              |              (480-96,1,5) <br> (288-96,1,5) <br> (288-96,1,5) <br> (288-96,1,5)              |    (576-192,2,7) <br> (1152-192,1,7) <br> (576-192,1,7) <br> (576-192,1,7)    |  (1152-320,1,7)                             |
|  Single-Path NAS [5]  |    (32)    |  (32-16,1,3) |          (48-24,2,3) <br> (72-24,1,3) <br> (72-24,1,3)         |         (144-40,2,5) <br> (120-40,1,3) <br> (120-40,1,3) <br> (120-40,1,3)         |              (240-80,2,5) <br> (240-80,1,3) <br> (240-80,1,3) <br> (240-80,1,3)              |              (480-96,1,5) <br> (288-96,1,5) <br> (288-96,1,5) <br> (288-96,1,5)              |   (576-192,2,5) <br> (1152-192,1,5) <br> (1152-192,1,5) <br> (1152-192,1,5)   |  (1152-320,1,3)                             |
| MobileNetV3-Large [6] |    (32)    |  (32-16,1,3) |                  (64-24,2,3) <br> (72-24,1,3)                  |                 (72-40,2,5)* <br> (120-40,1,5)* <br> (120-40,1,5)*                 |              (240-80,2,3) <br> (200-80,1,3) <br> (184-80,1,3) <br> (184-80,1,3)              |                              (480-112,1,3)* <br> (672-112,1,3)*                              |             (672-160,2,5)* <br> (960-160,1,5)* <br> (960-160,1,5)*            |           Conv2D <br> (X-960,1,1)           |
|  EfficientNet-B0 [7]  |    (32)    | (32-16,1,3)* |                  (96-24,2,3) <br> (144-24,1,3)                 |                           (144-40,2,5) <br> (240-40,1,5)                           |                       (240-80,2,3) <br> (480-80,1,3)* <br> (480-80,1,3)                      |                    (480-112,1,5)* <br> (672-112,1,5)* <br> (672-112,1,5)*                    | (672-192,2,5)* <br> (1152-192,1,5)* <br> (1152-192,1,5)* <br> (1152-192,1,5)* |   (1152-320,1,3)*                           |
|      MixNet-M [8]     |    (24)    |  (24-24,1,3) |                (144-32,2,3/5/7) <br> (96-32,1,3)               | (192-40,2,3/5/7/9)* <br> (240-40,1,3/5)* <br> (240-40,1,3/5)* <br> (240-40,1,3/5)* | (240-80,2,3/5/7)* <br> (240-80,1,3/5/7/9)* <br> (240-80,1,3/5/7/9)* <br> (240-80,1,3/5/7/9)* | (480-120,1,3)* <br> (360-120,1,3/5/7/9)* <br> (360-120,1,3/5/7/9)* <br> (360-120,1,3/5/7/9)* |   (720-200,2,3/5/7/9)* <br> (1200-200,1,3/5/7/9)* <br> (1200-200,1,3/5/7/9)*  | (1200-200,1,3/5/7/9)*                       |
|       ReXNet [9]      |    (32)    |  (32-16,1,3) |                  (96-27,2,3) <br> (162-38,1,3)                 |                          (228-50,2,3)* <br> (300-61,1,3)*                          |                      (366-72,2,3)* <br> (432-84,1,3)* <br> (504-95,1,3)*                     |                    (570-106,1,3)* <br> (636-117,1,3)* <br> (702-128,1,3)*                    |   (768-140,2,3)* <br> (840-151,1,3)* <br> (906-162,1,3)* <br> (972-174,1,3)*  |  (1044-185,1,3)                             |

<br>

### B. Visualization of Searched Models

We visualize the network architectures searched by SyncNAS below. Each block is parameterized by in_c, exp_c, out_c, k, and s, depending on the type of block.

![model_arch](https://github.com/beomwookang/SyncNAS/blob/main/pretrained/Searched_Models_Arch.png)


## Baseline Model Traning Information
- Optimizer: Stochastic Gradient Descent
- Learning Rate Scheduler: Cosine Annealing with Warm Restarts
  - Warm-up: 10 epochs
- Weight Decay: 1e-5
- Initial Learning Rate
  - MobileNetV2: 0.256
  - FBNet-B: 0.512
  - MnasNet: 0.512

<br>

## References
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. “Mobilenetv2: Inverted Residuals And Linear Bottlenecks”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018, pp. 4510–4520.

[2] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V Le. “Mnasnet: Platform-Aware Neural Architecture Search For Mobile”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2019, pp. 2820–2828.

[3] Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, and Kurt Keutzer. “Fbnet: Hardware-Aware Efficient Convnet Design Via Differentiable Neural Architecture Search”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2019, pp. 10734–10742.

[4] Han Cai, Ligeng Zhu, and Song Han. “ProxylessNAS: Direct Neural Architecture Search On Target Task And Hardware”. In: International Conference on Learning Representations (ICLR). 2019.

[5] Dimitrios Stamoulis, Ruizhou Ding, Di Wang, Dimitrios Lymberopoulos, Bodhi Priyantha, Jie Liu, and Diana Marculescu. “Single-Path Nas: Designing Hardware-Efficient Convnets In Less Than 4 Hours”. Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019.

[6] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. “Searching For MobilenetV3”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019, pp. 1314–1324.

[7] Mingxing Tan and Quoc Le. “Efficientnet: Rethinking Model Scaling For Convolutional Neural Networks”. In: International Conference on Machine Learning. 2019, pp. 6105–6114.

[8] Mingxing Tan and Quoc V Le. “Mixconv: Mixed Depthwise Convolutional Kernels”.  In Proceedings of the British Machine Vision Conference. (2019).

[9] Dongyoon Han, Sangdoo Yun, Byeongho Heo, and YoungJoon Yoo. “Rethinking Channel Dimensions For Efficient Model Design”. 2021. Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition.
