# Synchronization-aware NAS for an Efficient Collaborative Inference on Mobile Platforms

An anonymized private repo for our work under review.

## Evaluation of Pre-trained version of our model 

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

## MCTS (Monte Carlo Tree Search Sampling)

TBD


## Appendix

### A. Lightweight Model Design Trend

\\<!-- 마크다운 테이블 첨부 -->


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
