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

## Appendix

### Lightweight Model Design Trend

\\<!-- 마크다운 테이블 첨부 -->


<br>

## References

[1] Mao, J., Chen, X., Nixon, K. W., Krieger, C., & Chen, Y. (2017, March). “Modnn: Local Distributed Mobile Computing System For Deep Neural Network”. In Design, Automation & Test in Europe Conference & Exhibition (DATE), 2017 (pp. 1396-1401). IEEE.

[2] Kim, Y., Kim, J., Chae, D., Kim, D., & Kim, J. (2019, March). “μlayer: Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration And Processor-Friendly Quantization”. In Proceedings of the Fourteenth EuroSys Conference 2019 (pp. 1-15).

[3] Ko, Y., Yu, J. S., Bae, H. K., Park, Y., Lee, D., & Kim, S. W. (2021, December).” MASCOT: A Quantization Framework for Efficient Matrix Factorization in Recommender Systems”. In 2021 IEEE International Conference on Data Mining (ICDM) (pp. 290-299). IEEE.

[4] Zhou, L., Samavatian, M. H., Bacha, A., Majumdar, S., & Teodorescu, R. (2019, November). “Adaptive Parallel Execution Of Deep Neural Networks On Heterogeneous Edge Devices”. In Proceedings of the 4th ACM/IEEE Symposium on Edge Computing (pp. 195-208).

[5] Choi, K., Lee, S., Kang, B. W., & Park, Y. (2021, October). “Legion: Tailoring Grouped Neural Execution Considering Heterogeneity On Multiple Edge Devices”. In 2021 IEEE 39th International Conference on Computer Design (ICCD) (pp. 383-390). IEEE.

[6] Goel, A., Tung, C., Hu, X., Thiruvathukal, G. K., Davis, J. C., & Lu, Y. H. (2022, January). ”Efficient Computer Vision On Edge Devices With Pipeline-Parallel Hierarchical Neural Networks”. In 2022 27th Asia and South Pacific Design Automation Conference (ASP-DAC) (pp. 532-537). IEEE.

[7] Mellor, J., Turner, J., Storkey, A., & Crowley, E. J. (2021, July). “Neural Architecture Search Without Training”. In International Conference on Machine Learning (pp. 7588-7598). PMLR.

[8] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. “Mobilenetv2: Inverted Residuals And Linear Bottlenecks”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018, pp. 4510–4520.

[9] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V Le. “Mnasnet: Platform-Aware Neural Architecture Search For Mobile”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2019, pp. 2820–2828.

[10] Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, and Kurt Keutzer. “Fbnet: Hardware-Aware Efficient Convnet Design Via Differentiable Neural Architecture Search”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2019, pp. 10734–10742.

[11] Han Cai, Ligeng Zhu, and Song Han. “ProxylessNAS: Direct Neural Architecture Search On Target Task And Hardware”. In: International Conference on Learning Representations (ICLR). 2019.

[12] Dimitrios Stamoulis, Ruizhou Ding, Di Wang, Dimitrios Lymberopoulos, Bodhi Priyantha, Jie Liu, and Diana Marculescu. “Single-Path Nas: Designing Hardware-Efficient Convnets In Less Than 4 Hours”. Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019.

[13] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. “Searching For MobilenetV3”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019, pp. 1314–1324.

[14] Mingxing Tan and Quoc Le. “Efficientnet: Rethinking Model Scaling For Convolutional Neural Networks”. In: International Conference on Machine Learning. 2019, pp. 6105–6114.

[15] Mingxing Tan and Quoc V Le. “Mixconv: Mixed Depthwise Convolutional Kernels”.  In Proceedings of the British Machine Vision Conference. (2019).

[16] Dongyoon Han, Sangdoo Yun, Byeongho Heo, and YoungJoon Yoo. “Rethinking Channel Dimensions For Efficient Model Design”. 2021. Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition.
