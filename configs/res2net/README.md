# Res2Net for object detection and instance segmentation

## Introduction

We propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer.

|    Backbone     |Params. | GFLOPs  | top-1 err. | top-5 err. |
| :-------------: |:----:  | :-----: | :--------: | :--------: |
| ResNet-101      |44.6 M  | 7.8     |  22.63     |  6.44      |
| ResNeXt-101-64x4d |83.5M | 15.5    |  20.40     |  -         |
| HRNetV2p-W48    | 77.5M  | 16.1    |  20.70     |  5.50      |
| Res2Net-101     | 45.2M  | 8.3     |  18.77     |  4.64      |

Compared with other backbone networks, Res2Net requires fewer parameters and FLOPs.

**Note:**
- GFLOPs for classification are calculated with image size (224x224).

```
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2020},
  doi={10.1109/TPAMI.2019.2938758},
}
```

## Detection Results


### Faster R-CNN

|    Backbone     | Params. | GFLOPs | box AP |
| :-------------: | :----:  | :----: | :----: |
| R-101-FPN       | 60.52M  | 283.14 |  39.4  |
| X-101-64x4d-FPN | 99.25M  | 440.36 |  41.3  |
| HRNetV2p-W48    | 83.36M  | 459.66 |  41.5  |
| Res2Net-101     | 61.18M  | 293.68 |  42.3  |

### Mask R-CNN
|    Backbone     | Params. | GFLOPs | box AP | mask AP |
| :-------------: | :----:  | :----: | :----: | :----:  |
| R-101-FPN       | 63.17M  | 351.65 |  40.3  |  36.5   |
| X-101-64x4d-FPN | 101.9M  | 508.87 |  42.0  |  37.7   |
| HRNetV2p-W48    | 86.01M  | 528.17 |  42.9  |  38.3   |
| Res2Net-101     | 63.83M  | 362.18 |  43.3  |  38.6   |


### Cascade R-CNN

|    Backbone     | Params. | GFLOPs | box AP |
| :-------------: | :----:  | :----: | :----: |
| R-101-FPN       | 88.16M  | 310.78 |  42.5  |
| X-101-64x4d-FPN | 126.89M | 468.00 |  44.7  |
| HRNetV2p-W48    | 111.00M | 487.30 |  44.6  |
| Res2Net-101     | 88.82M  | 321.32 |  45.5  |


### Cascade Mask R-CNN

|    Backbone     | Params.  | GFLOPs | box AP | mask AP |
| :-------------: | :----:   | :----: | :----: | :----:  |
| R-101-FPN       | 96.09M   | 516.30 |  43.3  |  37.6   |
| X-101-64x4d-FPN | 134.82M  | 673.52 |  45.7  |  39.4   |
| HRNetV2p-W48    | 118.93M  | 692.82 |  46.0  |  39.5   |
| Res2Net-101     | 96.75M   | 526.84 |  46.1  |  39.4   |

### Hybrid Task Cascade (HTC)

|    Backbone     | Params.  | GFLOPs | box AP | mask AP |
| :-------------: | :-----:  | :----: | :----: | :----:  |
| R-101-FPN       | 99.03M   | 563.76 |  44.9  |  39.4   |
| X-101-64x4d-FPN | 137.75M  | 720.98 |  46.9  |  40.8   |
| HRNetV2p-W48    | 121.87M  | 740.28 |  47.0  |  41.0   |
| Res2Net-101     | 99.69M   | 574.30 |  47.5  |  41.3   |

**Note:**

- GFLOPs are calculated with image size (1280, 800).
- All detection methods in this page use pytorch style. Lr schd is 2x for Faster R-CNN and Mask R-CNN, and 20e for others. 
- Res2Net ImageNet pretrained models are in [Res2Net-PretrainedModels](https://github.com/Res2Net/Res2Net-PretrainedModels).
- More applications of Res2Net are in [Res2Net-Github](https://github.com/Res2Net/).
