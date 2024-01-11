## OpenDet

<img src="./docs/vis_images.png" width="78%"/>

> **Wasserstein Distance-based Expansion of Low-Density Latent Regions for Unknown Class Detection (CVPR2022)**<br>

OpenDet_CWA: OpenDet_CWA is implemented based on [detectron2](https://github.com/facebookresearch/detectron2) and [Opendet2] (https://github.com/csuhan/opendet2).

### Setup

The code is based on [detectron2 v0.5](https://github.com/facebookresearch/detectron2/tree/v0.5). 

* **Installation** 

Here is a from-scratch setup script.

```
conda create -n opendet2 python=3.8 -y
conda activate opendet2
pip install torch==2.0.1+cu117 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
git clone https://github.com/proxymallick/OpenDet_CWA.git
cd OpenDet_CWA
pip install -v -e .
```

* **Prepare datasets** 

Please follow [datasets/README.md](datasets/README.md) for dataset preparation. Then we generate VOC-COCO datasets.

```
bash datasets/opendet2_utils/prepare_openset_voc_coco.sh
# using data splits provided by us.
cp datasets/voc_coco_ann datasets/voc_coco -rf
```

### Model OD-CWA

# Results Summary

We report results on VOC and VOC-COCO-20, providing pretrained models. For full details, refer to the corresponding log file.

## Performance Metrics


### Model Zoo

We report the results on VOC and VOC-COCO-20, and provide pretrained models. Please refer to the corresponding log file for full results.

* **Faster R-CNN**
  ***ResNet-50***
| Method       | backbone | mAP<sub>K&uarr;</sub>(VOC) | WI<sub>&darr;</sub> | AOSE<sub>&darr;</sub> | mAP<sub>K&uarr;</sub> | AP<sub>U&uarr;</sub> |      Download     |
|--------------|:--------:|:--------------------------:|:-------------------:|:---------------------:|:---------------------:|:--------------------:| :----------------:|
| FR-CNN       |   R-50    | **80.10** | 18.39 | 15118 | **58.45** | - | 22.74 | 23391 | 55.26 | - | 18.49 | 25472 | 55.83 | - | [config](configs/retinanet_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/15fHfyA2HuXp6LfdTMBuHG6ZwtLcgvD-p?usp=sharing) |
| PROSER       |   R-50   |            79.42           |        20.44        |         14266         |         56.72         |         16.99        |  [config](configs/retinanet_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/15fHfyA2HuXp6LfdTMBuHG6ZwtLcgvD-p?usp=sharing) |
| ORE          |   R-50   |            79.80           |        18.18        |         12811         |         58.25         |         2.60         | [config](configs/retinanet_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/15fHfyA2HuXp6LfdTMBuHG6ZwtLcgvD-p?usp=sharing) |
| DS           |   R-50   |            79.70           |        16.76        |         13062         |         58.46         |         8.75         | [config](configs/retinanet_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/15fHfyA2HuXp6LfdTMBuHG6ZwtLcgvD-p?usp=sharing) |



| Method       | backbone | mAP<sub>K&uarr;</sub>(VOC) | WI<sub>&darr;</sub> | AOSE<sub>&darr;</sub> | mAP<sub>K&uarr;</sub> | AP<sub>U&uarr;</sub> |      Download     |
|--------------|:--------:|:--------------------------:|:-------------------:|:---------------------:|:---------------------:|:--------------------:| :----------------:|
| OpenDet(OD)  |  Swin-T  |           **80.02** | 14.95 | 11286 | **58.75** | 14.93 |  [config](configs/retinanet_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/15fHfyA2HuXp6LfdTMBuHG6ZwtLcgvD-p?usp=sharing) |
| OD-CWA   |  Swin-T  |        79.20 | **11.70** | **8748** | 57.58 | **15.36** |[config](configs/retinanet_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/15fHfyA2HuXp6LfdTMBuHG6ZwtLcgvD-p?usp=sharing) |
| OD-SN        |  Swin-T  |                      79.66 | **12.96**           | **9432**              | 57.86 | **14.78** | **16.28** |  [config](configs/retinanet_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/15fHfyA2HuXp6LfdTMBuHG6ZwtLcgvD-p?usp=sharing) |

* Significant improvements in $WI, AOSE,$ and $AP_{U}$ are achieved at the expense of a slight decrease in $mAP_{K}$. Numbers in bold black color indicate the best performing on that metric, and bold orange indicates second best.*


**Note**:
* You can also download the pre-trained models at [github release](https://github.com/csuhan/opendet2/releases) or [BaiduYun](https://pan.baidu.com/s/1I4Pp40pM84aeYTNeGc0kPA) with extracting code ABCD.
* The above results are reimplemented. Therefore, they are slightly different from our paper.
* The official code of ORE is at [OWOD](https://github.com/JosephKJ/OWOD). So we do not plan to include ORE in our code. 


### Train and Test

* **Testing**

First, you need to download pretrained weights in the model zoo, e.g., [OpenDet](https://drive.google.com/drive/folders/10uFOLLCK4N8te08-C-olRyDV-cJ-L6lU?usp=sharing).

Then, run the following command:
```
python tools/train_net.py --num-gpus 8 --config-file configs/faster_rcnn_R_50_FPN_3x_opendet.yaml \
        --eval-only MODEL.WEIGHTS output/faster_rcnn_R_50_FPN_3x_opendet/model_final.pth
```



* **Training**

The training process is the same as detectron2.
```
python tools/train_net.py --num-gpus 8 --config-file configs/faster_rcnn_R_50_FPN_3x_opendet.yaml
```

To train with the Swin-T backbone, please download [swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) and convert it to detectron2's format using [tools/convert_swin_to_d2.py](tools/convert_swin_to_d2.py).
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python tools/convert_swin_to_d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224_d2.pth
```
