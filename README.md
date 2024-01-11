## OpenDet

<img src="./docs/vis_images.png" width="78%"/>

> **Wasserstein Distance-based Expansion of Low-Density Latent Regions for Unknown Class Detection (CVPR2022)**<br>

OpenDet2: OpenDet is implemented based on [detectron2](https://github.com/facebookresearch/detectron2).

### Setup

The code is based on [detectron2 v0.5](https://github.com/facebookresearch/detectron2/tree/v0.5). 

* **Installation** 

Here is a from-scratch setup script.

```
conda create -n opendet2 python=3.8 -y
conda activate opendet2

conda install pytorch=1.8.1 torchvision cudatoolkit=10.1 -c pytorch -y
pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
git clone https://github.com/csuhan/opendet2.git
cd opendet2
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

| Method                   | VOC   | VOC-COCO-20 | VOC-COCO-40 | VOC-COCO-60 |
|--------------------------|-------|-------------|-------------|-------------|
|                          | $mAP_{K\uparrow}$ | $WI_{\downarrow}$ | $AOSE_{\downarrow}$ | $mAP_{K\uparrow}$ | $AP_{U\uparrow}$ | $WI_{\downarrow}$ | $AOSE_{\downarrow}$ | $mAP_{K\uparrow}$ | $AP_{U\uparrow}$ | $WI_{\downarrow}$ | $AOSE_{\downarrow}$ | $mAP_{K\uparrow}$ | $AP_{U\uparrow}$ |
|--------------------------|-------|-------------|-------------|-------------|
| FR-CNN                  | **80.10** | 18.39 | 15118 | **58.45** | - | 22.74 | 23391 | 55.26 | - | 18.49 | 25472 | 55.83 | - |
| CAC                    | 79.70 | 19.99 | 16033 | 57.76 | - | 24.72 | 25274 | 55.04 | - | 20.21 | 27397 | 55.96 | - |
| PROSER                 | 79.68 | 19.16 | 13035 | 57.66 | 10.92 | 24.15 | 19831 | 54.66 | 7.62 | 19.64 | 21322 | 55.20 | 3.25 |
| ORE                    | 79.80 | 18.18 | 12811 | 58.25 | 2.60 | 22.40 | 19752 | 55.30 | 1.70 | 18.35 | 21415 | 55.47 | 0.53 |
| DS                     | **80.04** | 16.98 | 12868 | 58.35 | 5.13 | 20.86 | 19775 | 55.31 | 3.39 | 17.22 | 21921 | 55.77 | 1.25 |
| OD                     | **80.02** | 14.95 | 11286 | **58.75** | 14.93 | 18.23 | 16800 | **55.83** | **10.58** | 14.24 | 18250 | **56.37** | **4.36** |
| OD-SN                  | 79.66 | **12.96** | **9432** | 57.86 | **14.78** | **16.28** | **14118** | **55.36** | 10.54 | **12.76** | **15251** | **56.07** | 4.17 |
| OD-CWA                 | 79.20 | **11.70** | **8748** | 57.58 | **15.36** | **14.58** | **13037** | **55.26** | **10.98** | **11.55** | **14984** | **55.73** | **4.45** |

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
