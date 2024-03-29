## OpenDet-Class Wasserstein Anchor (OD-CWA)

<img src="./docs/vis_images.png" width="78%"/>

> **Wasserstein Distance-based Expansion of Low-Density Latent Regions for Unknown Class Detection**<br>

OpenDet_CWA: OpenDet_CWA is implemented based on [detectron2](https://github.com/facebookresearch/detectron2) and [Opendet2] (https://github.com/csuhan/opendet2).

 
[arXiv paper:] (https://arxiv.org/pdf/2401.05594.pdf).
### Illustration of OD-CWA framework

<img src="./docs/od_cwa.png" width="78%"/>


## Abstract
This paper addresses the significant challenge in open-set object detection (OSOD): the tendency of state-of-the-art detectors to erroneously classify unknown objects as known categories with high confidence. We present a novel approach that effectively identifies unknown objects by distinguishing between high and low-density regions in latent space. Our method builds upon the Open-Det (OD) framework, introducing two new elements to the loss function. These elements enhance the known embedding space's clustering and expand the unknown space's low-density regions. The first addition is the Class Wasserstein Anchor (CWA), a new function that refines the classification boundaries. The second is a spectral normalisation step, improving the robustness of the model. Together, these augmentations to the existing Contrastive Feature Learner (CFL) and Unknown Probability Learner (UPL) loss functions significantly improve OSOD performance. Our proposed OpenDet-CWA (OD-CWA) method demonstrates: a) a reduction in open-set errors by approximately 17%-22%, b) an enhancement in novelty detection capability by 1.5%-16%, and c) a decrease in the wilderness index by 2%-20% across various open-set scenarios. These results represent a substantial advancement in the field, showcasing the potential of our approach in managing the complexities of open-set object detection.


<p align="center" width="100%">
<img src="./docs/inference.png" width="80%" />
</p>

<p align="center" width="80%">
<strong>Figure:</strong> Qualitative comparisons between proposed OD (top) and OD-CWA (bottom). Both models are trained on VOC and the detection results are visualised using images from COCO. The purple colour represents unknown and white represents known. White annotations represent classes seen by the model and purple annotation correspond to unknown classes.
</p>




### Setup

The code is based on [detectron2 v0.5](https://github.com/facebookresearch/detectron2/tree/v0.5). 

* **Installation** 

Here is a from-scratch setup script.

```
conda create -n opendet_cwa python=3.8 -y
conda activate opendet_cwa
pip install torch==2.0.0 torchvision torchaudio torchtext
pip install 'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
pip install geomloss pillow==9.4 opencv-python
git clone https://github.com/proxymallick/OpenDet_CWA.git
cd OpenDet_CWA
```

* **Prepare datasets** 

Please follow [README.md] (https://github.com/csuhan/opendet2/blob/main/datasets/README.md) of [Opendet2] (https://github.com/csuhan/opendet2)
to prepare the dataset.
It involves simple steps of following the script which utilizes VOC20{07,12} and COCO dataset to create combination of OS datasets using:

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

**Faster R-CNN**

| Method       | backbone | mAP<sub>K&uarr;</sub>(VOC) | WI<sub>&darr;</sub> | AOSE<sub>&darr;</sub> | mAP<sub>K&uarr;</sub> | AP<sub>U&uarr;</sub> |   Download   | 
|--------------|:--------:|:--------------------------:|:-------------------:|:---------------------:|:---------------------:|:--------------------:|  :------------:|
| FR-CNN       |   R-50    | **80.10** | 18.39 | 15118 | **58.45** | -  | [config](configs/faster_rcnn_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/10uFOLLCK4N8te08-C-olRyDV-cJ-L6lU?usp=sharing) |
| CAC  |   R-50   |   79.70 | 19.99 |16033 |57.76 |-| [config](configs/faster_rcnn_R_50_FPN_3x_opendet.yaml) [model](https://drive.google.com/drive/folders/1yf2-oHS35ZDOkiALf4Ge1FL3MQZGtiC8?usp=drive_link) |
| PROSER       |   R-50   |            79.42           |        20.44        |         14266         |         56.72         |         16.99        |  [config](configs/faster_rcnn_R_50_FPN_3x_proser.yaml) [model](https://drive.google.com/drive/folders/1_L85gisyvDtBXPe2UbI49vrd5FoBIOI_?usp=sharing) |
| ORE          |   R-50   |            79.80           |        18.18        |         12811         |         58.25         |         2.60         | [config]() [model]() |
| DS           |   R-50   |            79.70           |        16.76        |         13062         |         58.46         |         8.75         |  [config](configs/faster_rcnn_R_50_FPN_3x_ds.yaml) [model](https://drive.google.com/drive/folders/1OWDjL29E2H-_lSApXqM2r8PS7ZvUNtiv?usp=sharing) |
| OD         |  R-50   |            80.02           |        12.50        |         10758         |         58.64         |         14.38        | [config](configs/faster_rcnn_R_50_FPN_3x_opendet.yaml) [model](https://drive.google.com/drive/folders/1fzD0iJ6lJrPL4ffByeO9M-udckbYqIxY?usp=sharing) |
| OD-SN |   R-50   |  79.66|  12.96 |  9432|   57.86 |  14.78|   [config](configs/faster_rcnn_R_50_FPN_3x_opendet.yaml) [model](https://drive.google.com/drive/folders/1yfMulONdB8P5ijlJdpTFf-wrsFoKNzeN?usp=drive_link) |
| OD-CWA |   R-50   |   79.20|  **11.70** | **8748** |  57.58 |  **15.36** |  [config](configs/faster_rcnn_R_50_FPN_3x_opendet.yaml) [model](https://drive.google.com/drive/folders/1t48fAw0AscCo_5OFZS9aCT_wV0v1CknM?usp=drive_link) |


**Swin-T**
| Method       | backbone | mAP<sub>K&uarr;</sub>(VOC) | WI<sub>&darr;</sub> | AOSE<sub>&darr;</sub> | mAP<sub>K&uarr;</sub> | AP<sub>U&uarr;</sub> |   Download   | 
|--------------|:--------:|:--------------------------:|:-------------------:|:---------------------:|:---------------------:|:--------------------:|  :------------:|
| OpenDet(OD)  |  Swin-T  |           83.29 | 12.51 | 9875 | 63.17 | 15.77 |  [config](configs/faster_rcnn_Swin_T_FPN_3x_opendet.yaml) [model](https://drive.google.com/drive/folders/1j5SkEzeqr0ZnGVVZ4mzXSOvookHfvVvm?usp=sharing) |
| OD-SN        |  Swin-T  |                      82.49 | 14.39           | **7306**              | 61.59 | 16.45 | [config](configs/faster_rcnn_R_50_FPN_3x_opendet.yaml) [model](https://drive.google.com/drive/folders/1pcLE3Dp66gmboBnu6xask0RWI4a3HX7e?usp=drive_link) |
| OD-CWA   |  Swin-T  |        **83.34** | **10.35** | 8946 | **63.58** | **18.22** | [config](configs/faster_rcnn_R_50_FPN_3x_opendet.yaml) [model](https://drive.google.com/drive/folders/1A2VxT5BI3FteS5Y0S7pjkmJY_imjmT0c?usp=drive_link) |



**Note**:
* The above codes and repo has been modified from [OpenDet2](https://github.com/csuhan/opendet2)
* There were issues installing and running Opendet2 from the instructions and this repo provides modified codes
* The above results are taken from the paper and not the reimplemented version mentioned in (https://github.com/csuhan/opendet2). 
* The download column contains new models for CAC, OD-SN, OD-CWA for ResNet-50 and Swin-T backbone. However, the rest of the models for comparison are taken from [OpenDet2](https://github.com/csuhan/opendet2)


### Train and Test
* **Evaluation and Visualisation**

The embedding space visualisation can be conducted by running jupyter notebook [inference](docs/inference.ipynb) file. It loads the embeddings which are stored during the evaluation phase on the holdout test set. The notebook also contains the codes to generate the inter-cluster and intra-cluster distance that following 6 different metrics.

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

* **Embedding projections of Knowns (ID: 1-20) and Unknowns (ID: 80) of   OD (right) vs. OD-CWA (left)**
!<img src="./docs/emb.png" width="48%"/>|<img src="./docs/emb_od.png" width="48%"/>


If you use this repository, please cite:

```text
@misc{mallick2024wasserstein,
      title={Wasserstein Distance-based Expansion of Low-Density Latent Regions for Unknown Class Detection}, 
      author={Prakash Mallick and Feras Dayoub and Jamie Sherrah},
      year={2024},
      eprint={2401.05594},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**Contact**

If you have any questions or comments, please contact [Prakash Mallick](mailto:prakash.mallick@adelaide.edu.au).