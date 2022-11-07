# MoCo v2+SupContrast

The officially released code of [SupContrast](https://github.com/HobbitLong/SupContrast) is based on SimCLR, while this repository is based on [MoCo v2](https://github.com/facebookresearch/moco) for supervised contrastive learning.



## [2020-CVPR] MoCo: Momentum Contrast for Unsupervised Visual Representation Learning [[code]](https://github.com/facebookresearch/moco)

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

## [2020-NeurIPS] Supervised Contrastive Learning [[code]](https://github.com/HobbitLong/SupContrast)

<p align="center">
  <img src="https://github.com/HobbitLong/SupContrast/raw/master/figures/teaser.png" width="300">
</p>




### Unsupervised Training on Cifar-10

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

There are three frameworks: **main_moco_in**, **main_moco_out**, **main_moco_suploss**.

To do unsupervised pre-training of a ResNet-50 model on Cifar-10 in an 2-gpu machine, run:
```
CUDA_VISIBLE_DEVICES=0,1  python main_moco_out.py \
  -a resnet50 \
  --lr 0.12 \
  --batch-size 256 --moco-k 4096 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10013' --multiprocessing-distributed --world-size 1 --rank 0 \
  data/ [your imagenet-folder with train and val folders]
```


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 2-gpu machine, run:
```
CUDA_VISIBLE_DEVICES=0,1 python main_lincls.py \
  -a resnet50 \
  --lr 1.0 \
  --batch-size 256 \
  --pretrained checkpoint_0999.pth.tar \
  --dist-url 'tcp://localhost:10014' --multiprocessing-distributed --world-size 1 --rank 0 \
  data/ [your imagenet-folder with train and val folders]
```

## Comparison

Linear classification results on Cifar-10 with (2 or 8) 2080Ti GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>batch-size</th>
<th valign="bottom">ResNet-50<br/>top-1 acc.</th>
<th valign="bottom">ResNet-50<br/>top-5 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">SupCrossEntropy </td>
<td align="center">500</td>
<td align="center">1024</td>
<td align="center">95.0</td>
<td align="center">-</td>
</tr>
<tr><td align="left">SupContrast </td>
<td align="center">1000</td>
<td align="center">1024</td>
<td align="center">96.0</td>
<td align="center">-</td>
</tr>
<tr><td align="left">SupContrast (Our Rerun)</td>
<td align="center">1000</td>
<td align="center">1024</td>
<td align="center">95.6</td>
<td align="center">-</td>
</tr>

<tr><td align="left">MoCo v2</td>
<td align="center">200</td>
<td align="center">256</td>
<td align="center">87.4</td>
<td align="center">99.6</td>
</tr>
<tr><td align="left">MoCo v2 + SupContrast_In </td>
<td align="center">200</td>
<td align="center">256</td>
<td align="center">95.4</td>
<td align="center">99.9</td>
</tr>
<tr><td align="left">MoCo v2</td>
<td align="center">1000</td>
<td align="center">256</td>
<td align="center">93.6</td>
<td align="center">99.8</td>
</tr>
<tr><td align="left">MoCo v2 + SupContrast_In </td>
<td align="center">1000</td>
<td align="center">256</td>
<td align="center">96.1</td>
<td align="center">99.8</td>
</tr>
<tr><td align="left">MoCo v2 + SupContrast_Out </td>
<td align="center">1000</td>
<td align="center">256</td>
<td align="center">96.1</td>
<td align="center">99.9</td>
</tr>
<tr><td align="left">MoCo v2 + SupContrast_Suploss </td>
<td align="center">1000</td>
<td align="center">256</td>
<td align="center">96.0</td>
<td align="center">99.9</td>
</tr>
</tbody></table>


## Reference

This repository references papers [MoCo](https://arxiv.org/abs/1911.05722), [MoCo v2](https://arxiv.org/abs/2003.04297) and [SupContrast](https://arxiv.org/abs/2004.11362):
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```

```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```


```
@Article{khosla2020supervised,
  author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
  title   = {Supervised Contrastive Learning},
  journal = {arXiv preprint arXiv:2004.11362},
  year    = {2020},
}
```
