
# Spatially-Correlative Loss

[arXiv](https://arxiv.org/abs/2104.00854) | [website](http://www.chuanxiaz.com/publication/flsesim/)
<br>

<img src='imgs/FSeSim-frame.gif' align="center">
<br>

We provide the Pytorch implementation of "The Spatially-Correlative Loss for Various Image Translation Tasks". Based on the inherent self-similarity of object, we propose a new structure-preserving loss for one-sided unsupervised I2I network. The new loss will *deal only with spatial relationship of repeated signal, regardless of their original absolute value*. 

[The Spatially-Correlative Loss for Various Image Translation Tasks](https://arxiv.org/abs/2104.00854) <br>
[Chuanxia Zheng](http://www.chuanxiaz.com), [Tat-Jen Cham](http://www.ntu.edu.sg/home/astjcham/), [Jianfei Cai](https://research.monash.edu/en/persons/jianfei-cai) <br>
NTU and Monash University <br>
In CVPR2021 <br>

## ToDo
- a simple example to use the proposed loss

## Example Results

### Unpaired Image-to-Image Translation

<img src='imgs/unpairedI2I-translation.gif' align="center">

### Single Image Translation

<img src='imgs/single-translation.gif' align="center">

### [More results on project page](http://www.chuanxiaz.com/publication/flsesim/)

## Getting Started

### Installation
This code was tested with Pytorch 1.7.0, CUDA 10.2, and Python 3.7

- Install Pytoch 1.7.0, torchvision, and other dependencies from [http://pytorch.org](http://pytorch.org)
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate) for visualization

```
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/lyndonzheng/F-LSeSim
cd F-LSeSim
```

### [Datasets](https://github.com/taesungp/contrastive-unpaired-translation/blob/master/docs/datasets.md)
Please refer to the original [CUT](https://github.com/taesungp/contrastive-unpaired-translation) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to download datasets and learn how to create your own datasets.

### Training

- Train the *single-modal* I2I translation model:

```
sh ./scripts/train_sc.sh 
```

- Set ```--use_norm``` for cosine similarity map, the default similarity is dot-based attention score. ```--learned_attn, --augment``` for the learned self-similarity.
- To view training results and loss plots, run ```python -m visdom.server``` and copy the URL [http://localhost:port](http://localhost:port).
- Training models will be saved under the **checkpoints** folder.
- The more training options can be found in the **options** folder.
<br><br>


- Train the *single-image* translation model:

```
sh ./scripts/train_sinsc.sh 
```

As the *multi-modal* I2I translation model was trained on [MUNIT](https://github.com/NVlabs/MUNIT), we would not plan to merge the code to this repository. If you wish to obtain multi-modal results, please contact us at chuanxia001@e.ntu.edu.sg.

### Testing

- Test the *single-modal* I2I translation model:

```
sh ./scripts/test_sc.sh
```

- Test the *single-image* translation model:

```
sh ./scripts/test_sinsc.sh
```

- Test the FID score for all training epochs:

```
sh ./scripts/test_fid.sh
```

### Pretrained Models

Download the pre-trained models (will be released soon) using the following links and put them under```checkpoints/``` directory.

- ```Single-modal translation model```: [horse2zebra](https://drive.google.com/drive/folders/1k8Y5R6CnaDwfkha_lD5_yQTvoajoU6GR?usp=sharing), [semantic2image](https://drive.google.com/drive/folders/1xnF6wLTPhD35-2It8IIomJRhFZdr2qXp?usp=sharing), [apple2orange](https://drive.google.com/drive/folders/1Z9PwxkWlakDdv12Jha6WJRgO6cSfEZGs?usp=sharing)
- ```Single-image translation model```: [image2monet](https://drive.google.com/drive/folders/1QcGY9H0USWHJtcifRMWh_KHOJszME6-U?usp=sharing)

## Citation
```
@inproceedings{zheng2021spatiallycorrelative,
  title={The Spatially-Correlative Loss for Various Image Translation Tasks},
  author={Zheng, Chuanxia and Cham, Tat-Jen and Cai, Jianfei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Acknowledge
Our code is developed based on [CUT](https://github.com/taesungp/contrastive-unpaired-translation) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation,  [LPIPS](https://github.com/richzhang/PerceptualSimilarity) for diversity score, and [D&C](https://github.com/clovaai/generative-evaluation-prdc) for density and coverage evaluation.




