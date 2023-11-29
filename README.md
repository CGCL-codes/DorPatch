# DorPatch: Distributed and Occlusion-Robust Adversarial Patch to Evade Certifiable Defenses
By Chaoxiang He, Bin Benjamin Zhu

This repository contains the code that implements the Distributed and Occlusion-Robust Patch (DorPatch) described in our paper "DorPatch: Distributed and Occlusion-Robust adversarial Patch to Evade Certifiable Defenses" published at the Network and Distributed System Security Symposium (NDSS) 2024.

## Instructions to Run the Code
1. Create the python environment using conda from requirements.txt using

   `$ conda create -n pytorch_dorpatch --file requirements.txt -c pytorch -c nvidia`

   and switch to the environment by

   `$ conda activate pytorch_dorpatch`

2. Prepare the dataset and pre-trained model.
   - Download the corresponding pretrained model (E.g., 'resnetv2_50x1_bit_distilled_cutout2_128_imagenet.pth' for imagenet from [PatchCleanser](https://github.com/inspire-group/PatchCleanser)) from [here](https://drive.google.com/drive/folders/10H1HIhJ6V8sO99x8g4WxtTahCrTkyTCF) and put it into "pretrained_models/$dataset/".

   - Download the corresponding dataset (E.g., download the validation set of imagenet from [here](https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)) and prepare the data in the directoty $dataset_directory.
3. Run the code with

   `$ python main.py --data_dir $dataset_directory`


## Citation
If you are using our code for research purpose, please cite our paper.

```
@inproceedings{dorpatch_ndss24,
  author    = {Chaoxiang He and Xiaojing Ma and Bin Benjamin Zhu and Yimiao Zeng and 
               Hanqing Hu and Xiaofan Bai and Hai Jin and Dongmei Zhang},
  title     = {{DorPatch}: Distributed and Occlusion-Robust Adversarial Patch
               to Evade Certifiable Defenses},
  booktitle = {Network and Distributed System Security Symposium, {NDSS}, 2024,
               San Diego, CA, USA, February 26 - March 1, 2024},
  publisher = {The Internet Society},
  year      = {2024},
  url       = {https://dx.doi.org/10.14722/ndss.2024.24920}
}
```

## Paper's Link
Our paper can be downloaded [here](https://www.ndss-symposium.org/ndss-paper/dorpatch-distributed-and-occlusion-robust-adversarial-patch-to-evade-certifiable-defenses/).
