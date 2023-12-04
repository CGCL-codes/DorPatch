# DorPatch: Distributed and Occlusion-Robust Adversarial Patch to Evade Certifiable Defenses

By Chaoxiang He, Bin Benjamin Zhu

This repository contains the code for the paper "DorPatch: Distributed and Occlusion-Robust Adversarial Patch to Evade Certifiable Defenses", published at the Network and Distributed System Security Symposium (NDSS) 2024.

## Files

```
├── README.md                        #this file 
├── requirement.txt                  #required package
| 
├── main.py                          #entrance to run the code
├── utils.py                         #utils for the code
├── attack.py                        #code for attacking methods (e.g., DorPatch)
| 
├── defenses
|   └── PatchCleanser.py             #code for the SOTA certifiable defense of PatchCleanser
|
├── pretrained_models
|   └──imagenet                      #model directory for imagenet
|
└── results                          #directory for generated results

$dataset_directory
    └──imagenet                      #data directory for imagenet
```

## Instructions to Run the Code

1. Create the Python environment using conda from `requirements.txt`:

   `$ conda create -n pytorch_dorpatch --file requirements.txt -c pytorch -c nvidia`

   and switch to the environment by

   `$ conda activate pytorch_dorpatch`

2. Prepare the dataset and pre-trained model.
   - Download the corresponding pretrained model (e.g., 'resnetv2_50x1_bit_distilled_cutout2_128_imagenet.pth' for ImageNet from [PatchCleanser](https://github.com/inspire-group/PatchCleanser)) from [here](https://drive.google.com/drive/folders/10H1HIhJ6V8sO99x8g4WxtTahCrTkyTCF) and put it into "pretrained_models/$dataset/". $dataset is a variable referring to a dataset (e.g., $dataset can be set to 'imagenet' for ImageNet dataset).

   - Download the corresponding dataset (e.g., download the validation set of ImageNet from [here](https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)) and prepare the data in the directory `$dataset_directory`. `$dataset_directory` could be an arbitrary directory that stores datasets and is shared for data-reusing (e.g., the validation set of ImageNet is prepared in '/data/imagenet/val', `$dataset_directory` should be set to '/data').
3. Run the code with
   `$ python main.py --data_dir $dataset_directory`

## Citation

If you are using our code for research purposes, please cite our paper.

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
Our paper can be downloaded from [here](https://www.ndss-symposium.org/ndss-paper/dorpatch-distributed-and-occlusion-robust-adversarial-patch-to-evade-certifiable-defenses/).
