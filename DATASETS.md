# Datasets

For training, we mainly use [RealEstate10K](https://google.github.io/realestate10k/index.html), [DL3DV](https://github.com/DL3DV-10K/Dataset), and [ACID](https://infinite-nature.github.io/) datasets. We provide the data processing scripts to convert the original datasets to pytorch chunk files which can be directly loaded with this codebase. 

Expected folder structure:

```
├── datasets
│   ├── re10k
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── dl3dv
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
```

By default, we assume the datasets are placed in `datasets/re10k`, `datasets/dl3dv`, and `datasets/acid`. Otherwise you will need to specify your dataset path with `dataset.DATASET_NAME.roots=[YOUR_DATASET_PATH]` in the running script.

We also provide instructions to convert additional datasets to the desired format.



## RealEstate10K

For experiments on RealEstate10K, we primarily follow [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat) to train and evaluate on 256x256 resolution.

Please refer to [here](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) for acquiring the processed 360p dataset (360x640 resolution).

If you would like to train and evaluate on the high-resolution RealEstate10K dataset, you will need to download the 720p (720x1280) version. Please refer to [here](https://github.com/yilundu/cross_attention_renderer/tree/master/data_download) for the downloading script. Note that the script by default downloads the 360p videos, you will need to modify the`360p` to `720p` in [this line of code](https://github.com/yilundu/cross_attention_renderer/blob/master/data_download/generate_realestate.py#L137) to download the 720p videos.

After downloading the 720p dataset, you can use the scripts [here](https://github.com/dcharatan/real_estate_10k_tools/tree/main/src) to convert the dataset to the desired format in this codebase.



## DL3DV

In the DL3DV experiments, we trained with RealEstate10k at 256x256, 512x512 and 368x640 resolutions, respectively.

For the training set, we use the [DL3DV-480p](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) dataset (270x480 resolution), where the 140 scenes in the test set are excluded during processing the training set. After downloading the [DL3DV-480p](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) dataset, you can then use the script [src/scripts/convert_dl3dv.py](src/scripts/convert_dl3dv.py) to convert the training set.

Please note that you will need to update the dataset paths in the aforementioned processing scripts.

If you would like to train on the high-resolution DL3DV dataset, you will need to download the [DL3DV-960P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P) version (540x960 resolution). Simply follow the same procedure for data processing (use the `images_4` folder instead of `images_8`).



## Test Only Datasets

### DTU

You can download processed DTU datasets from [here](https://drive.google.com/file/d/1Bd9il_O1jjom6Lk9NP77K8aeQP0d9eEt/view?usp=drive_link).

### ScanNet++

You can download processed ScanNet++ datasets from [here](https://drive.google.com/file/d/1bmkNjXuWLhAOkf-6liS0yyARCybZyQSE/view?usp=sharing).

### ScanNet-1500

For ScanNet-1500, you need to download download `test.npy` [here](https://github.com/zju3dv/LoFTR/blob/master/assets/scannet_test_1500/test.npz) and the corresponding test dataset [here](https://drive.google.com/file/d/1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd/view).

## Custom Datasets

If you would like to train and/or evaluate on additional datasets, just modify the [data processing scripts](src/scripts) to convert the dataset format. Kindly note the [camera conventions](https://github.com/cvg/NoPoSplat/tree/main?tab=readme-ov-file#camera-conventions) used in this codebase.
