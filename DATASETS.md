# Datasets

For training, we mainly use [RealEstate10K](https://google.github.io/realestate10k/index.html), [DL3DV](https://github.com/DL3DV-10K/Dataset) and [WikiArt](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan) datasets. We provide the data processing scripts to convert the original datasets to pytorch chunk files which can be directly loaded with this codebase. 

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

By default, we assume the datasets are placed in `datasets/re10k`, `datasets/dl3dv`. Otherwise you will need to specify your dataset path with `dataset.DATASET_NAME.roots=[YOUR_DATASET_PATH]` in the running script.

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



## WikiArt
Download link [here](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan)

Expected folder structure:

```
├── datasets
│   ├── wikiart
│   ├── ├── images
│   ├── ├── ├── abstract
│   ├── ├── ├── ├── *.jpg
│   ├── ├── ├── animal-painting
│   ├── ├── ├── ├── *.jpg
│   ├── ├── ├── ...
│   ├── ├── ├── ├── *.jpg
```

Then you can run the process code [here](src\scripts\wikiart_process.py), you should modify the dataset paths marked by `# TODO`. 

Expected folder structure:

```
├── datasets
│   ├── wikiart_combine
│   ├── ├── images_combine
│   ├── ├── ├── train
│   ├── ├── ├── ├── *.jpg
│   ├── ├── ├── test
│   ├── ├── ├── ├── *.jpg
```
