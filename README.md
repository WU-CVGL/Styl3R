<h1 align="center"> Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles </h1>

<div align="center">
  <!-- <a href=https://nickisdope.github.io/Styl3R/ target="_blank"><img src="https://img.shields.io/badge/Project-Page-green.svg" height=22px></a>
  <a href="https://arxiv.org/abs/<ARXIV PAPER ID>" target="_blank"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a> -->

<h3 align="center"><a href="https://arxiv.org/abs/2505.21060">Paper</a> | <a href="https://nickisdope.github.io/Styl3R/">Project Page</a> </h3>

📢 The training and inference code is updated, please inform us if you have encountered some issues.

<!-- <img src="assets/teaser_crop-9.gif" width="600" height="258"/> -->

![teaser](assets/teaser_crop-9.gif)

</div>

## 📝 Summary
- We introduce a feed-forward network for 3D stylization that operates on sparse, unposed content images and an arbitrary style image, does not require test-time optimization, and generalizes well to out-of-domain inputs.
- We design a dual-branch network architecture that decouples appearance and structure modeling, effectively enhancing the joint learning of novel view synthesis and 3D stylization.
- Our method achieves state-of-the-art zero-shot 3D stylization performance, surpassing existing zero-shot methods and approximate the efficacy of style-specific optimization techniques.

## 📦 Pre-trained Checkpoints

## 📚 Datasets
Please refer to **[DATASETS.md](DATASETS.md)** for detailed dataset preparation.

After setting up the datasets (**RE10K**, **DL3DV**, and **WikiArts**):

1. Modify the dataset paths marked with `# TODO` in  
   [`generate_scene_style_correspondences.py`](src/test/generate_scene_style_correspondences.py).  
   Then, run the script to generate **scene-style correspondence** `.json` files.

2. Update the `root` paths in:  
   - [`re10k_style.yaml`](config/dataset/re10k_style.yaml)  
   - [`dl3dv_style.yaml`](config/dataset/dl3dv_style.yaml)  

3. Update the `style_root` paths in:  
   - [`re10k_3view_style_1x.yaml`](config/experiment/re10k_3view_style_1x.yaml)  
   - [`re10k_3view_style_8x8.yaml`](config/experiment/re10k_3view_style_8x8.yaml)  
   - [`re10k_dl3dv_3view_style_1x.yaml`](config/experiment/re10k_dl3dv_3view_style_1x.yaml)  
   - [`re10k_dl3dv_3view_style_8x8.yaml`](config/experiment/re10k_dl3dv_3view_style_8x8.yaml)


## 🚀 Training

### Stage 1: NVS Pretraining
#### (1) 2-view Model (RE10K)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m src.main_style +experiment=re10k_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_multi-view_tok-sty-NVS-pretrain \
    data_loader.train.batch_size=10 \
    dataset.re10k_style.view_sampler.num_context_views=2 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=3125 \
    model.encoder.stylized=False \
    model.encoder.gaussian_adapter.sh_degree=0
```

#### (2) 4-view Model (RE10K)
Initialize from the pretrained **2-view** model by setting `checkpointing.load` to the corresponding checkpoint path.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.main_style +experiment=re10k_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_4multi-view_tok-sty-NVS-pretrain \
    data_loader.train.batch_size=6 \
    dataset.re10k_style.view_sampler.num_context_views=4 \
    dataset.re10k_style.view_sampler.num_target_views=6 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=3125 \
    model.encoder.stylized=False \
    model.encoder.gaussian_adapter.sh_degree=0 \
    checkpointing.load='outputs/exp_re10k_multi-view_tok-sty-NVS-pretrain/2025-04-29_19-47-17/checkpoints/epoch_0-step_15000.ckpt'
```

#### (3) 4-view Model (RE10K + DL3DV)

Initialize from the pretrained **4-view RE10K** model.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.main_style +experiment=re10k_dl3dv_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_dl3dv_4multi-view_tok-sty-NVS-pretrain \
    data_loader.train.batch_size=3 \
    dataset.re10k_style.view_sampler.num_context_views=4 \
    dataset.re10k_style.view_sampler.num_target_views=6 \
    dataset.dl3dv_style.view_sampler.num_context_views=4 \
    dataset.dl3dv_style.view_sampler.num_target_views=6 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=3125 \
    model.encoder.stylized=False \
    model.encoder.gaussian_adapter.sh_degree=0 \
    checkpointing.load='outputs/exp_re10k_4multi-view_tok-sty-NVS-pretrain/2025-05-03_14-44-57/checkpoints/epoch_0-step_18750.ckpt'
```

### Stage 2: Stylization Fine-tuning
#### (1) 2-view Model (RE10K)
Initialize from the **2-view NVS-pretrained** checkpoint.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m src.main_style +experiment=re10k_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_debug \
    wandb.name=re10k_multi-view_tok-sty-stylization_content-h3-h4 \
    data_loader.train.batch_size=14 \
    dataset.re10k_style.view_sampler.num_context_views=2 \
    trainer.max_steps=35001 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=5000 \
    model.encoder.stylized=True \
    model.encoder.gaussian_adapter.sh_degree=0 \
    loss=style \
    train.identity_loss=true \
    checkpointing.load='outputs/exp_re10k_multi-view_tok-sty-NVS-pretrain/2025-04-29_19-47-17/checkpoints/epoch_0-step_15000.ckpt'
```

#### (2) 4-view Model (RE10K)
Initialize from the **4-view NVS-pretrained** checkpoint.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.main_style +experiment=re10k_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_debug \
    wandb.name=re10k_4multi-view_tok-sty-stylization \
    data_loader.train.batch_size=8 \
    dataset.re10k_style.view_sampler.num_context_views=4 \
    dataset.re10k_style.view_sampler.num_target_views=6 \
    trainer.max_steps=35001 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=5000 \
    model.encoder.stylized=True \
    model.encoder.gaussian_adapter.sh_degree=0 \
    loss=style \
    train.identity_loss=true \
    checkpointing.load='outputs/exp_re10k_4multi-view_tok-sty-NVS-pretrain/2025-05-03_14-44-57/checkpoints/epoch_0-step_18750.ckpt'
```

#### (3) 4-view Model (RE10K + DL3DV)
Initialize from the **4-view NVS-pretrained** checkpoint on RE10K + DL3DV.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.main_style +experiment=re10k_dl3dv_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_debug \
    wandb.name=re10k_dl3dv_4multi-view_tok-sty-stylization \
    data_loader.train.batch_size=3 \
    dataset.re10k_style.view_sampler.num_context_views=4 \
    dataset.re10k_style.view_sampler.num_target_views=6 \
    dataset.dl3dv_style.view_sampler.num_context_views=4 \
    dataset.dl3dv_style.view_sampler.num_target_views=6 \
    trainer.max_steps=35001 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=5000 \
    model.encoder.stylized=True \
    model.encoder.gaussian_adapter.sh_degree=0 \
    loss=style \
    train.identity_loss=true \
    checkpointing.load='outputs/exp_re10k_dl3dv_4multi-view_tok-sty-NVS-pretrain/2025-05-05_21-11-11/checkpoints/epoch_0-step_15625.ckpt'
```

## 🎨 Inference
> **Note:** Set `checkpointing.load` to a checkpoint obtained **after stylization fine-tuning**.

### Stylize a scene in RE10K
```
CUDA_VISIBLE_DEVICES=0 python -m infer_model_re10k \
    +experiment=re10k_3view_style_1x.yaml \
    wandb.name=re10k_tok-sty_inference \
    model.encoder.stylized=True \
    model.encoder.gaussian_adapter.sh_degree=0 \
    test.pose_align_steps=50 \
    checkpointing.load=outputs/exp_re10k_dl3dv_4multi-view_tok-sty-stylization_b2x3/2025-05-06_19-31-34/checkpoints/epoch_0-step_35000.ckpt
```

https://github.com/user-attachments/assets/ad619cbd-6c64-4993-960f-c5978e9b3522

### Stylize a scene in Tanks and Temples (COLMAP format)
```
CUDA_VISIBLE_DEVICES=0 python -m infer_model_colmap \
    +experiment=re10k_3view_style_1x.yaml \
    wandb.name=video_colmap_tok-sty_inference \
    model.encoder.stylized=True \
    model.encoder.gaussian_adapter.sh_degree=0 \
    test.pose_align_steps=50 \
    checkpointing.load=outputs/exp_re10k_dl3dv_4multi-view_tok-sty-stylization_b2x3/2025-05-06_19-31-34/checkpoints/epoch_0-step_35000.ckpt
```

## 🧩 Debugging
A [launch.json](.vscode/launch.json) configuration is provided for debugging all training and inference commands in **VSCode**.
- Update the `python` field to your Python interpreter path.
- `Update `checkpointing.load` to your corresponding checkpoint path.

## 🚧 TODO
- [x] Release inference code and pretrained models
- [ ] Release gradio demo 
- [x] Release training code

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{wang2025styl3r,
  title={Styl3R: Instant 3D Stylized Reconstruction for Arbitrary Scenes and Styles},
  author={Wang, Peng and Liu, Xiang and Liu, Peidong},
  journal={arXiv preprint arXiv:2505.21060},
  year={2025}
}
