# overfit on a single scene
CUDA_VISIBLE_DEVICES=2 python -m src.main_style +experiment=re10k_style_1x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_debug \
    wandb.name=re10k_token_style_overfit \
    data_loader.train.batch_size=1 \
    dataset.re10k_style.overfit_to_scene=11491a312c6b8f58 \
    dataset.re10k_style.specified_style_image_path=colmap_test_data/styles/tiger.jpg \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=style

# train token stylizer on a single GPU
CUDA_VISIBLE_DEVICES=6 python -m src.main_style +experiment=re10k_style_1x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_debug \
    wandb.name=re10k_token_style_dpt_pre-sty-enc_finetune \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=style

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m src.main_style +experiment=re10k_style_4x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_debug \
    wandb.name=re10k_token_style \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=style

# NVS pretraining
CUDA_VISIBLE_DEVICES=1,2,6,7 python -m src.main_style +experiment=re10k_style_4x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_token_style_pretrain \
    trainer.val_check_interval=100 \
    checkpointing.every_n_train_steps=5_000 \
    model.encoder.stylized=False \
    model.encoder.gs_sh_head_type=dpt \
    train.identity_loss=false \
    train.distiller=mast3r \
    train.distill_max_steps=1000 \
    # checkpointing.load='outputs/exp_re10k_token_style_pretrain/2025-04-18_21-12-18/checkpoints/epoch_0-step_35000.ckpt' \
    # trainer.max_steps=40000

CUDA_VISIBLE_DEVICES=7 python -m src.main_style +experiment=re10k_style_1x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_token_style_pretrain_branched \
    trainer.val_check_interval=100 \
    checkpointing.every_n_train_steps=5_000 \
    model.encoder.stylized=False \
    model.encoder.gs_sh_head_type=dpt \
    train.identity_loss=false \
    train.distiller=mast3r \
    train.distill_max_steps=30000 

# NVS pretraining after pts3d distillation
python -m src.main_style +experiment=re10k_style_NVS_8x12 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_token_style_pretrain_pre-distilled \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=5_000 

# noposplat baseline for NVS
python -m src.main +experiment=re10k_8x12 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_noposplat_baseline \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=5_000

# distill pts3d
python -m src.main_style +experiment=re10k_style_distill_8x32 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_token_style_distill_pretrain 

# stylization training
python -m src.main_style +experiment=re10k_style_8x6 \
    wandb.mode=online \
    wandb.name=re10k_token_style 

# ================== NVS pretraining stage ==================
# train multi-view model
# context_view = 2, sh 0
# NVS pretrain
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

# context_view = 2
# input_image_shape [352, 640]
# init from 2 view model trained on [256, 256]
python -m src.main_style +experiment=re10k_3view_style_8x8 \
    wandb.mode=disabled \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_multi-view_HD_tok-sty-NVS-pretrain \
    data_loader.train.batch_size=2 \
    dataset.re10k_style.input_image_shape=[720,1280] \
    dataset.re10k_style.view_sampler.num_context_views=2 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=3125 \
    model.encoder.stylized=False \
    model.encoder.gaussian_adapter.sh_degree=0 \
    checkpointing.load='outputs/exp_re10k_multi-view_tok-sty-NVS-pretrain/2025-04-29_19-47-17/checkpoints/epoch_0-step_15000.ckpt'

# context_view = 4, target_view = 6, sh 0
# init from 2 view model
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

# context_view = 4, target_view = 6, sh 0
# input_image_shape [352, 640]
# init from 4 view model trained on [256, 256]
python -m src.main_style +experiment=re10k_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_pretrain \
    wandb.name=re10k_4multi-view_352x640_tok-sty-NVS-pretrain \
    data_loader.train.batch_size=2 \
    dataset.re10k_style.input_image_shape=[352, 640] \
    dataset.re10k_style.view_sampler.num_context_views=4 \
    dataset.re10k_style.view_sampler.num_target_views=6 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=3125 \
    model.encoder.stylized=False \
    model.encoder.gaussian_adapter.sh_degree=0 \
    checkpointing.load='outputs/exp_re10k_4multi-view_tok-sty-NVS-pretrain/2025-05-03_14-44-57/checkpoints/epoch_0-step_18750.ckpt'


# re10k + dl3dv, init from re10k 4view model
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

# ================== stylization stage ==================
# stylization fintuning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m src.main_style +experiment=re10k_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_debug \
    wandb.name=re10k_multi-view_tok-sty-stylization \
    data_loader.train.batch_size=14 \
    dataset.re10k_style.view_sampler.num_context_views=2 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=3125 \
    model.encoder.stylized=True \
    model.encoder.gaussian_adapter.sh_degree=0 \
    loss=style \
    train.identity_loss=true \
    checkpointing.load='outputs/exp_re10k_multi-view_tok-sty-NVS-pretrain/2025-04-29_19-47-17/checkpoints/epoch_0-step_15000.ckpt'

# stylization fintuning (content loss with h3 and h4), this works better than the upper one!
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

# for 2 view, resolution [352, 640]
python -m src.main_style +experiment=re10k_3view_style_8x8 \
    wandb.mode=online \
    wandb.project=noposplat_xiang_token_style_debug \
    wandb.name=re10k_multi-view_352x640_tok-sty-stylization_content-h3-h4 \
    data_loader.train.batch_size=4 \
    dataset.re10k_style.input_image_shape=[352,640] \
    dataset.re10k_style.view_sampler.num_context_views=2 \
    trainer.max_steps=35001 \
    trainer.val_check_interval=500 \
    checkpointing.every_n_train_steps=5000 \
    model.encoder.stylized=True \
    model.encoder.gaussian_adapter.sh_degree=0 \
    loss=style \
    train.identity_loss=true \
    checkpointing.load='outputs/exp_re10k_multi-view_352x640_tok-sty-NVS-pretrain/2025-07-29_20-15-05/checkpoints/epoch_0-step_18750.ckpt'

# for 4 view model
python -m src.main_style +experiment=re10k_3view_style_8x8 \
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

# stylization finetuning with re10k+dl3dv
python -m src.main_style +experiment=re10k_dl3dv_3view_style_8x8 \
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