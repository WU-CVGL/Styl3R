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
# stylization fintuning (content loss with h3 and h4), this works better
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