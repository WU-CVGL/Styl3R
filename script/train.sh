# 1x8
python -m src.main_style +experiment=re10k_style_1x8 \
    wandb.mode=online \
    wandb.name=re10k_feature_head_full_test \
    data_loader.train.batch_size=8 \
    trainer.val_check_interval=1000 \
    checkpointing.every_n_train_steps=25_000

# 4x8
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m src.main_style +experiment=re10k_style_4x8 \
    wandb.mode=online \
    wandb.name=re10k_feature_head_512_full_4x \
    model.encoder.appearance_feature_dim=512

# 4x8, 256, sh 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main_style +experiment=re10k_style_4x8 \
    wandb.mode=online \
    wandb.name=re10k_feature_head_256_full_4x_sh0 \
    model.encoder.appearance_feature_dim=256 \
    model.encoder.gaussian_adapter.sh_degree=0

# style training 4x8
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main_style +experiment=re10k_style_4x8 \
    wandb.mode=online \
    wandb.name=re10k_style_training_4x8_h4 \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=style \
    model.encoder.pretrained_weights='ckpts/pretrained_attribute_head.ckpt'

# style training for 512-dim feature head with relu4_1
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m src.main_style +experiment=re10k_style_4x8 \
    wandb.mode=online \
    wandb.name=re10k_style_training_4x8_512_h4 \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=style \
    model.encoder.pretrained_weights='outputs/exp_re10k_feature_head_512_full_4x/2025-03-20_02-06-01/checkpoints/epoch_0-step_37500.ckpt' \
    model.encoder.appearance_feature_dim=512

# style training 1x8
CUDA_VISIBLE_DEVICES=0 python -m src.main_style +experiment=re10k_style_1x8 \
    wandb.mode=online \
    wandb.name=re10k_style_training_1x8 \
    trainer.val_check_interval=400 \
    model.encoder.stylized=True \
    loss=style \
    model.encoder.pretrained_weights='ckpts/pretrained_attribute_head.ckpt'

# style training AdaAttN 4x4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main_style +experiment=re10k_style_4x4 \
    wandb.mode=online \
    wandb.name=re10k_style_training_4x4_adaattn \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=adaattn \
    model.encoder.pretrained_weights='ckpts/pretrained_attribute_head.ckpt'

# larger style loss weight (no big difference)
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m src.main_style +experiment=re10k_style_4x4 \
    wandb.mode=online \
    wandb.name=re10k_style_training_4x4_adaattn_lam0.6 \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=adaattn \
    loss.adaattn.lam=0.6 \
    model.encoder.pretrained_weights='ckpts/pretrained_attribute_head.ckpt'

# 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.main_style +experiment=re10k_style_4x4 \
    wandb.mode=online \
    wandb.name=re10k_style_training_4x4_adaattn_sh0_finetue_feature_head \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=adaattn \
    model.encoder.pretrained_weights='outputs/exp_re10k_feature_head_256_full_4x_sh0/2025-03-22_01-08-05/checkpoints/epoch_0-step_37500.ckpt' \
    model.encoder.gaussian_adapter.sh_degree=0

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m src.main_style +experiment=re10k_style_4x4 \
    wandb.mode=online \
    wandb.name=re10k_style_training_4x4_adaattn_sh0 \
    trainer.val_check_interval=100 \
    model.encoder.stylized=True \
    loss=adaattn \
    model.encoder.pretrained_weights='outputs/exp_re10k_feature_head_256_full_4x_sh0/2025-03-22_01-08-05/checkpoints/epoch_0-step_37500.ckpt' \
    model.encoder.gaussian_adapter.sh_degree=0