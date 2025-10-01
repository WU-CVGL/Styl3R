# ================== NVS pretraining stage ==================
# train multi-view model
# context_view = 2, sh 0

# >>> proxy >>>
export http_proxy=http://10.0.1.68:48889 &&\
export https_proxy=http://10.0.1.68:48889 &&\
export HTTP_PROXY=http://10.0.1.68:48889 &&\
export HTTPS_PROXY=http://10.0.1.68:48889 &&\
export NO_PROXY="localhost,127.0.0.0/8,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,cvgl.lab,.cvgl.lab,westlake.edu.cn,.westlake.edu.cn,.edu.cn" &&\
export no_proxy="localhost,127.0.0.0/8,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,lins.lab,.lins.lab,westlake.edu.cn,.westlake.edu.cn,.edu.cn"

export WANDB_BASE_URL='http://10.0.1.68:8081'
export WANDB_API_KEY='local-ba9aff568438e57aa60ac9644993e7ae048cdd61'
# <<< proxy <<<

# NVS pretrain
# CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/noposplat/bin/python -m src.main_style +experiment=re10k_3view_style_1x \
#     wandb.mode=online \
#     wandb.project=noposplat_xiang_token_style_pretrain \
#     wandb.name=re10k_multi-view_tok-sty-NVS-pretrain \
#     data_loader.train.batch_size=4 \
#     dataset.re10k_style.view_sampler.num_context_views=2 \
#     trainer.val_check_interval=500 \
#     checkpointing.every_n_train_steps=3125 \
#     model.encoder.stylized=False \
#     model.encoder.gaussian_adapter.sh_degree=0 \
#     # > "./logs/output2.log" 2>&1

# PYTHONUNBUFFERED=1 /opt/conda/envs/noposplat/bin/python -m src.main_style +experiment=re10k_3view_style_8x8 \
#     wandb.mode=online \
#     wandb.project=noposplat_xiang_token_style_pretrain \
#     wandb.name=re10k_4multi-view_352x640_tok-sty-NVS-pretrain \
#     data_loader.train.batch_size=1 \
#     dataset.re10k_style.input_image_shape=[352,640] \
#     dataset.re10k_style.view_sampler.num_context_views=4 \
#     dataset.re10k_style.view_sampler.num_target_views=6 \
#     trainer.val_check_interval=500 \
#     checkpointing.every_n_train_steps=3125 \
#     model.encoder.stylized=False \
#     model.encoder.gaussian_adapter.sh_degree=0 \
#     checkpointing.load='outputs/exp_re10k_4multi-view_tok-sty-NVS-pretrain/2025-05-03_14-44-57/checkpoints/epoch_0-step_18750.ckpt' \
#     > "/wangpeng/liuxiang/noposplat_private/logs/output.log" 2>&1 

# PYTHONUNBUFFERED=1 /opt/conda/envs/noposplat/bin/python -m src.main_style +experiment=re10k_3view_style_8x8 \
#     wandb.mode=online \
#     wandb.project=noposplat_xiang_token_style_pretrain \
#     wandb.name=re10k_multi-view_352x640_tok-sty-NVS-pretrain \
#     data_loader.train.batch_size=2 \
#     dataset.re10k_style.input_image_shape=[352,640] \
#     dataset.re10k_style.view_sampler.num_context_views=2 \
#     trainer.val_check_interval=500 \
#     checkpointing.every_n_train_steps=3125 \
#     model.encoder.stylized=False \
#     model.encoder.gaussian_adapter.sh_degree=0 \
#     checkpointing.load='outputs/exp_re10k_multi-view_tok-sty-NVS-pretrain/2025-04-29_19-47-17/checkpoints/epoch_0-step_15000.ckpt' \
#     > "/wangpeng/liuxiang/noposplat_private/logs/output.log" 2>&1 

PYTHONUNBUFFERED=1 /opt/conda/envs/noposplat/bin/python -m src.main_style +experiment=re10k_3view_style_8x8 \
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
    checkpointing.load='outputs/exp_re10k_multi-view_352x640_tok-sty-NVS-pretrain/2025-07-29_20-15-05/checkpoints/epoch_0-step_18750.ckpt' \
    > "/wangpeng/liuxiang/noposplat_private/logs/output.log" 2>&1 