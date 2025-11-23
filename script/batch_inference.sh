#!/bin/bash

frame_groups=(
    "[0,1,2,3]"
    "[4,5,6,7]"
    "[8,9,10,11]"
    "[12,13,14,15]"
)  # could be extended to more frames

for style in $(seq -w 0 49); do  # could be extended to more style images
    for frames in "${frame_groups[@]}"; do

        echo ">>> Running style ${style}, frames ${frames}"

        CUDA_VISIBLE_DEVICES=0 python -m infer_model_tnt_batch \
            +experiment=re10k_3view_style_1x.yaml \
            +scene_name=truck \
            +frame_ids=${frames} \
            +style_id="'${style}'" \
            wandb.name=video_colmap_tok-sty_inference \
            model.encoder.stylized=True \
            model.encoder.gaussian_adapter.sh_degree=0 \
            test.pose_align_steps=50 \
            checkpointing.load=outputs/re10k_dl3dv_4v/re10k_dl3dv_4v.ckpt \
            hydra.run.dir=outputs_render

    done
done
