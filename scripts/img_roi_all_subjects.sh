model_type=BI # fixed after this time
subj=1
fusion_ver=1 # fixed
mlp_ver=mlp3 # AdaptiveMaxPool1d(h)
cat_loss=softmargin
wd=0.01
pretrained_wgt_path=./trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth

DEFAULT=" --wandb-project Brain2Valence --wandb-entity beotborry --wandb-log --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss}"
# for lr in 1e-5
# do
#     CUDA_VISIBLE_DEVICES=0 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
#     --model-name emotic_pretrained_emo_vis_roi_finetune_lr_${lr}_pn_512_ep_100 \
#     --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 512 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     --wgt-path ${pretrained_wgt_path} $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
#     --model-name emotic_pretrained_emo_vis_roi_finetune_lr_${lr}_pn_1024_ep_100 \
#     --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 1024 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     --wgt-path ${pretrained_wgt_path} $DEFAULT & wait
    
#     CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
#     --model-name emotic_pretrained_emo_vis_roi_finetune_lr_${lr}_pn_2048_ep_100 \
#     --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 2048 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     --wgt-path ${pretrained_wgt_path} $DEFAULT & wait
    
#     CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
#     --model-name emotic_pretrained_emo_vis_roi_finetune_lr_${lr}_pn_4096_ep_100 \
#     --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 4096 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     --wgt-path ${pretrained_wgt_path} $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=0 python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
#     --model-name emotic_pretrained_emo_vis_roi_finetune_lr_${lr}_pn_512_ep_100 \
#     --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 512 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     --best $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
#     --model-name emotic_pretrained_emo_vis_roi_finetune_lr_${lr}_pn_1024_ep_100 \
#     --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 1024 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     --best $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
#     --model-name emotic_pretrained_emo_vis_roi_finetune_lr_${lr}_pn_2048_ep_100 \
#     --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 2048 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     --best $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
#     --model-name emotic_pretrained_emo_vis_roi_finetune_lr_${lr}_pn_4096_ep_100 \
#     --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 4096 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     --best $DEFAULT & wait
# done


# for lr in 1e-5
# do
    # CUDA_VISIBLE_DEVICES=0 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
    # --model-name emotic_pretrained_emo_roi_finetune_lr_${lr}_pn_512_ep_100 \
    # --epochs 30 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 512 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    # --wgt-path ${pretrained_wgt_path} $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
    # --model-name emotic_pretrained_emo_roi_finetune_lr_${lr}_pn_1024_ep_100 \
    # --epochs 30 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 1024 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    # --wgt-path ${pretrained_wgt_path} $DEFAULT & wait
    
    # CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
    # --model-name emotic_pretrained_emo_roi_finetune_lr_${lr}_pn_2048_ep_100 \
    # --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 2048 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    # --wgt-path ${pretrained_wgt_path} $DEFAULT & wait
    
    # CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
    # --model-name emotic_pretrained_emo_roi_finetune_lr_${lr}_pn_4096_ep_100 \
    # --epochs 30 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 4096 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    # --wgt-path ${pretrained_wgt_path} $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=0 python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
    # --model-name emotic_pretrained_emo_roi_finetune_lr_${lr}_pn_512_ep_100 \
    # --epochs 30 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 512 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    # --best $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
    # --model-name emotic_pretrained_emo_roi_finetune_lr_${lr}_pn_1024_ep_100 \
    # --epochs 30 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 1024 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    # --best $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
    # --model-name emotic_pretrained_emo_roi_finetune_lr_${lr}_pn_2048_ep_100 \
    # --epochs 100 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 2048 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    # --best $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
    # --model-name emotic_pretrained_emo_roi_finetune_lr_${lr}_pn_4096_ep_100 \
    # --epochs 30 --batch-size 4 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 4096 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    # --best $DEFAULT & wait
# done


# for lr in 1e-6 3e-6 5e-6 1e-5 3e-5 5e-5 1e-4
# do
    # CUDA_VISIBLE_DEVICES=0 python3 -W "ignore" main.py --exec_mode train --all-subjects \
    # --model-name vis_roi_scratch_lr_${lr}_bs_52_pn_512_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 512 --data roi --cat-only --fusion-ver ${fusion_ver} \
    # $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode train --all-subjects \
    # --model-name vis_roi_scratch_lr_${lr}_bs_52_pn_1024_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 1024 --data roi --cat-only --fusion-ver ${fusion_ver} \
    # $DEFAULT & wait
    
    # CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode train --all-subjects \
    # --model-name vis_roi_scratch_lr_${lr}_bs_52_pn_2048_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 2048 --data roi --cat-only --fusion-ver ${fusion_ver} \
    # $DEFAULT & wait
    
    # CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode train --all-subjects \
    # --model-name vis_roi_scratch_lr_${lr}_bs_52_pn_4096_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --pool-num 4096 --data roi --cat-only --fusion-ver ${fusion_ver} \
    # $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
#     --model-name vis_roi_scratch_lr_${lr}_bs_52_pn_512_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 512 --data roi --cat-only --fusion-ver ${fusion_ver} \
#     --best $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
#     --model-name vis_roi_scratch_lr_${lr}_bs_52_pn_1024_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 1024 --data roi --cat-only --fusion-ver ${fusion_ver} \
#     --best $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
#     --model-name vis_roi_scratch_lr_${lr}_bs_52_pn_2048_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 2048 --data roi --cat-only --fusion-ver ${fusion_ver} \
#     --best $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
#     --model-name vis_roi_scratch_lr_${lr}_bs_52_pn_4096_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 4096 --data roi --cat-only --fusion-ver ${fusion_ver} \
#     --best $DEFAULT & wait
# done


for lr in 1e-6 3e-6 5e-6 1e-5 3e-5 5e-5 1e-4
do
#     CUDA_VISIBLE_DEVICES=0 python3 -W "ignore" main.py --exec_mode train --all-subjects \
#     --model-name emo_roi_scratch_lr_${lr}_bs_52_pn_512_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 512 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
#     $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode train --all-subjects \
#     --model-name emo_roi_scratch_lr_${lr}_bs_52_pn_1024_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 1024 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
#     $DEFAULT & wait
    
#     CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode train --all-subjects \
#     --model-name emo_roi_scratch_lr_${lr}_bs_52_pn_2048_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 2048 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
#     $DEFAULT & wait
    
#     # CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode train --all-subjects \
#     # --model-name emo_roi_scratch_lr_${lr}_bs_52_pn_4096_ep_50 \
#     # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     # --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     # --brain-backbone ${mlp_ver} --pool-num 4096 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
#     # $DEFAULT & wait

    CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    --model-name emo_roi_scratch_lr_${lr}_bs_52_pn_512_ep_50 \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 512 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    --best $DEFAULT & wait

    CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    --model-name emo_roi_scratch_lr_${lr}_bs_52_pn_1024_ep_50 \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 1024 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    --best $DEFAULT & wait

    CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    --model-name emo_roi_scratch_lr_${lr}_bs_52_pn_2048_ep_50 \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 2048 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    --best $DEFAULT & wait

    CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    --model-name emo_roi_scratch_lr_${lr}_bs_52_pn_4096_ep_50 \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 4096 --data emo_roi --cat-only --fusion-ver ${fusion_ver} \
    --best $DEFAULT & wait
done

# for lr in 1e-6 3e-6 5e-6 1e-5 3e-5 5e-5 1e-4
# do
#     CUDA_VISIBLE_DEVICES=0 python3 -W "ignore" main.py --exec_mode train --all-subjects \
#     --model-name emo_vis_roi_scratch_lr_${lr}_bs_52_pn_512_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 512 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     $DEFAULT & wait

#     CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode train --all-subjects \
#     --model-name emo_vis_roi_scratch_lr_${lr}_bs_52_pn_1024_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 1024 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     $DEFAULT & wait
    
#     CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode train --all-subjects \
#     --model-name emo_vis_roi_scratch_lr_${lr}_bs_52_pn_2048_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 2048 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     $DEFAULT & wait
    
#     CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode train --all-subjects \
#     --model-name emo_vis_roi_scratch_lr_${lr}_bs_52_pn_4096_ep_50 \
#     --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#     --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
#     --brain-backbone ${mlp_ver} --pool-num 4096 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
#     $DEFAULT & wait

    CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    --model-name emo_vis_roi_scratch_lr_${lr}_bs_52_pn_512_ep_50 \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 512 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
    --best $DEFAULT & wait

    CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    --model-name emo_vis_roi_scratch_lr_${lr}_bs_52_pn_1024_ep_50 \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 1024 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
    --best $DEFAULT & wait

    CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    --model-name emo_vis_roi_scratch_lr_${lr}_bs_52_pn_2048_ep_50 \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 2048 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
    --best $DEFAULT & wait

    CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    --model-name emo_vis_roi_scratch_lr_${lr}_bs_52_pn_4096_ep_50 \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 4096 --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
    --best $DEFAULT & wait
done

# for lr in 1e-6 3e-6 5e-6 1e-5 3e-5 5e-5 1e-4
# do

    # CUDA_VISIBLE_DEVICES=0 python3 -W "ignore" main.py --exec_mode train --all-subjects \
    # --model-name img_scratch_lr_${lr}_bs_52_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type emotic --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --data brain3d --with-nsd --cat-only \
    # $DEFAULT & wait
    
    # CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode train --all-subjects \
    # --model-name img_scratch_lr_${lr}_bs_52_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type emotic --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --data brain3d --with-nsd --cat-only \
    # $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode train --all-subjects \
    # --model-name img_scratch_lr_${lr}_bs_52_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type emotic --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --data brain3d --with-nsd --cat-only \
    # $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=3 python3 -W "ignore" main.py --exec_mode train --all-subjects \
    # --model-name img_scratch_lr_${lr}_bs_52_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type emotic --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --data brain3d --with-nsd --cat-only \
    # $DEFAULT & wait

    # CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode predict --all-subjects \
    # --model-name img_scratch_lr_${lr}_bs_52_ep_50 \
    # --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    # --task-type emotic --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    # --brain-backbone ${mlp_ver} --data brain3d --with-nsd --cat-only \
    # --best $DEFAULT & wait
    
# done