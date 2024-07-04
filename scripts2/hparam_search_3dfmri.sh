# 여러 h-param 및 layer 조합으로 val set 기준 
# - lr: 1e-4, 3e-4, 1e-3, 3e-3
# - mlp1, mlp2 (only roi)
# - fusion1, fusion2
# 16가지 조합 (8가지)
# - roi, 3dfmri, EMOTIC_only, roi_only
# 4가지 task

# 3dfmri + image

device=1
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
for lr in 3e-3 1e-3 3e-4 1e-4
do
    for fusion_ver in 1 2
    do 
        
    echo "Training Start....: lr: $lr, fusion_ver: $fusion_ver"

    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
    --wandb-log --model-name 3dfmri_img_${lr}_fus${fusion_ver} --notes 0703_1_hparam --group hparam_srch_3dfmri --wandb-project fMRI_Emotion --wandb-entity donghochoi \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type brain --pretrain default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone resnet18 --data brain3d --cat-only --fusion-ver ${fusion_ver} & wait

    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
    --wandb-log --model-name 3dfmri_img_${lr}_fus${fusion_ver} --notes 0703_1_hparam --group hparam_srch_3dfmri --wandb-project fMRI_Emotion --wandb-entity donghochoi \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type brain --pretrain default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone resnet18 --data brain3d --cat-only --fusion-ver ${fusion_ver} \
    --best & wait

    done
done

