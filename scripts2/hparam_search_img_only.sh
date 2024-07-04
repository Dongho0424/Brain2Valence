# 여러 h-param 및 layer 조합으로 val set 기준 
# - lr: 1e-4, 3e-4, 1e-3, 3e-3
# - mlp1, mlp2 (only roi)
# - fusion1, fusion2
# 16가지 조합 (8가지)
# - roi, 3dfmri, EMOTIC_only, roi_only
# 4가지 task

# EMOTIC_only

device=2
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
for lr in 3e-3 1e-3 3e-4 1e-4
do
    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
    --wandb-log --model-name img_only_${lr} --notes 0703_3_hparam --group hparam_srch_img_only --wandb-project fMRI_Emotion --wandb-entity donghochoi \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type emotic --pretrained default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd --cat-only & wait

    # epoch, batch_size for wandb logging
    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
    --wandb-log --model-name img_only_${lr} --notes 0703_3_hparam --group hparam_srch_img_only --wandb-project fMRI_Emotion --wandb-entity donghochoi \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type emotic --pretrained default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd --cat-only \
    --best & wait
done

