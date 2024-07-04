# 여러 h-param 및 layer 조합으로 val set 기준 
# - lr: 1e-4, 3e-4, 1e-3, 3e-3
# - mlp1, mlp2
# - fusion1, fusion2
# 16가지 조합
# - roi, 3dfmri, EMOTIC_only, roi_only
# 4가지 task

# image + roi

device=0
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
# for lr in 3e-3 1e-3 3e-4 1e-4
for lr in 5e-4 8e-4
do
    for fusion_ver in 1 2
    do 
        for mlp_ver in mlp1 mlp2
        do
        
        echo "Training Start....: lr: $lr, fusion_ver: $fusion_ver, mlp_ver: $mlp_ver"

        CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
        --wandb-log --model-name roi_img_${lr}_${mlp_ver}_fus${fusion_ver} --notes 0704_1_hparam --group hparam_srch_roi --wandb-project fMRI_Emotion --wandb-entity donghochoi \
        --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
        --task-type brain --pretrain default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone ${mlp_ver} --data roi --cat-only --fusion-ver ${fusion_ver} & wait

        CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
        --wandb-log --model-name roi_img_${lr}_${mlp_ver}_fus${fusion_ver} --notes 0704_1_hparam --group hparam_srch_roi --wandb-project fMRI_Emotion --wandb-entity donghochoi \
        --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
        --task-type brain --pretrain default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone ${mlp_ver} --data roi --cat-only --fusion-ver ${fusion_ver} \
        --best & wait
        done
    done
done

