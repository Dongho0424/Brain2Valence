# 240703
# EMOTIC dataset을 전부 사용하여 pretraining 했던 모델의 weight 사용
# roi + image
# lr 정도만 바꿔서 해보기
# DO NOT backbone freeze
# - lr: 1e-4, 3e-4, 1e-3, 3e-3
# - mlp2
# - fusion1

# 240705
# pretraining: sub1가 본 이미지를 제외한 EMOTIC dataset으로 
# revised BrainModel ( pretrained weight, image, fusion, final 전부 사용하는 모델 )
# roi + image
# lr 바꿔서 해보기
# backbone freeze
# - lr: 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5 1e-6
# - weight decay: 0.01, 0.05 (more decayed for preventing overfitting)
# - mlp2 fixed
# - fusion1 fixed

device=2
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
fusion_ver=1 # fixed
mlp_ver=mlp2
for lr in 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5 1e-6
do
    for wd in 0.01 0.05
    do
        echo "Training Start....: lr: $lr, fusion_ver: $fusion_ver, mlp_ver: $mlp_ver, weight_decay: $wd"

        CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
        --wandb-log --model-name finetuning_roi_img_${lr}_wd${wd} --notes _0705_2 --group finetuning_roi_img --wandb-project fMRI_Emotion --wandb-entity donghochoi \
        --epochs 30 --batch-size 52 --lr ${lr} --weight-decay ${wd} --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
        --task-type brain --pretrained EMOTIC --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone ${mlp_ver} --data roi --cat-only --fusion-ver ${fusion_ver} & wait

        CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
        --wandb-log --model-name finetuning_roi_img_${lr}_wd${wd} --notes _0705_2 --group finetuning_roi_img --wandb-project fMRI_Emotion --wandb-entity donghochoi \
        --epochs 30 --batch-size 52 --lr ${lr} --weight-decay ${wd} --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
        --task-type brain --pretrained EMOTIC --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone ${mlp_ver} --data roi --cat-only --fusion-ver ${fusion_ver} \
        --best & wait
    done
done

