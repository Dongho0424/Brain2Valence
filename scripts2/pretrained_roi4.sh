# 240703
# EMOTIC dataset을 전부 사용하여 pretraining 했던 모델의 weight 사용
# roi + image
# lr 정도만 바꿔서 해보기
# DO NOT backbone freeze
# - lr: 1e-4, 3e-4, 1e-3, 3e-3
# - mlp2
# - fusion1

device=0
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
fusion_ver=2
mlp_ver=mlp2
epoch=50
batch_size=26
for lr in 3e-3 1e-3 5e-4 3e-4 1e-4
do
    
    echo "Training Start....: lr: $lr, fusion_ver: $fusion_ver, mlp_ver: $mlp_ver"

    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
    --wandb-log --model-name pretrained_roi_img_${lr}_${mlp_ver}_fus${fusion_ver}_epoch${epoch}_batch${batch_size} --notes 0704_1_not_freeze --group pretrained_roi_not_freeze --wandb-project fMRI_Emotion --wandb-entity donghochoi \
    --epochs ${epoch} --batch-size ${batch_size} --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} --brain-backbone ${mlp_ver} --data roi --cat-only --fusion-ver ${fusion_ver} & wait

    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
    --wandb-log --model-name pretrained_roi_img_${lr}_${mlp_ver}_fus${fusion_ver}_epoch${epoch}_batch${batch_size} --notes 0704_1_not_freeze --group pretrained_roi_not_freeze --wandb-project fMRI_Emotion --wandb-entity donghochoi \
    --epochs ${epoch} --batch-size ${batch_size} --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} --brain-backbone ${mlp_ver} --data roi --cat-only --fusion-ver ${fusion_ver} \
    --best & wait

done

