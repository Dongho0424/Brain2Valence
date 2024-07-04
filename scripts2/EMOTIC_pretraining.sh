# 240702
# EMOTIC dataset을 전부 사용하여 pretraining 했던 모델의 weight 사용
# body model, context model의 weight을 사용하고, roi backbone 및 fusion network은 scratch 부터 학습
# pretrained weight의 성능을 확인하는 것이 목적. roi, mlp2, cat_only, fusion 2 는 유지
# 왜 Pretrained weight을 가지고 실험하는데 mAP 값이 더 낮지?
# 오늘 EMOTIC whole dataset으로 돌린 weight 끌고 와서 실험.

device=0
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
for lr in 3e-3 1e-3
do
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --model-name Emotic_Pretrained_for_compare_${lr} --notes 0702_3_default_pretrain --group emotic_pretraining --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrain default --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --model-name Emotic_Pretrained_for_compare_${lr} --notes 0702_3_default_pretrain --group emotic_pretraining --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrain default --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 \
 --best & wait

done

# # train using whole EMOTIC dataset
# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train \
#  --wandb-log --model-name train_whole_emotic_dataset --notes _0702_1 --group emotic_pretraining --wandb-project fMRI_Emotion --wandb-entity donghochoi \
#  --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
#  --task-type emotic --pretrain default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --data brain3d --cat-only & wait

# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict \
#  --wandb-log --model-name train_whole_emotic_dataset --notes _0702_1 --group emotic_pretraining --wandb-project fMRI_Emotion --wandb-entity donghochoi \
#  --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
#  --task-type emotic --pretrain default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --data brain3d --cat-only \
#  --best & wait
