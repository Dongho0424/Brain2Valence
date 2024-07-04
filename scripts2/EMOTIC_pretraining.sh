# 240702
# EMOTIC dataset을 전부 사용하여 pretraining 했던 모델의 weight 사용
# body model, context model의 weight을 사용하고, roi backbone 및 fusion network은 scratch 부터 학습
# pretrained weight의 성능을 확인하는 것이 목적. roi, mlp2, cat_only, fusion 2 는 유지
# 왜 Pretrained weight을 가지고 실험하는데 mAP 값이 더 낮지?
# 오늘 EMOTIC whole dataset으로 돌린 weight 끌고 와서 실험.

# 240705
# EMOTIC dataset 중 subj1가 본 이미지를 제외한 것으로 pretraining 하기

device=0
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
for lr in 3e-3 1e-3 3e-4 1e-4 1e-5
do
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --model-name Pretrainig_${lr} --notes 0705_1_excluding_intersec --group emotic_pretraining --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrained default --pretraining --image-backbone resnet18 --model-type ${model_type} --data roi --cat-only & wait


# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --model-name Pretrainig_${lr} --notes 0705_1_excluding_intersec --group emotic_pretraining --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrained default --pretraining --image-backbone resnet18 --model-type ${model_type} --data roi --cat-only & wait
 --best & wait

done
