# 240702
# EMOTIC dataset을 전부 사용하여 pretraining
# 난 병신이다. backbone freeze하고 pretraining 하면 어떡함?
# 당연히 backbone freeze 없이 학습  

device=0
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
for lr in 3e-3
do

# # train using whole EMOTIC dataset
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train \
 --wandb-log --model-name pretrain_whole_emotic_dataset --notes _0702_not_freeze --group emotic_pretraining --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain default --image-backbone resnet18 --model-type ${model_type} --data brain3d --cat-only & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict \
 --wandb-log --model-name pretrain_whole_emotic_dataset --notes _0702_not_freeze --group emotic_pretraining --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain default --image-backbone resnet18 --model-type ${model_type} --data brain3d --cat-only \
 --best & wait

done
