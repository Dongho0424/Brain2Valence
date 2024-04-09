device=3
model_type=BI
# lr=1e-4
for lr in 5e-5 3e-4 1e-3
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0329_${model_type}_SL1_${lr}_1 --model-name 0329_${model_type}_SL1_${lr}_1 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --model-type ${model_type} & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0329_${model_type}_SL1_${lr}_1 --model-name 0329_${model_type}_SL1_${lr}_1 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --model-type ${model_type} \
 --best & wait

done