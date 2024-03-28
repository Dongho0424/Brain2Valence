device=1
lr=1e-4
for model_type in BI B I 
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0328_${model_type}_adamw_SL1_1 --model-name 0328_${model_type}_adamw_SL1_1 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 50 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --model-type ${model_type} & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0328_${model_type}_adamw_SL1_1 --model-name 0328_${model_type}_adamw_SL1_1 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 50 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --model-type ${model_type} \
 --best & wait

done