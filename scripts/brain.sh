device=1
model_type=BI
for lr in 1e-4 5e-4 1e-3 
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0411_imgres18_3dres18_${lr}_${model_type} --model-name 0411_imgres18_3dres18_${lr}_${model_type} --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 50 --batch-size 32 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_L2 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone resnet18 --data brain3d & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0411_imgres18_3dres18_${lr}_${model_type} --model-name 0411_imgres18_3dres18_${lr}_${model_type} --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 50 --batch-size 32 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_L2 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone resnet18 --data brain3d \
 --best & wait

done