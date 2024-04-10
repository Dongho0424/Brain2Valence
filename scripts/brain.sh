device=0
model_type=BI
for lr in 1e-4 5e-4 1e-3
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0410_imgres18_3dres18_${lr}_1 --model-name 0410_imgres18_3dres18_${lr}_1 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 50 --batch-size 32 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone resnet18 --data brain3d & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0410_imgres18_3dres18_${lr}_1 --model-name 0410_imgres18_3dres18_${lr}_1 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 50 --batch-size 32 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone resnet18 --data brain3d \
 --best & wait

done