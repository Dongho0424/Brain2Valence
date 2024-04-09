device=0
model_type=BI
# lr=1e-4
for lr in 1e-4 5e-4 1e-3
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0410_coco-only_${lr}_1 --model-name 0410_coco-only_${lr}_1 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --model-type ${model_type} --coco-only & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0410_coco-only_${lr}_1 --model-name 0410_coco-only_${lr}_1 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --model-type ${model_type} --coco-only \
 --best & wait

done