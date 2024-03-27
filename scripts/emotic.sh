device=2
for lr in 1e-4 # 1e-3 1e-5 
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0321adamw_L2 --model-name 0321adamw_L2 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 50 --batch-size 26 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_L2 \
 --task-type emotic --pretrain --backbone-freeze & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0321adamw_L2 --model-name 0321adamw_L2 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 50 --batch-size 26 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_L2 \
 --task-type emotic --best & wait

done