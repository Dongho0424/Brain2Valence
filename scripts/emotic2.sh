device=1

## Training

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0321adam_SL1 --model-name 0321adam_SL1 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 50 --batch-size 26 --lr 1e-4 --weight-decay 5e-4 --optimizer adam --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze & wait

## Prediction

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0321adam_SL1 --model-name 0321adam_SL1 --wandb-project Emotic --wandb-entity donghochoi \
 --model resnet18 --data brain3d --epochs 50 --batch-size 26 --lr 1e-4 --weight-decay 5e-4 --optimizer adam --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --best & wait

