### resnet50

# device=0
# for lr in 1e-5 1e-4 1e-3
# do

# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
#  --wandb-log --wandb-name 0316_lr_${lr}_wd_1-e2 --model-name 0316_lr_${lr}_wd_1-e2 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model resnet50 --data brain3d --epochs 70 --batch-size 32 --lr ${lr} --weight-decay 1e-2 --seed 42 --criterion mse \
#  --task-type img2vad --pretrain & wait

# # epoch, batch_size for wandb logging
# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
#  --wandb-log --wandb-name 0316_lr_${lr}_wd_1-e2 --model-name 0316_lr_${lr}_wd_1-e2 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model resnet50 --data brain3d --epochs 70 --batch-size 32 --lr ${lr} --weight-decay 1e-2 --seed 42 --criterion mse \
#  --task-type img2vad --best & wait

# done

## using Adam and Step scheduler
# backbone freeze

device=0
for lr in 1e-5 1e-4 1e-3
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0318adam_lr_${lr} --model-name 0318adam_lr_${lr} --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet50 --data brain3d --epochs 70 --batch-size 32 --lr ${lr} --weight-decay 5e-4 --optimizer adam --scheduler step --criterion mse \
 --task-type img2vad --pretrain & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0318adam_lr_${lr} --model-name 0318adam_lr_${lr} --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet50 --data brain3d --epochs 70 --batch-size 32 --lr ${lr} --weight-decay 5e-4 --optimizer adam --scheduler step --criterion mse \
 --task-type img2vad --best & wait

done