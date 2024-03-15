### resnet50

device=0
for lr in 5e-4 1e-3 1e-2
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0315_img2vad_lr_${lr} --model-name 0315_img2vad_lr_${lr} --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet50 --data brain3d --epochs 70 --batch-size 32 --lr ${lr} --weight-decay 5e-4 --seed 42 --criterion mse \
 --task-type img2vad --pretrain & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0315_img2vad_lr_${lr} --model-name 0315_img2vad_lr_${lr} --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet50 --data brain3d \
 --task-type img2vad --best & wait

done