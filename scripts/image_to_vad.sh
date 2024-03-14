### resnet50

device=0

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0314_img2vad_3_res50 --model-name 0314_img2vad_3_res50 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet50 --data brain3d --epochs 70 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion mse \
 --task-type img2vad --pretrain & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0314_img2vad_3_res50 --model-name 0314_img2vad_3_res50 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet50 --data brain3d \
 --task-type img2vad --best & wait

### resnet18

# device=1

# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --all-subjects \
#  --wandb-log --wandb-name 0314_img2vad_2_res18 --model-name 0314_img2vad_2_res18 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model resnet18 --data brain3d --epochs 70 --batch-size 52 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion mse \
#  --task-type img2vad --pretrain & wait

# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --all-subjects \
#  --wandb-log --wandb-name 0314_img2vad_2_res18 --model-name 0314_img2vad_2_res18 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model resnet18 --data brain3d \
#  --task-type img2vad --best & wait