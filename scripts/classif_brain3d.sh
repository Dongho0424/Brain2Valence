## all subjects

# # number of classes: 3
# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --all-subjects \
#  --wandb-log --wandb-name 0211_classif3_2 --model-name classif3_resnet18_2 --wandb-project Brain2Valence --wandb-entity donghochoi\
#  --model resnet18 --task-type classif --num-classif 3 --data brain3d --epochs 50 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --all-subjects \
#  --wandb-log --wandb-name 0211_classif3_2 --model-name classif3_resnet18_2 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model resnet18 --task-type classif --num-classif 3 --data brain3d \
#  --best & wait

# # number of classes: 5
# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --all-subjects \
#  --wandb-log --wandb-name 0211_classif5_1 --model-name classif5_resnet18_1 --wandb-project Brain2Valence --wandb-entity donghochoi\
#  --model resnet18 --task-type classif --num-classif 5 --data brain3d --epochs 50 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --all-subjects \
#  --wandb-log --wandb-name 0211_classif5_1 --model-name classif5_resnet18_1 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model resnet18 --task-type classif --num-classif 5 --data brain3d \
#  --best & wait

# 02 23 2024
# 10 classif
# data: brain 3d
# backbone: res 18

CUDA_VISIBLE_DEVICES=0 python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0223_classif10_res18_2 --model-name 0223_classif10_res18_2 --wandb-project Brain2Valence --wandb-entity donghochoi\
 --model resnet18 --data brain3d --epochs 70 --batch-size 32 --lr 1e-5 --weight-decay 0.1 --seed 42 --criterion ce \
 --task-type classif --num-classif 10 --sampler & wait

CUDA_VISIBLE_DEVICES=0 python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0223_classif10_res18_2 --model-name 0223_classif10_res18_2 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet18 --data brain3d \
 --task-type classif --num-classif 10 --sampler --best & wait