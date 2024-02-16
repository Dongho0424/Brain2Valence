## all subjects

# number of classes: 3
CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0211_classif3_2 --model-name classif3_resnet18_2 --wandb-project Brain2Valence --wandb-entity donghochoi\
 --model resnet18 --task-type classif --num-classif 3 --data brain3d --epochs 50 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0211_classif3_2 --model-name classif3_resnet18_2 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet18 --task-type classif --num-classif 3 --data brain3d \
 --best & wait

# number of classes: 5
CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --all-subjects \
 --wandb-log --wandb-name 0211_classif5_1 --model-name classif5_resnet18_1 --wandb-project Brain2Valence --wandb-entity donghochoi\
 --model resnet18 --task-type classif --num-classif 5 --data brain3d --epochs 50 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --all-subjects \
 --wandb-log --wandb-name 0211_classif5_1 --model-name classif5_resnet18_1 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet18 --task-type classif --num-classif 5 --data brain3d \
 --best & wait
