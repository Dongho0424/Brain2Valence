## all subjects

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --all-subjects \
 --wandb-name 0211_classif3_1 --model-name classif3_resnet18_1 --wandb-project Brain2Valence --wandb-entity donghochoi\
 --model resnet18 --task-type classif --num-classif 3 --epochs 50 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --all-subjects \
 --wandb-name 0211_classif3_1 --model-name classif3_resnet18_1 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet18 --task-type classif --num-classif 3 & wait
