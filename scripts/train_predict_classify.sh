## all subjects

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --all-subjects \
 --wandb-name 0211_classif_3 --model-name classif_3_resnet18 --wandb-project Brain2Valence --wandb-entity donghochoi\
 --model resnet18 --model-type classif --num-classif 3 --epochs 70 --batch-size 12 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --all-subjects \
 --wandb-name 0211_classif_3 --model-name classif_3_resnet18 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet18 --model-type classif & wait
