subj=1
device=3

# 02 23 2024
# 10 classif
# data: roi
# backbone: MLP

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0223_subject${subj}_classif10_mlp --model-name 0223_subject${subj}_classif10_mlp --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --task-type classif --num-classif 10 --data roi --epochs 70 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0223_subject${subj}_classif10_mlp --model-name 0223_subject${subj}_classif10_mlp --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --task-type classif --num-classif 10 --data roi \
 --best & wait
