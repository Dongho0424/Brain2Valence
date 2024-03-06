subj=1
device=3

# 02 23 2024
# 10 classif
# data: roi
# backbone: MLP
# use weighted sampler

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0223_subject${subj}_classif10_mlp_4 --model-name 0223_subject${subj}_classif10_mlp_4 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --data roi --epochs 70 --batch-size 32 --lr 1e-5 --weight-decay 0.1 --seed 42 --criterion ce \
 --task-type classif --num-classif 10 --sampler & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0223_subject${subj}_classif10_mlp_4 --model-name 0223_subject${subj}_classif10_mlp_4 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --data roi \
 --task-type classif --num-classif 10 --sampler --best & wait
