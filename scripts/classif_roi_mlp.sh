subj=1
device=3

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0214_subject${subj}_classif_mlp_2 --model-name subject${subj}_classif_mlp_2 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --task-type classif --num-classif 3 --data roi --epochs 70 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0214_subject${subj}_classif_mlp_2 --model-name subject${subj}_classif_mlp_2 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --task-type classif --num-classif 3 --data roi \
 --best & wait
