subj=2
device=0

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-name 0212_subject${subj}_classif_mlp --model-name subject${subj}_classif_mlp --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --task-type classif --num-classif 3 --data roi --epochs 70 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion ce & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-name 0212_subject${subj}_classif_mlp --model-name subject${subj}_classif_mlp --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --task-type classif --num-classif 3 --data roi & wait
