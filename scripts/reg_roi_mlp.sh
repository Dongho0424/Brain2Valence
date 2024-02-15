subj=1
device=3

# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
#  --wandb-name 0214_subject${subj}_reg_mlp_3 --model-name subject${subj}_reg_mlp_3 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model mlp --task-type reg --data roi --epochs 70 --batch-size 32 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion mae & wait

# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
#  --wandb-name 0214_subject${subj}_reg_mlp_3 --model-name subject${subj}_reg_mlp_3 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model mlp --task-type reg --data roi \
#  --best & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-name 0214_subject${subj}_reg_mlp_3 --model-name subject${subj}_reg_mlp_3 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model mlp --task-type reg --data roi \
 & wait
