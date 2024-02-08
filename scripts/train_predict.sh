# for learning_rate in 1e-4 3e-4 
# do
# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --wandb-name res18_mae_02 --model-name all_subjects --wandb-project Brain2Valence --wandb-entity donghochoi\
#  --epochs 100 --batch-size 24 --lr $learning_rate --weight-decay 0.1 --seed 42 --criterion mae & wait

# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --wandb-name res18_mae_02_predict \
#     --model-name all_subjects --wandb-project Brain2Valence --wandb-entity donghochoi & wait
# done 

## all subjects

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --all-subjects \
 --wandb-name 0204_resnet50_mae --model-name all_subjects_resnet50_mae --wandb-project Brain2Valence --wandb-entity donghochoi\
 --model resnet50 --epochs 100 --batch-size 12 --lr 1e-4 --weight-decay 0.001 --seed 42 --criterion mae & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --all-subjects \
 --model resnet50 --wandb-name 0204_resnet50_mae --model-name all_subjects_resnet50_mae --wandb-project Brain2Valence --wandb-entity donghochoi & wait

# for each subject

for subj in 1 2 5 7
do
CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --subj ${subj} \
 --wandb-name 0204_subject${subj}_res18_mse --model-name subject${subj}_res18_mse --wandb-project Brain2Valence --wandb-entity donghochoi\
 --epochs 70 --batch-size 16 --lr 1e-4 --weight-decay 0.1 --seed 42 --criterion mse & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-name 0204_subject${subj}_res18_mse --model-name subject${subj}_res18_mse --wandb-project Brain2Valence --wandb-entity donghochoi & wait
done