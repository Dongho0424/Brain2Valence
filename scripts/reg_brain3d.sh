## all subjects

# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --all-subjects \
#  --wandb-log --wandb-name 0215_resnet18_mae_1 --model-name all_subjects_resnet18_mae_1 --wandb-project Brain2Valence --wandb-entity donghochoi\
#  --model resnet18 --task-type reg --data brain3d --epochs 100 --batch-size 32 --lr 1e-5 --weight-decay 0.001 --seed 42 --criterion mae & wait

# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --all-subjects \
#  --wandb-log --wandb-name 0215_resnet18_mae_1 --model-name all_subjects_resnet18_mae_1 --wandb-project Brain2Valence --wandb-entity donghochoi \
#  --model resnet18 --task-type reg --data brain3d \
#  --best & wait


# for each subject

for subj in 1 #2 5 7
do
CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0216_subject${subj}_res18_2 --model-name subject${subj}_res18_2 --wandb-project Brain2Valence --wandb-entity donghochoi\
 --model resnet18 --task-type reg --data brain3d --epochs 100 --batch-size 32 --lr 1e-5 --weight-decay 0.001 --seed 42 --criterion mae & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0216_subject${subj}_res18_2 --model-name subject${subj}_res18_2 --wandb-project Brain2Valence --wandb-entity donghochoi \
 --model resnet18 --task-type reg --data brain3d \
 --best & wait
done