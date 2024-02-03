# for learning_rate in 1e-4 3e-4 
# do
# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --wandb-name res18_mae_02 --model-name all_subjects --wandb-project Brain2Valence --wandb-entity donghochoi\
#  --epochs 100 --batch-size 24 --lr $learning_rate --weight-decay 0.1 --seed 42 --criterion mae & wait

# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --wandb-name res18_mae_02_predict \
#     --model-name all_subjects --wandb-project Brain2Valence --wandb-entity donghochoi & wait
# done 

## lr: 1e-4

# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --wandb-name res18_mae_2 --model-name all_subjects_res18_mae_2 --wandb-project Brain2Valence --wandb-entity donghochoi\
#  --epochs 100 --batch-size 24 --lr 1e-4 --weight-decay 0.01 --seed 42 --criterion mae & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --wandb-name res18_mae_2_predict \
    --model-name all_subjects_res18_mae_2 --wandb-project Brain2Valence --wandb-entity donghochoi & wait

## lr: 3e-4

# CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --wandb-name res18_mae_3 --model-name all_subjects_res18_mae_3 --wandb-project Brain2Valence --wandb-entity donghochoi\
#  --epochs 100 --batch-size 24 --lr 3e-4 --weight-decay 0.02 --seed 42 --criterion mae & wait

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --wandb-name res18_mae_3_predict \
    --model-name all_subjects_res18_mae_3 --wandb-project Brain2Valence --wandb-entity donghochoi & wait