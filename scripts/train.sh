# for learning_rate in 1e-4
# do
#     CUDA_VISIBLE_DEVICES=1 python3 iaps_train.py --epochs 100 --batch-size 32 --lr $learning_rate --weight-decay 0.0001 --seed 0 --wandb-run-name clip_lr_search_mae_normalize --criterion mae --normalize --model ViT-L/14 --test_split
# done

CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode train --wandb-name res18_mae_02 --model-name all_subjects --wandb-project Brain2Valence --wandb-entity donghochoi\
 --epochs 100 --batch-size 16 --lr 1e-4 --weight-decay 0.1 --seed 42 --criterion mae 
