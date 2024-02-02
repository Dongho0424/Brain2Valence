# may contain executed date in wandb-name or model-name
CUDA_VISIBLE_DEVICES=3 python3 main.py --exec_mode predict --wandb-name res18_mae_01_predict \
    --model-name all_subjects --wandb-project Brain2Valence --wandb-entity donghochoi