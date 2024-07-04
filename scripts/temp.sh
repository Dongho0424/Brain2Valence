CUDA_VISIBLE_DEVICES=0 python3 main.py --exec_mode predict \
 --wandb-log --wandb-name 0510_for_inference_1 --model-name 0329_BI_SL1_1e-3_1 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr 1e-3 --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrained --backbone-freeze --image-backbone resnet18 --model-type BI --data brain3d \
 --best