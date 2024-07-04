device=0
model_type=BI # fixed after this time
subj=1
for lr in 3e-3 # 1e-3 5e-4
do

# CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
#  --wandb-log --wandb-name 0510_for_inference --model-name 0509_mlp2_${lr}_subj${subj}_cat_only_fus2 --wandb-project Emotic --wandb-entity donghochoi \
#  --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
#  --task-type brain --pretrained --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 & wait

# epoch, batch_size for wan`db logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0510_for_inference_1 --model-name 0509_mlp2_${lr}_subj${subj}_cat_only_fus2 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrained --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 \
 --best & wait

done