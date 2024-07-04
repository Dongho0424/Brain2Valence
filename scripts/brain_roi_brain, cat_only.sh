# 2024-05-10
device=0
model_type=brain_only # changed!
subj=1
for lr in 1e-4 
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0510_mlp2_${lr}__subj${subj}_brain_cat_only --model-name 0510_mlp2_${lr}_subj${subj}_brain_cat_only --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrained --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only & wait

# epoch, batch_size for wan`db logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0510_mlp2_${lr}__subj${subj}_brain_cat_only --model-name 0510_mlp2_${lr}_subj${subj}_brain_cat_only --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrained --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only \
 --best & wait

done