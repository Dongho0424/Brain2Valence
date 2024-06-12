# 2024-05-10
device=0
model_type=brain_only # changed!
subj=1
for lr in 3e-3 1e-3 5e-4 1e-4 5e-5
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0510_mlp2_${lr}_subj${subj}_brain_cat_only_fus2 --model-name 0510_mlp2_${lr}_subj${subj}_brain_cat_only_fus2 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 & wait

# epoch, batch_size for wan`db logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0510_mlp2_${lr}_subj${subj}_brain_cat_only_fus2 --model-name 0510_mlp2_${lr}_subj${subj}_brain_cat_only_fus2 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 \
 --best & wait

done