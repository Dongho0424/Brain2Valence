device=0
model_type=BI
subj=1
for lr in 3e-3 1e-3 5e-4 1e-4 
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj}\
 --wandb-log --wandb-name 0507_EMOTIC_${lr}_cat_only_subj${subj} --model-name 0507_EMOTIC_${lr}_cat_only_subj${subj} --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd --cat-only & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj}\
 --wandb-log --wandb-name 0507_EMOTIC_${lr}_cat_only_subj${subj} --model-name 0507_EMOTIC_${lr}_cat_only_subj${subj} --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd --cat-only \
 --best & wait

done