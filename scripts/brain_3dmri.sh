device=0
model_type=BI
subj=1
for lr in 5e-3 3e-3 1e-3 1e-4
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0505_mri_${lr}_${model_type}_subj${subj}_SL1 --model-name 0505_mri_${lr}_${model_type}_subj${subj}_SL1 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 50 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrained --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone resnet18 --data brain3d & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0505_mri_${lr}_${model_type}_subj${subj}_SL1 --model-name 0505_mri_${lr}_${model_type}_subj${subj}_SL1 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 50 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrained --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone resnet18 --data brain3d \
 --best & wait

done