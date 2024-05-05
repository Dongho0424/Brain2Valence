device=0
model_type=BI
subj=2
for lr in 5e-4 1e-4 #1e-3
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0502_${model_type}_SL1_${lr}_subj${subj} --model-name 0502_${model_type}_SL1_${lr}_subj${subj} --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0502_${model_type}_SL1_${lr}_subj${subj} --model-name 0502_${model_type}_SL1_${lr}_subj${subj} --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type emotic --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd \
 --best & wait

done