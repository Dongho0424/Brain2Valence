device=1
model_type=BI # fixed after this time
subj=1
for lr in 5e-3 3e-3 1e-3 1e-4
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0505_roi_mlp2_${lr}_subj${subj}_L2 --model-name 0505_roi_mlp2_${lr}_subj${subj}_L2 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 50 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_L2 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0505_roi_22_${lr}_subj${subj}_L2 --model-name 0505_roi_22_${lr}_subj${subj}_L2 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 50 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_L2 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi \
 --best & wait

done