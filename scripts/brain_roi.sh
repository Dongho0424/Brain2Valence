device=0
model_type=BI # fixed after this time
subj=1
for lr in 3e-3 1e-3
do

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --wandb-name 0507_mlp_cycle_${lr}_subj${subj}_SL1 --model-name 0507_mlp_cycle_${lr}_subj${subj}_SL1 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cycle --criterion emotic_SL1 \
 --task-type brain --pretrained --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp --data roi & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --wandb-name 0507_mlp_cycle_${lr}_subj${subj}_SL1 --model-name 0507_mlp_cycle_${lr}_subj${subj}_SL1 --wandb-project Emotic --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cycle --criterion emotic_SL1 \
 --task-type brain --pretrained --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp --data roi \
 --best & wait

done