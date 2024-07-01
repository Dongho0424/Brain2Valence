device=0
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
for lr in 2e-3
do
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --model-name wandb_group_test_5 --notes 0701_5_test --group wandb_group_test --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --model-name wandb_group_test_5 --notes 0701_5_test --group wandb_group_test --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrain --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 \
 --best & wait

done