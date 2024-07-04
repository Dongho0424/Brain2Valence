# 240702
# 말그대로 val dataset으로 epoch마다 ap mean 구하는 것으로 코드 바꿨는데 이를 테스트

device=0
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
for lr in 3e-3
do
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
 --wandb-log --model-name val_apmean_test --notes 0703_1 --group val_apmean_test --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrained default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 & wait

# epoch, batch_size for wandb logging
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
 --wandb-log --model-name val_apmean_test --notes 0703_1 --group val_apmean_test --wandb-project fMRI_Emotion --wandb-entity donghochoi \
 --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
 --task-type brain --pretrained default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone mlp2 --data roi --cat-only --fusion-ver 2 \
 --best & wait

done