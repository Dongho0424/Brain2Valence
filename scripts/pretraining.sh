# Pretraining image extractor with EMOTIC dataset
# Excluding images shown to subj1
# if --pretrained: default, then pretraining on top of ImageNet & Places365 pretrained weight
# if --pretrained: None, then pretraining from scratch

device=0
model_type=BI
subj=1
for lr in 9e-6
do
    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
    --model-name TEST --notes TEST --group TEST --wandb-project TEST --wandb-entity TEST \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type emotic --pretrained default --pretraining --image-backbone resnet18 --model-type ${model_type} --data roi --cat-only \
    #--wandb-log

    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
    --model-name TEST --notes TEST --group TEST --wandb-project TEST --wandb-entity TEST \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type emotic --pretrained None --pretraining --image-backbone resnet18 --model-type ${model_type} --data roi --cat-only \
    --best #--wandb-log
done