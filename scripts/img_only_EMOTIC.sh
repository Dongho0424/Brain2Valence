device=0
model_type=BI
subj=1
pretrained_wgt_path=./trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth

for lr in 1e-5 
do
    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
    --model-name TEST --notes TEST --group TEST --wandb-project TEST --wandb-entity TEST \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type emotic --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd --cat-only \
    --wgt-path ${pretrained_wgt_path} #--wandb-log

    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
    --model-name TEST --notes TEST --group TEST --wandb-project TEST --wandb-entity TEST \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 \
    --task-type emotic --pretrained None --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd --cat-only \
    --best --wgt-path ${pretrained_wgt_path} #--wandb-log
done