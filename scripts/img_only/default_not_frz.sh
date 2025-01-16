device=0
model_type=BI # fixed after this time
subj=1
cat_loss=softmargin
wd=0.01
pretrained_wgt_path=./pretrained_wgts/EMOTIC_pretrained_img_extractor_weight/best_model.pth
project="verify_v1"

for lr in 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 8e-6 
do
    model_name="img_roi_default_not_frz_lr_${lr}"
    group="img_roi_default_not_frz"
    note="250116_1"

    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} \
    --model-name $model_name --notes $note --group $group --wandb-project $project --wandb-entity "donghochoi" \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay $wd --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss} \
    --task-type emotic --pretrained default --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd --cat-only \
    --wandb-log

    CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} \
    --model-name $model_name --notes $note --group $group --wandb-project $project --wandb-entity "donghochoi" \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay $wd --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss} \
    --task-type emotic --pretrained default --image-backbone resnet18 --model-type ${model_type} --data brain3d --with-nsd --cat-only \
    --best --wandb-log
done