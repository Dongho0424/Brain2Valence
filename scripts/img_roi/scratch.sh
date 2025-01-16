device=1
model_type=BI # fixed after this time
subj=1
fusion_ver=1 # fixed
mlp_ver=mlp3 # AdaptiveMaxPool1d(h)
cat_loss=softmargin
wd=0.01
pool_num=2024
pretrained_wgt_path=./pretrained_wgts/EMOTIC_pretrained_img_extractor_weight/best_model.pth
project="verify_v1"

# for lr in 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 8e-6 
for lr in 1e-5
do
    # model_name="img_roi_scratch_lr_${lr}"
    model_name="img_roi_scratch_test1"
    group="img_roi_scratch"
    note="250116_1"

    CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
    --model-name $model_name --notes $note --group $group --wandb-project $project --wandb-entity "donghochoi" \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num ${pool_num} --data roi --cat-only --fusion-ver ${fusion_ver} \
    --wandb-log 

    CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
     --model-name $model_name --notes $note --group $group --wandb-project $project --wandb-entity "donghochoi" \
    --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num ${pool_num} --data roi --cat-only --fusion-ver ${fusion_ver} \
    --best --wandb-log 
done

