device=0
model_type=BI # fixed after this time
subj=1
fusion_ver=1 # fixed
mlp_ver=mlp3 # AdaptiveMaxPool1d(h)
cat_loss=softmargin
wd=0.01
pool_num=2048
pretrained_wgt_path=./trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth
for lr in 1e-5 
do
    CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
    --model-name TEST --notes TEST --group TEST --wandb-project TEST --wandb-entity TEST \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay ${wd} --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss} \
    --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num ${pool_num} --data roi --cat-only --fusion-ver ${fusion_ver} \
    --wgt-path ${pretrained_wgt_path} #--wandb-log 

    CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj ${subj} \
    --model-name TEST --notes TEST --group TEST --wandb-project TEST --wandb-entity TEST \
    --epochs 30 --batch-size 52 --lr ${lr} --weight-decay ${wd} --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss} \
    --task-type brain --pretrained None --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num ${pool_num} --data roi --cat-only --fusion-ver ${fusion_ver} \
    --best #--wandb-log 
done