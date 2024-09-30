model_type=BI # fixed after this time
subj=1
fusion_ver=1 # fixed
mlp_ver=mlp3 # AdaptiveMaxPool1d(h)
cat_loss=softmargin
wd=0.01
pretrained_wgt_path=./trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth

DEFAULT=" --wandb-project Brain2Valence --wandb-entity beotborry --wandb-log --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss}"

for lr in 1e-5
do
   CUDA_VISIBLE_DEVICES=2 python3 -W "ignore" main.py --exec_mode train --subj ${subj} \
    --model-name emotic_pretrained_vis_roi_finetune_lr_${lr}_bs_52_pn_2048_ep_100 \
    --epochs 100 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
    --task-type brain --pretrained EMOTIC --image-backbone resnet18 --model-type ${model_type} \
    --brain-backbone ${mlp_ver} --pool-num 2048 --data roi --cat-only --fusion-ver ${fusion_ver} \
    --wgt-path ${pretrained_wgt_path} $DEFAULT & wait
done
