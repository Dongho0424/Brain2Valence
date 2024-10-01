model_type=BI # fixed after this time
# subj=1
fusion_ver=1 # fixed
mlp_ver=mlp3 # AdaptiveMaxPool1d(h)
cat_loss=softmargin
wd=0.01
pretrained_wgt_path=./trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth
all_subjects="1 2 5 7"

DEFAULT=" --wandb-project Brain2Valence --wandb-entity beotborry --wandb-log --dataset-ver 2 --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss}"
DONGHO=" --wandb-project dataset_v2 --wandb-entity donghochoi --wandb-log --dataset-ver 2 --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss}"

for lr in 5e-6 8e-6 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 
do
    for subj in 1 2 5 7 "$all_subjects"
    do
        if [ "$subj" == "$all_subjects" ]; then
            model_name="img_scratch_lr_${lr}_subj_1257_bs_52_ep_50"
        else
            model_name="img_scratch_lr_${lr}_subj_${subj}_bs_52_ep_50"
        fi

        CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode train --subj $subj \
        --model-name $model_name \
        --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
        --task-type emotic --pretrained None --image-backbone resnet18 --model-type ${model_type} \
        --brain-backbone ${mlp_ver} --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} \
        $DONGHO

        CUDA_VISIBLE_DEVICES=1 python3 -W "ignore" main.py --exec_mode predict --subj $subj \
        --model-name $model_name \
        --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
        --task-type emotic --pretrained None --image-backbone resnet18 --model-type ${model_type} \
        --brain-backbone ${mlp_ver} --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} \
        --best $DONGHO
    done
done