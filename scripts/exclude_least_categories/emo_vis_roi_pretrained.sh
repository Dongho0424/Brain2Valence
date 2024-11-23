###########################
# Excluding Minor Categories Strategy
###########################

device=1
model_type=BI # fixed after this time
# subj=1
fusion_ver=1 # fixed
mlp_ver=mlp3 # AdaptiveMaxPool1d(h)
cat_loss=softmargin
wd=0.01
pretrained_wgt_path=./trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth
all_subjects="1 2 5 7"

DONGHO=" --wandb-project v2_w.o_least_categories --wandb-entity donghochoi --wandb-log --dataset-ver 2 --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss}"

## pretrained ##
## pretrained ##

# exclude strategy
# 1: metadata에서 한 row에서 제외하고자 하는 카테고리가 하나라도 있으면 아예 row 삭제
# 2: metadata에서 한 row에서 제외하고자 하는 카테고리만 지우기. 만약 category가 하나도 안남으면 row 삭제
for st in 1 2
do
    EXCLUDE="--exclude-least --exclude-strategy ${st}"
    for lr in 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 8e-6
    do
        for subj in 2 5 7 "$all_subjects" #1
        do
            for pn in 1024 2048 4096
            do
                if [ "$subj" == "$all_subjects" ]; then
                    model_name="emo_vis_roi_pretrained_lr_${lr}_subj_1257_pn_${pn}_exclude_least_st${st}"
                else
                    model_name="emo_vis_roi_pretrained_lr_${lr}_subj_${subj}_pn_${pn}_exclude_least_st${st}"
                fi

                CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj $subj \
                --model-name $model_name \
                --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
                --task-type brain --pretrained default --image-backbone resnet18 --model-type ${model_type} \
                --brain-backbone ${mlp_ver} --pool-num ${pn} --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
                $DONGHO $EXCLUDE

                CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $subj \
                --model-name $model_name \
                --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
                --task-type brain --pretrained default --image-backbone resnet18 --model-type ${model_type} \
                --brain-backbone ${mlp_ver} --pool-num ${pn} --data emo_vis_roi --cat-only --fusion-ver ${fusion_ver} \
                --best $DONGHO $EXCLUDE
            done
        done
    done
done