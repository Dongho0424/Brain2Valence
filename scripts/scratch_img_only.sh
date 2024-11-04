device=2
model_type=BI # fixed after this time
# subj=1
fusion_ver=1 # fixed
mlp_ver=mlp3 # AdaptiveMaxPool1d(h)
cat_loss=softmargin
wd=0.01
pretrained_wgt_path=./trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth
all_subjects="1 2 5 7"

DONGHO="--wandb-entity donghochoi --wandb-log --wandb-project scratch_v2 --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss}"

# img_only

### scratch ###

## datasetv2 ##
pretrained="None"
# train은 subject가 본 이미지대로 하더라도 test set은 all subject가 본 test set으로
for lr in 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 8e-6
do
    for subj in 1 2 5 7 "$all_subjects"
    do
        if [ "$subj" == "$all_subjects" ]; then
            model_name="img_scratch_lr_${lr}_subj_1257_v2"
        else
            model_name="img_scratch_lr_${lr}_subj_${subj}_v2"
        fi

        CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj $subj \
        --model-name $model_name \
        --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
        --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
        --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
        $DONGHO

        # CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $all_subjects \
        CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $subj \
        --model-name $model_name \
        --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
        --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
        --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
        --best $DONGHO
    done
done

## datasetv1 ##
# train은 subject가 본 이미지대로 하더라도 test set은 all subject가 본 test set으로
for lr in 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 8e-6
do
    for subj in 1 2 5 7 "$all_subjects"
    do
        if [ "$subj" == "$all_subjects" ]; then
            model_name="img_scratch_lr_${lr}_subj_1257_v1"
        else
            model_name="img_scratch_lr_${lr}_subj_${subj}_v1"
        fi

        CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj $subj \
        --model-name $model_name \
        --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
        --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
        --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 1 \
        $DONGHO

        # CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $all_subjects \
        CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $subj \
        --model-name $model_name \
        --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
        --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
        --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 1 \
        --best $DONGHO
    done
done