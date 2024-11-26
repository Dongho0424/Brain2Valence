###########################
# Excluding Minor Categories Strategy
###########################

device=3
model_type=BI # fixed after this time
# subj=1
fusion_ver=1 # fixed
mlp_ver=mlp3 # AdaptiveMaxPool1d(h)
cat_loss=softmargin
wd=0.01
pretrained_wgt_path=./trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth
all_subjects="1 2 5 7"

DONGHO="--wandb-project v2_w.o_low_categories --wandb-entity donghochoi --wandb-log --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss}"

# img_only

### scratch ###
### scratch ###

pretrained="default"
# for st in 1 2
# do
#     EXCLUDE="--exclude-low --exclude-strategy ${st}"
#     for lr in 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 8e-6
#     do
#         for subj in 1 2 5 7 "$all_subjects"
#         do
#             if [ "$subj" == "$all_subjects" ]; then
#                 model_name="img_pretrained_lr_${lr}_subj_1257_exclude_low_st${st}"
#             else
#                 model_name="img_pretrained_lr_${lr}_subj_${subj}_exclude_low_st${st}"
#             fi

#             CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj $subj \
#             --model-name $model_name \
#             --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#             --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
#             --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
#             $DONGHO $EXCLUDE

#             CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $subj \
#             --model-name $model_name \
#             --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
#             --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
#             --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
#             --best $DONGHO $EXCLUDE
#         done
#     done
# done

# st 1, lr 3e-5
for st in 1
do
    EXCLUDE="--exclude-low --exclude-strategy ${st}"
    for lr in 3e-5
    do
        for subj in 1 2 5 7 "$all_subjects"
        do
            if [ "$subj" == "$all_subjects" ]; then
                model_name="img_pretrained_lr_${lr}_subj_1257_exclude_low_st${st}_"
            else
                model_name="img_pretrained_lr_${lr}_subj_${subj}_exclude_low_st${st}_"
            fi

            CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj $subj \
            --model-name $model_name \
            --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
            --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
            --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
            $DONGHO $EXCLUDE

            CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $subj \
            --model-name $model_name \
            --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
            --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
            --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
            --best $DONGHO $EXCLUDE
        done
    done
done

# left st 1
for st in 1
do
    EXCLUDE="--exclude-low --exclude-strategy ${st}"
    for lr in 5e-5 1e-4 3e-4 5e-4 8e-6
    do
        for subj in 1 2 5 7 "$all_subjects"
        do
            if [ "$subj" == "$all_subjects" ]; then
                model_name="img_pretrained_lr_${lr}_subj_1257_exclude_low_st${st}"
            else
                model_name="img_pretrained_lr_${lr}_subj_${subj}_exclude_low_st${st}"
            fi

            CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj $subj \
            --model-name $model_name \
            --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
            --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
            --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
            $DONGHO $EXCLUDE

            CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $subj \
            --model-name $model_name \
            --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
            --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
            --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
            --best $DONGHO $EXCLUDE
        done
    done
done

# left st 2

for st in 2
do
    EXCLUDE="--exclude-low --exclude-strategy ${st}"
    for lr in 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 8e-6
    do
        for subj in 1 2 5 7 "$all_subjects"
        do
            if [ "$subj" == "$all_subjects" ]; then
                model_name="img_pretrained_lr_${lr}_subj_1257_exclude_low_st${st}"
            else
                model_name="img_pretrained_lr_${lr}_subj_${subj}_exclude_low_st${st}"
            fi

            CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode train --subj $subj \
            --model-name $model_name \
            --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
            --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
            --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
            $DONGHO $EXCLUDE

            CUDA_VISIBLE_DEVICES=${device} python3 -W "ignore" main.py --exec_mode predict --subj $subj \
            --model-name $model_name \
            --epochs 50 --batch-size 52 --lr ${lr} --weight-decay ${wd} \
            --task-type emotic --pretrained $pretrained --image-backbone resnet18 --model-type ${model_type} \
            --data roi --cat-only --with-nsd --fusion-ver ${fusion_ver} --dataset-ver 2 \
            --best $DONGHO $EXCLUDE
        done
    done
done