# 0708
# 훈련이 너무 안돼 답답해서, 하나의 데이터를 model이 기억할 수 있는지, test 때 완벽하게 나오는지 test
# 만약 overfitting이 되지 않는다면, test 때 AUPRC가 0.9999가 나오지 않는다면 모델에 문제가 있거나
# 혹은 데이터에 문제가 있음
# 현재 하나의 데이터는 category가 5개짜리, 1개 짜리로도 해보자.
# lr: 1e-4
# epoch: 500 (하나의 사진 300번 보여주는 거니까 얼마 안됨.)
# batch_size: 1
# pretrained: default

# 결과
# AP 결과를 보니까 5개의 cat에 대해 정확히 일치하는 것을 보여줌.
# 근데 AUPRC가 0.2임. 왜? 완벽하게 맞춘 거 아니야?

device=0
model_type=BI # fixed after this time
subj=1
# notes: date_#_tag
# group: required
lr=1e-4
fusion_ver=999
mlp_ver=mlp2
cat_loss=softmargin
CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode train --subj ${subj} --one-point \
    --wandb-log --model-name one_point_${lr} --notes 0708_1 --group one_point --wandb-project fMRI_Emotion --wandb-entity donghochoi \
    --epochs 300 --batch-size 1 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss} \
    --task-type brain --pretrained default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone ${mlp_ver} --data roi --cat-only --fusion-ver ${fusion_ver} & wait

CUDA_VISIBLE_DEVICES=${device} python3 main.py --exec_mode predict --subj ${subj} --one-point \
    --wandb-log --model-name one_point_${lr} --notes 0708_1 --group one_point --wandb-project fMRI_Emotion --wandb-entity donghochoi \
    --epochs 300 --batch-size 1 --lr ${lr} --weight-decay 0.01 --optimizer adamw --scheduler cosine --criterion emotic_SL1 --cat-criterion ${cat_loss} \
    --task-type brain --pretrained default --backbone-freeze --image-backbone resnet18 --model-type ${model_type} --brain-backbone ${mlp_ver} --data roi --cat-only --fusion-ver ${fusion_ver} \
    --best & wait

