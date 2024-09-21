![Model Structure](/model_sturcture.png)

## task 별 실행 명령어 argument 정리
### task
> task: img(subj1) + ROI => 26 emotion category classification
### pretrained list
- image feature extractor (resnet18)
    - None
    - ImageNet & Places365
    - EMOTIC dataset
### 명령어 및 조합 정리
- `--exec_mode`: train or predict
- `--pretrained`: image feature extractor에 pretrained data 종류
    - `None`: from scratch
    - `default`: ImageNet & Places365
    - `EMOTIC`: pretrained weight by EMOTIC
- `--backbone-freeze`: 학습 도중 pretrained 된 backbone freeze 여부
- `--image-backbone`: image 학습하는 backbone. 
    - resnet18만 가능하게 해놨음
- `--brain-backbone`: brain data 학습하는 backbone. 
    - mlp3: AdaptiveMaxPool(h) 사용 `pool-num` 으로 h 조절.
- `--cat-only`: 최종 예측하는 task를 26 emotion categories의 classification로 한정. 
- `--wandb-log`: wandb에 업로드. 
    - `--notes`, `--group`, `--wandb-project`, `--wandb-entity` setting required.
### pretrained weight
- Imagenet: by torchvision package. nothing to do.
- Places365: `data/places/resnet18_state_dict.pth`
- EMOTIC: `trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth`
## ROI + img model
### example 1
- training from **scratch**
- **image + ROI model**
- adapt AdaptiveMaxPooling(h) to fMRI data
```
bash ./scripts/img_roi_None.sh
```
### example 2
- pretrained by **ImageNet and Places365** dataset
    - First introduced by EMOTIC paper.
- **image + ROI model**
- adapt AdaptiveMaxPooling(h) to fMRI data
```
bash ./scripts/img_roi_default.sh
```
### example 3
- pretrained by **EMOTIC** dataset
- **image + ROI model**
- adapt AdaptiveMaxPooling(h) to fMRI data
```
bash ./scripts/img_roi_EMOTIC.sh
```
## img only model
### example 4
- pretrained by **EMOTIC** dataset
- **image Only model**
```
bash ./scripts/img_only_EMOTIC.sh
```

## pretraining img extractor by EMOTIC dataset
### example 5
- pretraining img extractor by EMOTIC dataset
- *excluding images shown to subj1*
- `--pretrained: default`, then pretraining on top of ImageNet & Places365 pretrained weight
- `--pretrained: None`, then pretraining from scratch
- The combination of previously pretrained weight is as follows.
    1. lr: 9e-6
    2. `--pretrained: default`
```
bash ./scripts/pretraining.sh
```