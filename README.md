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
### pretrained weight (by EMOTIC)
- `trained_models/EMOTIC_pretrained_img_extractor_weight/best_model.pth`
### example
- image + ROI model
- pretrained by EMOTIC dataset
- adapt AdaptiveMaxPooling(h) to fMRI data
```
bash ./scripts/img_roi.sh
```
