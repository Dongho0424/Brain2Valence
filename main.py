import wandb
import argparse
from argparse import ArgumentParser
from trainer import Trainer
from predictor import Predictor
from emotic_trainer import EmoticTrainer
from emotic_predictor import EmoticPredictor
from brain_trainer import BrainTrainer
from brain_predictor import BrainPredictor
from utils import set_seed

def get_args():
    args = ArgumentParser(description="Brain to Valence Decoding")

    # vscode debugging
    args.add_argument('--debug', action='store_true', help='vscode debugging mode')

    # wandb related arguments
    args.add_argument('--wandb-log', action='store_true', help='use wandb')
    # wandb_name: wandb의 id로 쓰이면서, 날짜, 모델, lr, 등등 wandb log에서 구분하기 쉽도록 함.
    # ex) 240203_res18_mae_01_predict
    # args.add_argument('--wandb-name', type=str, default='test', help='name as id, particular wandb run')
    args.add_argument('--wandb-project', type=str, default='Brain2Valence', help='name of wandb project')
    args.add_argument('--wandb-entity', type=str, default='donghochoi', help='name of wandb entity')
    args.add_argument('--group', type=str, required=True, help='group name for wandb run')
    args.add_argument('--notes', type=str, required=True, help='date_#_tag')

    # execute options
    # model_name: model 저장 디렉토리 및 현재 모델의 개괄 설명 간단히
    # kind of all_subjects_res18_mae_01, subject1_res18_mae_01
    args.add_argument('--model-name', type=str, default='',required=True, help='name of model')
    args.add_argument('--task-type', type=str, default="reg", choices=['fMRI_emotion', 'brain', 'emotic', 'img2vad', 'reg', 'classif'], required=True, help='regression for valence(float), multiple classification for valence type')
    args.add_argument('--data', type=str, default="brain3d", choices=['brain3d', 'roi'], required=True, help='data for our task. brain3d: whole brain 3d voxel, roi: well-picked brain 1d array. CAUTION: roi is only with particular subjects.')
    args.add_argument('--all-subjects', action='store_true', default=False, help='train or predict for all subjects')
    args.add_argument('--subj', type=int, default=1, choices=[1,2,5,7], help='train or predict for particular subject number')
    args.add_argument('--exec_mode', type=str, choices=['train', 'predict'], required=True, help='execution mode')
    args.add_argument('--seed', type=int, default=42, help='random seed')

    # for train
    args.add_argument("--criterion", type=str, default="mae", choices=["mse", "mae", "ce", "emotic_L2", "emotic_SL1"], help="Criterion. mse or mae for valence, crossentropy for classification") 
    args.add_argument("--cat-criterion", type=str, default="emotic", choices=["emotic", "softmargin"], help="MultiLabel clf, loss in EMOTIC paper or BCEloss (actually same as MultiLabelSoftMarginLoss)") 
    args.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    args.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    args.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    args.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"], help="Optimizer")
    args.add_argument("--scheduler", type=str, default="cosine", choices=["step", "cosine", "cycle"], help="Scheduler")
    args.add_argument("--save-path", type=str, default="./trained_models", help="Save path")
    args.add_argument("--normalize", action="store_true", help="Normalize", default=False)
    args.add_argument("--n_layers", type=int, default=1, help="Number of layers")
    args.add_argument("--mode", type=str, default="valence", help="Mode", choices=["both", "valence"])
    args.add_argument("--mix_dataset", action="store_true", help="Mix dataset", default=False)
    args.add_argument("--one-point", action="store_true", help="Debugging, only one data point training", default=False)

    # for predict
    args.add_argument("--best", action="store_true", help="Use best model", default=False)
    
    # task type: regression
    reg_args = args.add_argument_group('regression')

    # task type: classification
    classif_args = args.add_argument_group('classification')
    classif_args.add_argument("--num-classif", type=int, default=3, choices=[3, 5, 10], help="Number of classification type. \n3 for 0~4, 4~7, 7~10, 5 for 0~2, 2~4, 4~6, 6~8, 8~10, \n10 for 10 class clf. similar to regression")
    classif_args.add_argument("--sampler", action="store_true", help="Use weighted random sampler", default=False)

    # task type: image to VAD
    img2vad_args = args.add_argument_group('img2vad')
    img2vad_args.add_argument('--pretrained', type=str, default="None", choices=["None", "default", "EMOTIC"], help='default: pretrained by ImageNet + Places365, EMOTIC: pretrained by EMOTIC dataset')  
    img2vad_args.add_argument('--backbone-freeze', action='store_true', default=False, help='Freeze pretrained backbone')
    
    # EMOTIC Reproduce
    emotic_args = args.add_argument_group('emotic')
    emotic_args.add_argument("--model-type", type=str, default="BI", choices=["BI", "B", "I", "brain_only"], help="BI: use both body and image") 
    emotic_args.add_argument("--coco-only", action="store_true", help="Use EMOTIC && COCO dataset", default=False)
    emotic_args.add_argument("--with-nsd", action="store_true", help="Use NSD dataset given subjects", default=False)
    emotic_args.add_argument("--pretraining", action="store_true", help="Pretraining with EMOTIC dataset", default=False)

    # Brain Task
    # use brain3d or roi as guidance to help predicting image => emotic categories
    emotic_args = args.add_argument_group('brain')
    emotic_args.add_argument("--image-backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Image backbone")
    emotic_args.add_argument("--brain-backbone", type=str, default="resnet18", choices=["resnet18", "resnet50", "mlp1", "mlp2"], help="Brain backbone")
    # only predict category. Update model, criterion, training, validation, predict 
    emotic_args.add_argument("--cat-only",  action="store_true", help="predict cat only", default=False)
    emotic_args.add_argument("--fusion-ver", type=int, default=1, choices=[1, 2, 999], help="1: EMOTIC, 2: bn, 999: one_point")


    args = args.parse_args()

    return args

def main(args):

    if args.exec_mode == "train":
        if args.task_type == "emotic":
            trainer = EmoticTrainer(args=args)
            trainer.train()
        elif args.task_type == "brain" or args.task_type == "fMRI_emotion":
            trainer = BrainTrainer(args=args)
            trainer.train()
        else:
            trainer = Trainer(args=args)
            trainer.train()
    elif args.exec_mode == "predict":

        if args.task_type == "emotic":
            predictor = EmoticPredictor(args=args)
            predictor.predict()
        elif args.task_type == "brain" or args.task_type == "fMRI_emotion":
            predictor = BrainPredictor(args=args)
            predictor.predict()
        else:
            predictor = Predictor(args=args)
            predictor.predict()
    else:
        raise NotImplementedError(f'exec_mode {args.exec_mode} is not implemented')
    
if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    main(args)