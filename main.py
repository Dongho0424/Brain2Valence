import wandb
import argparse
from argparse import ArgumentParser
from trainer import Trainer
from predictor import Predictor
from utils import set_seed

def get_args():
    args = ArgumentParser(description="Brain to Valence Decoding")

    # vscode debugging
    args.add_argument('--debug', action='store_true', help='vscode debugging mode')

    # wandb related arguments
    args.add_argument('--wandb-log', action='store_true', help='use wandb')
    # wandb_name: wandb의 id로 쓰이면서, 날짜, 모델, lr, 등등 wandb log에서 구분하기 쉽도록 함.
    # ex) 240203_res18_mae_01_predict
    args.add_argument('--wandb-name', type=str, default='test', help='name as id, particular wandb run')
    args.add_argument('--wandb-project', type=str, default='Brain2Valence', help='name of wandb project')
    args.add_argument('--wandb-entity', type=str, default='donghochoi', help='name of wandb entity')

    # execute options
    # model_name: model 저장 디렉토리 및 현재 모델의 개괄 설명 간단히
    # kind of all_subjects_res18_mae_01, subject1_res18_mae_01
    args.add_argument('--model-name', type=str, default='all_subjects',required=True, help='name of model')
    args.add_argument('--task-type', type=str, default="reg", choices=['reg', 'classif'], required=True, help='regression for valence(float), multiple classification for valence type')
    args.add_argument('--data', type=str, default="brain3d", choices=['brain3d', 'roi'], required=True, help='data for our task. brain3d: whole brain 3d voxel, roi: well-picked brain 1d array. CAUTION: roi is only with particular subjects.')
    args.add_argument('--all-subjects', action='store_true', default=False, help='train or predict for all subjects')
    args.add_argument('--subj', type=int, default=1, choices=[1,2,5,7], help='train or predict for particular subject number')
    args.add_argument('--exec_mode', type=str, choices=['train', 'predict'], required=True, help='execution mode')
    args.add_argument('--seed', type=int, default=42, help='random seed')

    # for train
    args.add_argument("--criterion", type=str, default="mae", choices=["mse", "mae", "ce"], help="Criterion. mse or mae for valence, crossentropy for classification") 
    args.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    args.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    args.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    args.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", 'mlp'], help="Model. CAUTION: mlp is only for roi data")
    args.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    args.add_argument("--optimizer", type=str, default="adamw", help="Optimizer")
    args.add_argument("--scheduler", type=str, default="cosine", help="Scheduler")
    args.add_argument("--save-path", type=str, default="./trained_models", help="Save path")
    args.add_argument("--normalize", action="store_true", help="Normalize", default=False)
    args.add_argument("--n_layers", type=int, default=1, help="Number of layers")
    args.add_argument("--mode", type=str, default="valence", help="Mode", choices=["both", "valence"])
    args.add_argument("--mix_dataset", action="store_true", help="Mix dataset", default=False)

    # for predict
    args.add_argument("--best", action="store_true", help="Use best model", default=False)
    
    # # split arguments with respect to execution mode
    # train_args = args.add_argument_group('train')
    # predict_args = args.add_argument_group('predict')
    
    # # train only arguments
    # train_args.add_argument('')

    # # predict only arguments
    # predict_args.add_argument('')
    reg_args = args.add_argument_group('regression')
    # reg_args.add_argument()

    classif_args = args.add_argument_group('classification')
    classif_args.add_argument("--num-classif", type=int, default=3, choices=[3, 5, 10], help="Number of classification type. \n3 for 0~4, 4~7, 7~10, 5 for 0~2, 2~4, 4~6, 6~8, 8~10, \n10 for 10 class clf. similar to regression")
    classif_args.add_argument("--sampler", action="store_true", help="Use weighted random sampler", default=False)
    
    args = args.parse_args()

    return args

def main(args):

    if args.exec_mode == "train":
        trainer = Trainer(args=args)
        trainer.train()
    elif args.exec_mode == "predict":
        predictor = Predictor(args=args)
        predictor.predict()
    else:
        raise NotImplementedError(f'exec_mode {args.exec_mode} is not implemented')
    
if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    main(args)