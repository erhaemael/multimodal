import argparse
import torch
from exp.exp_sup import Exp_All_Task as Exp_All_Task_SUP
import random
import numpy as np
import wandb
from utils.ddp import is_main_process, init_distributed_mode
import os


# Tentukan lokasi penyimpanan di Google Drive
drive_checkpoint_path = "/content/drive/MyDrive/checkpoints"
os.makedirs(drive_checkpoint_path, exist_ok=True)

VERSION = os.getenv("VERSION", "none")
WANDB_USER = os.getenv("WANDB_USER", "none")
BVP_DATASETS = ["wesad"]
FIXED_SEED = 444

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTS supervised training')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                        required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='UniTS',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--dataset_name', type=str, required=True,
                        default='', help='dataset name')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='exp/all_task.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')

    # ddp
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')
    parser.add_argument("--memory_check", action="store_true", default=False)  # ! not used
    parser.add_argument("--large_model", action="store_true", default=True)

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument("--prompt_tune_epoch", type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=1,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='optimizer min learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float,
                        default=None, help='optimizer layer decay')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--lradj', type=str,
                        default='supervised', help='adjust learning rate')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, default=None,
                        help='location of pretrained model checkpoints')
    parser.add_argument('--debug', type=str,
                        default='enabled', help='disabled')
    parser.add_argument('--project_name', type=str,
                        default='tsfm-multitask', help='wandb project name')

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument("--share_embedding",
                        action="store_true", default=False)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--prompt_num", type=int, default=5)

    # task related settings
    # forecasting task
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float,
                        default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float,
                        default=1.0, help='prior anomaly ratio (%)')

    # zero-shot-forecast-new-length
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument('--zero_shot_forecasting_new_length',
                        type=str, default=None, help='unify')

    args = parser.parse_args()
    init_distributed_mode(args)

    print('Args in experiment:')
    print(args)

    win_size = 5
    # For each learning rate
    for lr in [5e-5, 1e-5, 5e-4, 1e-4, 5e-3]:
        # For each fold of the dataset, train and test the model
         for k in range(5):
            # Set seed to ensure reproducibility
            random.seed(FIXED_SEED)
            torch.manual_seed(FIXED_SEED)
            np.random.seed(FIXED_SEED)

            # If the source signal is BVP and the dataset has no BVP signal, skip
            if args.dataset_name not in BVP_DATASETS:
                break

            args.k = k  # Set the fold
            args.learning_rate = lr  # Set the learning rate
            args.source = 'BVP'  # Set the source signal
            task = args.task_name
            args.source = 'BVP'  # Set the source signal
            args.confounding = False
            args.win_size = win_size  # Set the window size
            args.step = 1  # Set the step size

            exp_name = f'{task}-lr{lr}-fold{k}-BVP'

            tags = [task, f"lr{lr}", 'BVP']
            if args.pretrained_weight is not None:
                tags.append('pretrained')
            else:
                tags.append('random')

            if is_main_process():
                wandb.init(
                    name=exp_name,
                    entity=WANDB_USER,  # use personal wandb account
                    # set the wandb project where this run will be logged
                    project=f"{args.project_name}-win{win_size}-{VERSION}",
                    # track hyperparameters and run metadata
                    config=args,
                    mode=args.debug,
                    group=f"{task}_lr{lr}-BVP",
                    tags=tags
                )

            # Cek apakah ada checkpoint yang bisa dimuat
            checkpoint_filename = f"checkpoint_{args.task_name}_lr{lr}_fold{k}_BVP.pth"
            checkpoint_path = os.path.join(args.checkpoints, checkpoint_filename)
            drive_checkpoint_file = os.path.join(drive_checkpoint_path, checkpoint_filename)

            exp = Exp_All_Task_SUP(args)

            # Jika checkpoint ada, load model
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint: {checkpoint_path}")
                exp.model.load_state_dict(torch.load(checkpoint_path))

            if args.is_training:
                for ii in range(args.itr):
                    setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
                        args.task_name,
                        args.model_id,
                        args.model,
                        args.data,
                        args.features,
                        args.d_model,
                        args.e_layers,
                        args.des, ii)

                    print(f'>>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
                    exp.train(setting)

                # Simpan checkpoint setelah training selesai
                torch.save(exp.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

                # Salin checkpoint ke Google Drive
                import shutil
                shutil.copy(checkpoint_path, drive_checkpoint_file)
                print(f"Checkpoint also saved to Google Drive at {drive_checkpoint_file}")
            else:
                ii = 0
                setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.d_model,
                    args.e_layers,
                    args.des, ii)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, load_pretrain=True)
                torch.cuda.empty_cache()

            # Close wandb
            if is_main_process():
                wandb.finish()
