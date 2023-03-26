import argparse
import os
import shutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

from dataset import rafdb as dataset

from pathlib import Path
from models import emotion_model
from utils import Logger, misc
from tensorboardX import SummaryWriter
from engine import train_supervised, train, validate
import wandb

wandb.init(project='mix-up emotions', entity='2neurons')

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Model & dataset paths
parser.add_argument('--raf_path', type=str, default='./data/RafDB', help='raf_dataset_path')
parser.add_argument('--model_path', type=str, default='./models/resnet18_msceleb.pth', help='pretrained_backbone_path')

# Method options
parser.add_argument('--out', default='result/exp030_mixup_224', help='Directory to output the result')
parser.add_argument('--method', type=str, default='mixup', help='flag for normal/mixup/mixmatch training')
parser.add_argument('--n-labeled', type=int, default=1000, help='Number of labeled data')
parser.add_argument('--train-iteration', type=int, default=1024, help='Number of iteration per epoch')

parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=0.3, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--ema', type=bool, default=True)

args = parser.parse_args()


state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)


def main():
    # Prepare things for logging in weights and biases
    wandb.run.name = args.out.split('/')[-1]
    wandb.config.update(args)
    exp_path = Path(args.out)
    # Create experiment dire
    Path.mkdir(exp_path, exist_ok=False)
    print(f'Training with {args.method} method')
    misc.setup_seed(args.manualSeed)

    # Define transformers to be applied for training/val data
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize,
        T.RandomErasing(scale=(0.02, 0.25))
    ])
    transform_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize
    ])

    # Define loaders for each case, for the supervised model we don't need an unlabeled loader
    if args.method == 'mixmatch':
        train_labeled, train_unlabeled, test = dataset.get_rafdb(root=args.raf_path,
                                                                 training_type='ssl',  # semi-supervised
                                                                 n_labeled=args.n_labeled,
                                                                 transform_train=transforms_train,
                                                                 transform_val=transform_val)
        unlabeled_trainloader = data.DataLoader(train_unlabeled, batch_size=args.batch_size, shuffle=True,
                                                num_workers=4, drop_last=True)
    else:
        train_labeled, test = dataset.get_rafdb(root=args.raf_path,
                                                training_type='normal',
                                                n_labeled=args.n_labeled,
                                                transform_train=transforms_train,
                                                transform_val=transform_val)

    labeled_trainloader = data.DataLoader(train_labeled, batch_size=args.batch_size, shuffle=True,
                                          num_workers=8, drop_last=True)
    test_loader = data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Model creation
    print('Creating ResNet18 model')
    model = emotion_model.Model(args.model_path).cuda()
    # If we use mixmatch EMA (estimated moved average reparametrization is used for the model weights)
    if args.ema:
        ema_model = emotion_model.Model(args.model_path)
        ema_model.cuda()
        for param in ema_model.parameters():
            param.detach_()
    # Find the best algorithm for hardware
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Define the type of loss used
    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    # criterion_focal = FocalLoss()

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # Define scheduler to decrease the lr at certain epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 100], gamma=0.1)

    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay) if args.ema else None
    start_epoch = 0
    best_acc = 0  # best test accuracy

    # Resume
    title = 'Rafdb'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(
            ['Train Loss',  'Test Loss', 'Test Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        if args.method == 'mixmatch':

            train_loss, train_loss_x, train_loss_u = train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer,
                                                           ema_optimizer, train_criterion, epoch, use_cuda)
            print("VAL on train")
            _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
            print('VALIDATION')
            test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')
            step = args.train_iteration * (epoch + 1)
            scheduler.step()
            state['lr'] = (scheduler.get_last_lr())[-1]

        else:
            use_mixup = True if args.method == 'mixup' else False
            print("use mixup:", use_mixup)
            train_loss = train_supervised(args, labeled_trainloader, model, optimizer, criterion, use_mixup, use_cuda)
            print("VAL on train")
            _, train_acc = validate(labeled_trainloader, model, criterion, epoch, use_cuda, mode='Train Stats')
            print('VALIDATION')
            test_loss, test_acc = validate(test_loader, model, criterion, epoch, use_cuda, mode='Test Stats ')
            step = epoch + 1

            scheduler.step(test_acc)
            # state['lr'] = (scheduler.get_last_lr())[-1]
            # # Increase  the value of alpha
            # if epoch in [15, 70, 120]:
            #     new_alpha = args.alpha + 0.1
            #     args.alpha = min(new_alpha, 0.4)
            #     print(f"New alpha: {args.alpha}")

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)

        # append logger file
        logger.append([train_loss, test_loss, test_acc])

        # log to weights and biases
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            'test_acc': test_acc,
            'lr': state['lr']
        })
        wandb.watch(model)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        wandb.run.summary["best_accuracy"] = best_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict() if args.ema else None,
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # Consistency regularization
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

class FocalLoss(object):
    def __call__(self, inputs, targets):
        loss_ce = torchvision.ops.sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction='mean')
        return loss_ce


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


if __name__ == '__main__':
    main()
