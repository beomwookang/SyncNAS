import argparse
import os
import shutil
import time
import torch, torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ptflops import get_model_complexity_info
import warnings

from torch_modules import TorchBranchedModel

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Evaluating for ImageNet-1000')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--base_model', default='mobilenet_v2', type=str,
                    choices=['mobilenet_v2', 'mnasnet_b1', 'fbnet_c'],
                    help='target base model to be adapted (default: mobilnet_v2)')
parser.add_argument('--path', type=str, metavar='PATH', help='path to ImageNet dataset')

best_acc1 = 0
best_acc5 = 0


def main():
    global args, best_acc1, best_acc5
    args = parser.parse_args()

    valdir = os.path.join(args.path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    numberofclass = 1000

    #Load PyTorch Model
    assert args.base_model in ['mobilenet_v2', 'mnasnet_b1', 'fbnet_c']
    print("\nBase Model: %s" %args.base_model)
    model_name = "syncnas_"+args.base_model+"_100"
    model_config = "model_configs/"+model_name+".json"
    model = TorchBranchedModel(model_config)


    #Count Model Complexity
    macs, params = get_model_complexity_info(model, (3,224,224), as_strings=True, print_per_layer_stat=False, verbose=False)
    print("\nLoaded Model: %s" %(model_name))
    print("MACs: %s, Params: %s" %(macs, params))


    #Load Params by API Call
    pretrained = "pretrained/"+model_name+".pth"
    model.load_state_dict(torch.load(pretrained))
    model = torch.nn.DataParallel(model).cuda()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # evaluate on validation set
    print("\nValidating...")
    acc1, acc5, val_loss = validate(val_loader, model, criterion)

    print('Accuracy top-1: ', acc1, '\t\tAccuracy top-5: ', acc5)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test (on val set): [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
              'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses,
            top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# print accruacy
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
