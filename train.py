"""
Train script for CheXNet
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import pprint
import warnings
import json
from torch.optim import lr_scheduler
from torch.autograd import Variable
from densenet import densenet121

import util
import shutil

class DenseNet(nn.Module):
    def __init__(self, config, nclasses):
        super(DenseNet, self).__init__()
        self.model_ft = densenet121(pretrained=not config.scratch, drop_rate=config.drop_rate)
        num_ftrs = self.model_ft.classifier.in_features
        self.model_ft.classifier = nn.Linear(num_ftrs, nclasses)
        self.config = config
    def forward(self, x):
        return self.model_ft(x)

def transform_data(data, use_gpu, train=False):
    inputs, labels = data
    labels = labels.type(torch.FloatTensor)
    if use_gpu is True:
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs = Variable(inputs, requires_grad=False, volatile=not train)
    labels = Variable(labels, requires_grad=False, volatile=not train)
    return inputs, labels


def train_epoch(epoch, args, model, loader, criterion, optimizer):
    model.train()
    batch_losses = []
    for batch_idx, data in enumerate(loader):
      inputs, labels = transform_data(data, True, train=True)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels, epoch=epoch)
      loss.backward()
      optimizer.step()
      print("Epoch: {:d} Batch: {:d} ({:d}) Train Loss: {:.6f}".format(
          epoch, batch_idx, args.batch_size, loss.data))
      sys.stdout.flush()
      batch_losses.append(loss.data)
    train_loss = torch.mean(torch.stack(batch_losses))
    print("Training Loss: {:.6f}".format(train_loss))
    return train_loss


def test_epoch(model, loader, criterion, epoch=1):
    """
    Returns: (AUC, ROC AUC, F1, validation loss)
    """
    model.eval()
    test_losses = []
    outs = []
    gts = []
    for data in loader:
        for gt in data[1].numpy().tolist():
            gts.append(gt)
        inputs, labels = transform_data(data, True, train=False)
        outputs = model(inputs)
        loss = criterion(outputs, labels, epoch=epoch)
        test_losses.append(loss.data)
        out = torch.sigmoid(outputs).data.cpu().numpy()
        outs.extend(out)
    avg_loss = torch.mean(torch.stack(test_losses))
    print("Validation Loss: {:.6f}".format(avg_loss))
    outs = np.array(outs)
    gts = np.array(gts)
    return util.evaluate(gts, outs, loader.dataset.pathologies) + (avg_loss,)


def get_loss(dataset, weighted):

    criterion = nn.MultiLabelSoftMarginLoss()

    def loss(preds, target, epoch):

        if weighted:

            return dataset.weighted_loss(preds, target, epoch=epoch)

        else:

            return criterion(preds, target)

    return loss


def saveCheckpoint(model, best_model_wts, train_loss, val_loss, best_loss, optimizer, epoch):
  print('Saving checkpoint...')

  state = {
    'model' : model,
    'best_model_wts': best_model_wts,
    'train_loss' : train_loss,
    'val_loss' : val_loss,
    'best_loss': best_loss,
    'optimizer' : optimizer,
    'epoch' : epoch
  }

  if not os.path.exists('/content/GP/checkpoints'):
    os.makedirs('/content/GP/checkpoints')

  torch.save(state, '/content/GP/checkpoints/checkpoint_val%f_train%f_epoch%d' % (val_loss, train_loss, epoch))
  # shutil.copy('/content/GP/checkpoints/{}'.format("checkpoint_val%f_train%f_epoch%d" % (val_loss, train_loss, epoch)), '/content/drive/MyDrive/checkpoints')
  shutil.copy('/content/GP/checkpoints/{}'.format("checkpoint_val%f_train%f_epoch%d" % (val_loss, train_loss, epoch)), '/content/drive/MyDrive/Chest_X-Ray_GP/Experiences/Experience_2/Checkpoints')

  print('checkpoint saved successfully to directory {checkpoints}!!')


def loadCheckpoint():
  checkPoints = os.listdir('/content/GP/checkpoints')
  tmpMax = -1
  for chkpnt in checkPoints:
    if int(chkpnt[-1])>=tmpMax:
      tmpMax = int(chkpnt[-1])
      lastCheckpoint = chkpnt
  
  loaded_chkpnt = torch.load('/content/GP/checkpoints/{}'.format(lastCheckpoint))
  return loaded_chkpnt



def run(args):

    use_gpu = torch.cuda.is_available()
    model = None

    train, val = util.load_data(args)
    nclasses = train.dataset.n_classes
    print("Number of classes:", nclasses)

    if args.model == "densenet":
        model = DenseNet(args, nclasses)
    else:
        print("{} is not a valid model.".format(args.model))

    if use_gpu:
        model = model.cuda()

    train_criterion = get_loss(train.dataset, args.train_weighted)
    
    val_criterion = get_loss(val.dataset, args.valid_weighted)

    if args.optimizer == "adam":
        optimizer = optim.Adam(
                       filter(lambda p: p.requires_grad, model.model_ft.parameters()),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
                       filter(lambda p: p.requires_grad, model.model_ft.parameters()),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
    else:
        print("{} is not a valid optimizer.".format(args.optimizer))


    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.001, factor=0.1)

    epochToStartFrom = 1
    if os.path.exists('/content/GP/checkpoints'):
      if(len(os.listdir('/content/GP/checkpoints'))) != 0:
        loaded_chkpnt = loadCheckpoint()
        model, best_model_wts, train_loss, val_loss, best_loss, optimizer, epochToStartFrom = loaded_chkpnt['model'], loaded_chkpnt['best_model_wts'], loaded_chkpnt['train_loss'], loaded_chkpnt['val_loss'], loaded_chkpnt['best_loss'], loaded_chkpnt['optimizer'], loaded_chkpnt['epoch']+1

    else:
      best_model_wts, best_loss = model.state_dict(), float("inf")

    counter = 0
    for epoch in range(epochToStartFrom, args.epochs + 1):
        print("Epoch {}/{}".format(epoch, args.epochs))
        print("-" * 10)
        train_loss = train_epoch(epoch, args, model, train,train_criterion, optimizer)
        _, epoch_auc, _, valid_loss = test_epoch(model, val, val_criterion, epoch)
        scheduler.step(valid_loss)

        if (valid_loss < best_loss):
            best_loss = valid_loss
            best_model_wts = model.state_dict()
            counter = 0        
        else:
            counter += 1

        if counter > 3:
            break

        torch.save(best_model_wts, os.path.join(args.save_path, "val%f_train%f_epoch%d" % (valid_loss, train_loss, epoch)))
        saveCheckpoint(model, best_model_wts, train_loss, valid_loss, best_loss, optimizer,epoch)

        # shutil.copy('/content/GP/run_dir/{}'.format("val%f_train%f_epoch%d" % (valid_loss, train_loss, epoch)), '/content/drive/MyDrive/epochs')

    print("Best Validation Loss:", best_loss)


if __name__ == "__main__":
    """
    Usage
        Download the images data at https://nihcc.app.box.com/v/ChestXray-NIHCC
        To train on the original labels:
            python train.py --save_path run_dir --model densenet --batch_size 8 --horizontal_flip --epochs 10 --lr 0.0001 --train_weighted --valid_weighted --scale 512
        To train on the relabels:
            python train.py --save_path run_dir --model densenet --batch_size 8 --horizontal_flip --epochs 10 --lr 0.0001 --train_weighted --valid_weighted --scale 512 --tag relabeled
    """
    parser = util.get_parser()
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.save_path, "params.txt"), 'w') as out:
        json.dump(vars(args), out, indent=4)

    # load dataset files from dirve
    if not os.path.exists('/content/GP/data/'):
      os.makedirs('/content/GP/data/')

    shutil.copy('/content/drive/MyDrive/Chest_X-Ray_GP/Experiences/Experience_2/Dataset/train.csv', '/content/GP/data')
    shutil.copy('/content/drive/MyDrive/Chest_X-Ray_GP/Experiences/Experience_2/Dataset/valid.csv', '/content/GP/data')


    # save params file
    shutil.copy('/content/GP/run_dir/params.txt', '/content/drive/MyDrive/Chest_X-Ray_GP/Experiences/Experience_2')

    run(args)

