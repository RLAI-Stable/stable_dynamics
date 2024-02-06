#!/usr/bin/env python3

import argparse
import datetime
import glob
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util import DynamicLoad, setup_logging, to_variable

logger = setup_logging(os.path.basename(__file__))
torch.set_default_dtype(torch.float64)

def runbatch(args, model, loss, batch):
    X, Yactual = batch
    X = to_variable(X, cuda=torch.cuda.is_available())
    Yactual = to_variable(Yactual, cuda=torch.cuda.is_available())

    Ypred = model(X)
    return loss(Ypred, Yactual, X), Ypred

def test_model(args, model, test_dataloader, epoch=None, summarywriter=None):
    loss_parts = []
    model.eval()
    for batch_idx, data in enumerate(test_dataloader):
        loss, Ypred = runbatch(args, model, args.model.loss, data)
        loss_parts.append(np.array([l.cpu().item() for l in args.model.loss_flatten(loss)]))

        # Add parts to the summary if needed.
        args.model.summary(epoch, summarywriter, Ypred, data[0])

    return sum(loss_parts) / len(test_dataloader.dataset)

def build_loss_log(args, train_test, epoch, loss_elements):
    now = datetime.datetime.now()

    loss_row = {
        "timestamp": now,
        "learning_rate": args.learning_rate,
        "model_type": args.modeltype,
        "train_or_test": train_test,
        "epoch": epoch,
        "loss": loss_elements[0],
    }

    return loss_row

def main(args):
    writer = SummaryWriter(logdir=args.log_to)
    model = args.model.model
    dataset = args.dataset

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if args.test_with:
        test_dataloader = DataLoader(args.test_with, batch_size=args.batch_size, shuffle=False)
    else:
        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # TODO: Resume training support
    loss_list = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_parts = []

        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, _ = runbatch(args, model, args.model.loss, data)
            loss_parts.append(np.array([l.cpu().item() for l in args.model.loss_flatten(loss)]))
            optim_loss = loss[0] if isinstance(loss, (tuple, list)) else loss
            optim_loss.backward()
            optimizer.step()

        losses_in_epoch = sum(loss_parts) / len(dataset)
        loss_list.append(build_loss_log(args, "TRAIN", epoch, losses_in_epoch))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.weights.format(epoch=epoch))
            test_loss = test_model(args, model, test_dataloader, epoch=epoch, summarywriter=writer)
            loss_list.append(build_loss_log(args, "TEST", epoch, test_loss))

    
    with open(args.error_path, 'wb') as f:
        pickle.dump(loss_list, f)
        
    # Ensure the writer is completed.
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE on a set of videos, fine-tune it on a single video, and generate the decoder.')
    parser.set_defaults(func=lambda *a: parser.print_help())

    parser.add_argument('dataset', type=DynamicLoad("datasets"), help='dataset to train on')
    parser.add_argument('model', type=DynamicLoad("models"), help='model to train with')
    parser.add_argument('weights', type=str, help='save model weights')
    parser.add_argument('--log-to', type=str, help='log destination within runs/')
    parser.add_argument('--test-with', type=DynamicLoad("datasets"), default=None, help='dataset to test with instead of the training data')
    parser.add_argument('--modeltype', type=str, help='type of the model, stable or simple')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to run')
    parser.add_argument('--save-every', type=int, default=5, help='save after this many epochs')
    parser.add_argument('--error-path', type=str, help='path to save the evaluation errors of the model')

    parser.set_defaults(func=main)

    try:
        args = parser.parse_args()
        main(args)
    except:
        raise
