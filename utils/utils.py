import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from numpy import vstack
from torch.nn import BCELoss
from sklearn.metrics import accuracy_score
from sklearn import metrics

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum




def train_one_epoch(model, train_dl, optimizer, epoch, model_num=1):
    model.train()
    criterion = BCELoss()
    train_dl = tqdm(train_dl, file=sys.stdout)

    accu_loss = torch.zeros(1)
    the_loss = 0
    for i, data in enumerate(train_dl):

        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        if model_num == 2:
            (inputs1, inputs2, targets) = data
            yhat = model(inputs1, inputs2)
        if model_num == 3:
            (inputs1, inputs2, inputs3, targets) = data
            yhat = model(inputs1, inputs2, inputs3)
        else:
            (inputs, targets) = data
            yhat = model(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        loss.backward()

        accu_loss += loss.detach()

        the_loss = accu_loss.item() / (i + 1)
        train_dl.desc = "[train|epoch: {}] batch: {}, loss: {:.4f}"\
            .format(epoch, i, the_loss)

        # update model weights
        optimizer.step()

    return the_loss


def evaluate_model(model, val_dl, model_num=1):
    model.eval()
    predictions, actuals = [], []
    for i, data in enumerate(val_dl):
        # evaluate the model on the test set

        if model_num == 2:
            (inputs1, inputs2, targets) = data
            yhat = model(inputs1, inputs2)
        if model_num == 3:
            (inputs1, inputs2, inputs3, targets) = data
            yhat = model(inputs1, inputs2, inputs3)
        else:
            (inputs, targets) = data
            yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals = vstack(predictions), vstack(actuals)
    # fpr, tpr, threshold = metrics.roc_curve(actuals, predictions)
    # roc_auc = metrics.auc(fpr, tpr)

    # calculate accuracy
    acc = accuracy_score(actuals, predictions)

    return acc


# def train_one_epoch_3model(model, train_dl, optimizer, epoch):
#     model.train()
#     criterion = BCELoss()
#     train_dl = tqdm(train_dl, file=sys.stdout)
#
#     accu_loss = torch.zeros(1)
#     the_loss = 0
#     for i, (inputs1, inputs2, targets) in enumerate(train_dl):
#         # clear the gradients
#         optimizer.zero_grad()
#         # compute the model output
#         yhat = model(inputs1, inputs2)
#         # calculate loss
#         loss = criterion(yhat, targets)
#         loss.backward()
#
#         accu_loss += loss.detach()
#
#         the_loss = accu_loss.item() / (i + 1)
#         train_dl.desc = "[train|epoch: {}] batch: {}, loss: {:.4f}"\
#             .format(epoch, i, the_loss)
#
#         # update model weights
#         optimizer.step()
#
#     return the_loss
