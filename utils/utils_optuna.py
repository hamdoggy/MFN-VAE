import sys
import torch
import torch.nn as nn
# from tqdm import tqdm
from numpy import vstack
from torch.nn import BCELoss
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn import metrics


def loss_fn(y, x, mu, log_var):
    recons_loss = F.mse_loss(y, x)  
    kld_loss = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, 1), 0)
    return recons_loss + 0.5 * kld_loss


class MultiWeightedLoss(nn.Module):

    def __init__(self, num=4, the_weight=torch.tensor([0.25, 0.25, 0.25, 1.0])):
        super(MultiWeightedLoss, self).__init__()
        # params = torch.ones(num, requires_grad=True)
        params = the_weight
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += loss * self.params[i]
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def train_one_epoch(model, train_dl, optimizer, the_weight):
    model.train()
    multi_loss = MultiWeightedLoss(4, torch.tensor(the_weight))

    criterion_cl = BCELoss()

    accu_loss = torch.zeros(1)
    the_loss = 0
    for i, data in enumerate(train_dl):
        # clear the gradients
        optimizer.zero_grad()

        (inputs1, inputs2, inputs3, targets) = data
        # print(inputs1)
        yhat_t1wi, yhat_flair, yhat_dwi, yhat, \
        t1wi_mean, t1wi_var, flair_mean, flair_var, dwi_mean, dwi_var = model(inputs1, inputs2, inputs3)

        loss_t1wi = loss_fn(yhat_t1wi, inputs1, t1wi_mean, t1wi_var)
        loss_flair = loss_fn(yhat_flair, inputs2, flair_mean, flair_var)
        loss_dwi = loss_fn(yhat_dwi, inputs3, dwi_mean, dwi_var)

        loss_cl = criterion_cl(yhat, targets)

        loss_sum = multi_loss(loss_t1wi, loss_flair, loss_dwi, loss_cl)

        # calculate loss
        loss_sum.backward()

        accu_loss += loss_sum.detach()

        the_loss = accu_loss.item() / (i + 1)
        # train_dl.desc = "[train|epoch: {}] batch: {}, loss: {:.4f}" \
        #     .format(epoch, i, the_loss)

        # update model weights
        optimizer.step()

    return the_loss


def evaluate_model(model, val_dl):
    model.eval()
    predictions, actuals, roc_predict = [], [], []

    for i, data in enumerate(val_dl):
        # evaluate the model on the test set
        (inputs1, inputs2, inputs3, targets) = data
        yhat_t1wi, yhat_flair, yhat_dwi, yhat, \
        t1wi_mean, t1wi_var, flair_mean, flair_var, dwi_mean, dwi_var = model(inputs1, inputs2, inputs3)

        # retrieve numpy array
        yhat = yhat.detach().numpy()

        roc_predict.append(yhat)
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals, roc_predict = vstack(predictions), vstack(actuals), vstack(roc_predict)

    fpr, tpr, threshold = metrics.roc_curve(actuals, roc_predict)
    roc_auc = metrics.auc(fpr, tpr)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)

    return acc, roc_auc
