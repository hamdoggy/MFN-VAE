import argparse
import os
import time

import optuna
import torch
import torch.optim as optim

from data_pre_processing.load_data_torch import prepare_data_multi_modl
from model.Multi_model import MLP
from utils.utils_multi_loss import train_one_epoch, evaluate_model


def get_args_parser(batch_size=8, lr=0.01):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--lr', type=float, default=lr)

    path1 = "csv/3 crosval_feature1521/t1wi"
    path2 = "csv/3 crosval_feature1521/flair"
    path3 = "csv/3 crosval_feature1521/dwi"

    crossVal = "4"
    train_path = "crossVal_" + crossVal + "_train.csv"
    val_path = "crossVal_" + crossVal + "_val.csv"

    # Root directory of the dataset
    parser.add_argument('--train_data_path1', type=str,
                        default=os.path.join(path1, train_path))
    parser.add_argument('--train_data_path2', type=str,
                        default=os.path.join(path2, train_path))
    parser.add_argument('--train_data_path3', type=str,
                        default=os.path.join(path3, train_path))

    parser.add_argument('--val_data_path1', type=str,
                        default=os.path.join(path1, val_path))
    parser.add_argument('--val_data_path2', type=str,
                        default=os.path.join(path2, val_path))
    parser.add_argument('--val_data_path3', type=str,
                        default=os.path.join(path3, val_path))
    # Save path for model weights
    parser.add_argument('--save_path', type=str,
                        default=r'result/0')

    # Path to pre-trained weights
    parser.add_argument('--weights', type=str,
                        default=r'result/multi_model/3model_2_crossVal_' + crossVal + '' + '.pth')

    return parser.parse_args()


def load_data(args):
    # Load train and validation datasets
    train_data, val_data = prepare_data_multi_modl(args.train_data_path1,
                                                   args.train_data_path2,
                                                   args.train_data_path3,
                                                   args.val_data_path1,
                                                   args.val_data_path2,
                                                   args.val_data_path3,
                                                   batch_size=args.batch_size)

    return train_data, val_data


def training(args, loss_weight, trail, train_data_loader, val_data_loader):
    number_pth = len(os.listdir(args.save_path))
    save_path = os.path.join(args.save_path, str(number_pth) + '-' + 'Lr[{:.3f}]'.format(args.lr) + '-' +
                             'LossWeight[{:.2f},{:.2f},{:.2f},{:.2f}]'
                             .format(loss_weight[0], loss_weight[1], loss_weight[2], loss_weight[3]) + '.pth')

    model = MLP(1521)
    if args.weights != "":
        # Load pre-trained weights if specified
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc, val_acc, the_auc = 0.0, 0.0, 0.0
    # best_epoch = 0
    for epoch in range(args.epochs):
        # Train for one epoch
        train_one_epoch(model=model,
                        train_dl=train_data_loader,
                        optimizer=optimizer,
                        the_weight=loss_weight)

        # Evaluate on validation set
        val_acc, val_auc = evaluate_model(model=model,
                                          val_dl=val_data_loader)
        if val_acc >= best_acc:
            best_acc = val_acc
            # best_epoch = epoch
            the_auc = val_auc

            # Save model weights
            torch.save(model.state_dict(), save_path)

        trail.report(val_auc, epoch)
        if trail.should_prune():
            raise optuna.TrialPruned()

    return best_acc, the_auc


def objective(trail):
    lr = trail.suggest_float('lr', 1e-3, 1e-2, step=0.001)
    loss_weight_0 = trail.suggest_float('loss_weight_0', 0.1, 1, step=0.1)
    loss_weight_1 = trail.suggest_float('loss_weight_1', 0.1, 1, step=0.1)
    loss_weight_2 = trail.suggest_float('loss_weight_2', 0.1, 1, step=0.1)
    loss_weight_3 = trail.suggest_float('loss_weight_3', 0.1, 1, step=0.1)
    loss_weight = [loss_weight_0, loss_weight_1, loss_weight_2, loss_weight_3]

    # Start training with suggested hyperparameters
    acc, auc = training(args=get_args_parser(lr=lr),
                        loss_weight=loss_weight,
                        train_data_loader=train_loader,
                        val_data_loader=val_loader,
                        trail=trail)
    return auc


if __name__ == '__main__':
    st = time.time()
    train_loader, val_loader = load_data(get_args_parser(batch_size=32))

    # optuna-dashboard sqlite:///db.sqlite3
    study = optuna.create_study(study_name='loss_weight', direction='maximize', storage='sqlite:///db.sqlite3', load_if_exists=True)
    study.optimize(objective, n_trials=500)

    print('Time: {:.2f} min'.format((time.time() - st) / 60))

    # optuna.visualization.plot_param_importances(study).show()
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_slice(study).show()
