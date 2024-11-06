import os
import argparse
import time
import torch
import optuna
import torch.optim as optim
from model.Multi_model import MLP
from utils.utils_multi_loss import train_one_epoch, evaluate_model
from data_pre_processing.load_data_torch import prepare_data_multi_modl
import pandas as pd


def get_args_parser(batch_size, lr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--lr', type=float, default=lr)

    path1 = "csv/3 crosval_feature1521/t1wi"
    path2 = "csv/3 crosval_feature1521/flair"
    path3 = "csv/3 crosval_feature1521/dwi"

    crossVal = "4"
    train_path = "crossVal_" + crossVal + "_train.csv"
    val_path = "crossVal_" + crossVal + "_val.csv"

    # 数据集所在根目录
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
    # 保存权重
    parser.add_argument('--save_path', type=str,
                        default=r'result/0')

    # 权重路径
    parser.add_argument('--weights', type=str,
                        default=r'result/multi_model/3model_2_crossVal_' + crossVal + '' + '.pth')

    return parser.parse_args()


def training(args, loss_weight, trail):
    number_pth = len(os.listdir(args.save_path))
    save_path = os.path.join(args.save_path, str(number_pth) + '-' + 'Lr[{:.3f}]'.format(args.lr) + '-' +
                             'LossWeight[{:.2f},{:.2f},{:.2f},{:.2f}]'
                             .format(loss_weight[0], loss_weight[1], loss_weight[2], loss_weight[3]) + '.pth')

    train_loader, val_loader = prepare_data_multi_modl(args.train_data_path1,
                                                       args.train_data_path2,
                                                       args.train_data_path3,
                                                       args.val_data_path1,
                                                       args.val_data_path2,
                                                       args.val_data_path3,
                                                       batch_size=args.batch_size)

    model = MLP(1521)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc, val_acc, the_auc = 0.0, 0.0, 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        train_one_epoch(model=model,
                        train_dl=train_loader,
                        optimizer=optimizer,
                        the_weight=loss_weight)

        val_acc, val_auc = evaluate_model(model=model,
                                          val_dl=val_loader)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            the_auc = val_auc
            # save_path = os.path.join(args.save_path,
            #                          str(number_pth) + '-' + 'acc_' + '{:0.3f}'.format(best_acc) + '.pth')
            torch.save(model.state_dict(), save_path)

        trail.report(val_auc, epoch)
        if trail.should_prune():
            raise optuna.TrialPruned()

    return best_acc, the_auc


def objective(trail):
    batch_size = 32
    # batch_size = trail.suggest_int('batchsize', 4, 16)
    lr = trail.suggest_float('lr', 1e-3, 1e-2, step=0.001)
    loss_weight_0 = trail.suggest_float('loss_weight_0', 0.1, 1, step=0.1)
    loss_weight_1 = trail.suggest_float('loss_weight_1', 0.1, 1, step=0.1)
    loss_weight_2 = trail.suggest_float('loss_weight_2', 0.1, 1, step=0.1)
    loss_weight_3 = trail.suggest_float('loss_weight_3', 0.1, 1, step=0.1)
    loss_weight = [loss_weight_0, loss_weight_1, loss_weight_2, loss_weight_3]

    acc, auc = training(get_args_parser(batch_size, lr), loss_weight, trail)
    return auc


if __name__ == '__main__':
    st = time.time()

    study = optuna.create_study(study_name='loss_weight', direction='maximize')
    study.optimize(objective, n_trials=50)

    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)

    print('Time: {:.2f} min'.format((time.time() - st) / 60))

    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()
