import os
import torch
import argparse
import torch.optim as optim
import pandas as pd
from model.Multi_model import MLP
from utils.utils_multi_loss import train_one_epoch, evaluate_model
from data_pre_processing.load_data_torch import prepare_data, prepare_data_multi_modl


def get_args_parser(crossVal):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)

    path1 = "csv/3 crosval_feature1521/t1wi"
    path2 = "csv/3 crosval_feature1521/flair"
    path3 = "csv/3 crosval_feature1521/dwi"

    train_path = "crossVal_" + str(crossVal) + "_train.csv"
    val_path = "crossVal_" + str(crossVal) + "_val.csv"

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
                        default=r'result/0.pth')

    # 权重路径
    parser.add_argument('--weights', type=str,
                        default=r'result/multi_model/3model_2_crossVal_' + str(crossVal) + '' + '.pth')

    return parser.parse_args()


def tain_one_model(args, the_weight):
    train_loader, val_loader = prepare_data_multi_modl(args.train_data_path1,
                                                       args.train_data_path2,
                                                       args.train_data_path3,
                                                       args.val_data_path1,
                                                       args.val_data_path2,
                                                       args.val_data_path3,
                                                       batch_size=args.batch_size)

    model = MLP(1521)
    if args.weights != "":  # 加载预训练模型
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)
        print(model.load_state_dict(weights_dict, strict=False))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc, val_acc, the_auc = 0.0, 0.0, 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        # enumerate mini batches
        train_one_epoch(model=model,
                        train_dl=train_loader,
                        optimizer=optimizer,
                        the_weight=the_weight)

        val_acc, roc_auc = evaluate_model(model=model,
                                          val_dl=val_loader)

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            the_auc = roc_auc
            torch.save(model.state_dict(), args.save_path)

    return best_acc, the_auc


if __name__ == '__main__':
    weight = [0.2, 0.4, 0.6, 0.8, 1.0]

    all_result = []

    for a1 in weight:
        for a2 in weight:
            for a3 in weight:
                for a in weight:
                    acc_sum = 0
                    auc_sum = 0
                    for i in range(5):
                        opt = get_args_parser(i)

                        acc, auc = tain_one_model(opt, [a1, a2, a3, a])
                        acc_sum = acc_sum + acc
                        auc_sum = auc_sum + auc

                    result = [a1, a2, a3, a, acc_sum/5, auc_sum/5]
                    print(result)
                    all_result.append(result)

    pd.DataFrame(all_result).to_csv("all_result.csv")
