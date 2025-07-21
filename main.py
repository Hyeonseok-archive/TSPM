import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from utils import *
from model import STModel
from models import *
from config import TimesNetConfig, PatchTSTConfig, SparseTSFConfig


       

def parse_tuple(s):
    try:
        return tuple(map(int, s.strip('()').split(',')))
    except:
        raise argparse.ArgumentTypeError("Tuple must be in the format (x, y)")
    

if __name__ == '__main__':
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='1d')
    parser.add_argument('--floor', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--ex_name', type=str, default='ST')
    parser.add_argument('--loss', type=str, default='l1l2')
    
    # # parser.add_argument('--input_shape', type=tuple, default=(10, 1, 128, 128))
    # parser.add_argument('--input_shape', type=parse_tuple, default=(10, 1, 128, 128))
    # parser.add_argument('--seq_len', type=int, default=10)
    # parser.add_argument('--pred_len', type=int, default=20)
    # parser.add_argument('--hidden_dim', type=int, default=64)
    # parser.add_argument('--kernel_size', type=int, default=7)
    # # train ratio
    # parser.add_argument('--loss', type=str, default='l2')
    args = parser.parse_args()

    # # params
    # # epochs = 50
    # epochs = args.epochs
    # warmup_epoch = args.warmup_epoch
    # lr = args.lr
    # batch_size = args.batch_size

    # data_root = './dataset'
    # check_path(data_root)

    if args.data == '2d':
        # 2d heatmap seq dataset
        path_2d = os.path.join(args.data_dir, f'2d/floor{args.floor}.npy')
        path_1d = os.path.join(args.data_dir, f'1d/floor{args.floor}.csv')

        data_2d = np.load(path_2d)
        data_1d = pd.read_csv(path_1d).drop(['Date', 'sum'], axis=1).to_numpy()

        data_2d_trans, min_2d, max_2d = minmax(data_2d, axis=0)
        data_1d_trans, min_1d, max_1d = minmax(data_1d, axis=0)

        X2d, y2d = make_sequence(data_2d_trans, seq_len=args.seq_len)
        X1d, y1d = make_sequence(data_1d_trans, seq_len=args.seq_len)
        
        if args.seq_len != args.pred_len:
            y1d = y1d[:, :args.pred_len]
            

        train_size = int(len(X2d) * args.ratio)
        eval_size = int(len(X2d) * args.val_ratio)

        train_X = X2d[:train_size]
        val_X = X2d[train_size:train_size+eval_size]
        test_X = X2d[train_size+eval_size:]
    elif args.data == '1d':
        # 1d seq dataset
        path_1d = os.path.join(args.data_dir, f'1d/floor{args.floor}.csv')

        data_1d = pd.read_csv(path_1d).drop(['Date', 'sum'], axis=1).to_numpy()

        data_1d_trans, min_1d, max_1d = minmax(data_1d, axis=0)

        X1d, y1d = make_sequence(data_1d_trans, seq_len=args.seq_len)

        if args.seq_len != args.pred_len:
            y1d = y1d[:, :args.pred_len]

        train_size = int(len(X1d) * args.ratio)
        eval_size = int(len(X1d) * args.val_ratio)

        train_X = X1d[:train_size]
        val_X = X1d[train_size:train_size+eval_size]
        test_X = X1d[train_size+eval_size:]
    else:
        print('Error!')

    train_y = y1d[:train_size]
    val_y = y1d[train_size:train_size+eval_size]
    test_y = y1d[train_size+eval_size:]

    train_set = CustomDataset(train_X, train_y)
    val_set = CustomDataset(val_X, val_y)
    test_set = CustomDataset(test_X, test_y)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    print(f'train: {train_X.shape[0]}')
    print(f'val: {val_X.shape[0]}')
    print(f'test: {test_X.shape[0]}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xx, yy = next(iter(train_loader))
    if args.floor < 3:
        section=4
    else:
        section=5
    # model = STModel(in_shape=xx.shape[1:], s_hid=32, t_hid=512, n_section=section)
    # model = NLinear(seq_len=args.seq_len, pred_len=args.seq_len, c_in=section)
    # model = DLinear(seq_len=args.seq_len, pred_len=args.seq_len, c_in=section)
    # model = SegRNN(
    #     seq_len=args.seq_len, pred_len=args.seq_len, enc_in=section, d_model=128,
    #     seg_len=24 if args.seq_len % 24 == 0 else 12,
    #     dropout=0.0, rnn_type='gru', dec_way='pmf', channel_id=True, revin=True
    # )
    config = TimesNetConfig(
        seq_len=args.seq_len, pred_len=args.pred_len, enc_in=section, c_out=section
    )
    model = TimesNet(config)
    # config = PatchTSTConfig(
    #     seq_len=args.seq_len, seg_len = 24 if args.seq_len % 24 ==0 else 12, enc_in=section
    # )
    # model = PatchTST(config)
    # config = SparseTSFConfig(
    #     seq_len=args.seq_len, enc_in=section,
    #     period_len = 24 if args.seq_len % 24 ==0 else 12
    # )
    # model = SparseTSF(config)
    save_path = f'./logs/{args.ex_name}'

    if args.data == '2d':
        flops = FlopCountAnalysis(model, torch.randn(1, args.seq_len, 12, 20))
    else:
        flops = FlopCountAnalysis(model, torch.randn(1, args.seq_len, section))

    check_path(save_path)
    check_path(os.path.join(save_path, f'f{args.floor}'))
    with open(os.path.join(save_path, f'floor{args.floor}_{args.epochs}epochs.txt'), 'w') as log_file:
        print(flop_count_table(flops))
        print(flop_count_table(flops), file=log_file)

    model = model.to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineLRScheduler(optimizer=opt, t_initial=args.epochs, lr_min=1e-06, warmup_lr_init=1e-05, warmup_t=args.warmup_epoch)
        
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    best_loss = np.inf
    best_epoch = 0
    
    
    with open(os.path.join(save_path, f'floor{args.floor}_{args.epochs}epochs.txt'), 'a') as log_file:
        # params = count_parameters(model)
        # print(f'model params: {params}')
        # print(f'model params: {params}', file=log_file)

        # print(f'model gflops: {flops.total()/1e-9}')
        # print(f'model gflops: {flops.total()/1e-9}', file=log_file)
            
        for epoch in range(args.epochs):
            # train
            model.train()
            t0 = time.time()
            loss_train = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()

                if args.seq_len < args.pred_len:
                    preds_y = []
                    d = args.pred_len // args.seq_len
                    m = args.pred_len % args.seq_len
                    cur_seq = X.clone()
                    for _ in range(d):
                        cur_seq = model(cur_seq)
                        preds_y.append(cur_seq)
                    if m != 0:
                        cur_seq = model(cur_seq)
                        preds_y.append(cur_seq[:, :m])
                    preds = torch.cat(preds_y, dim=1)
                elif args.seq_len > args.pred_len:
                    preds = model(X)[:, :args.pred_len]
                else:
                    preds = model(X)
                
                if args.loss == 'l2':
                    loss = criterion1(preds, y)
                else:
                    loss = (10*criterion1(preds, y) + criterion2(preds, y))
                loss.backward()
                opt.step()
                
                loss_train += loss.item()
            loss_train = loss_train / len(train_loader)
            
            print(f'Epoch {epoch + 1} | Loss : {loss_train:.6f} | Time : {time.time() - t0:.4f}')
            print(f'Epoch {epoch + 1} | Loss : {loss_train:.6f} | Time : {time.time() - t0:.4f}', file=log_file)

            # test
            model.eval()
            t1 = time.time()
            with torch.no_grad():
                total_mse, total_mae = 0, 0
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)

                    # preds = model(X)
                    if args.seq_len < args.pred_len:
                        preds_y = []
                        d = args.pred_len // args.seq_len
                        m = args.pred_len % args.seq_len

                        cur_seq = X.clone()
                        for _ in range(d):
                            cur_seq = model(cur_seq)
                            preds_y.append(cur_seq)

                        if m != 0:
                            cur_seq = model(cur_seq)
                            preds_y.append(cur_seq[:, :m])

                        preds = torch.cat(preds_y, dim=1)
                    elif args.seq_len > args.pred_len:
                        preds = model(X)[:, :args.pred_len]                        
                    else:
                        preds = model(X)

                    # preds = preds.detach().cpu().numpy()
                    # y = y.detach().cpu().numpy()

                    # mse_batch = MSE(preds, y)
                    # mae_batch = MAE(preds, y)
                    if args.loss == 'l2':
                        loss = criterion1(preds, y)
                    else:
                        loss = (10*criterion1(preds, y) + criterion2(preds, y))
                    # loss = (10*criterion1(preds, y) + criterion2(preds, y))

                    # total_mse += mse_batch
                    # total_mae += mae_batch
                    total_mse += loss.item()
                
                # total_mse = total_mse / len(test_loader)
                # total_mae = total_mae / len(test_loader)
                total_mse = total_mse / len(test_loader)
            


            scheduler.step(epoch=epoch, metric=total_mse)

            # print(f'Epoch {epoch + 1} | Test MSE : {total_mse:.6f} | Test MAE : {total_mae:.6f} | Time : {time.time() - t1:.4f}')
            # print(f'Epoch {epoch + 1} | Test MSE : {total_mse:.6f} | Test MAE : {total_mae:.6f} | Time : {time.time() - t1:.4f}', file=log_file)
            print(f'Epoch {epoch + 1} | Val Loss : {total_mse:.6f} | Time : {time.time() - t1:.4f}')
            print(f'Epoch {epoch + 1} | Val Loss : {total_mse:.6f} | Time : {time.time() - t1:.4f}', file=log_file)
            if total_mse < best_loss:
                best_loss = total_mse
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(save_path, f'f{args.floor}/best_model.pth'))
                print(f'Best model saved with loss {best_loss:.6f} at epoch {epoch + 1}')
                print(f'Best model saved with loss {best_loss:.6f} at epoch {epoch + 1}', file=log_file)

        torch.save(model.state_dict(), os.path.join(save_path, f'f{args.floor}/last_model.pth'))

        print('Test')
        print('Test', file=log_file)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load(os.path.join(save_path, f'f{args.floor}/best_model.pth')))
            total_mse, total_mae, total_rmse, total_mape, total_smape = 0, 0, 0, 0, 0
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                # preds = model(X)
                if args.seq_len < args.pred_len:
                    preds_y = []
                    d = args.pred_len // args.seq_len
                    m = args.pred_len % args.seq_len
                    cur_seq = X.clone()
                    for _ in range(d):
                        cur_seq = model(cur_seq)
                        preds_y.append(cur_seq)
                    if m != 0:
                        cur_seq = model(cur_seq)
                        preds_y.append(cur_seq[:, :m])

                    preds = torch.cat(preds_y, dim=1)
                elif args.seq_len > args.pred_len:
                    preds = model(X)[:, :args.pred_len]                    
                else:
                    preds = model(X)

                pred = preds.detach().cpu().numpy()
                true = y.detach().cpu().numpy()

                mse_batch = MSE(pred, true, axis=(0, 1))
                mae_batch = MAE(pred, true, axis=(0, 1))
                rmse_batch = MSE(pred, true, axis=(0, 1), root=True)
                mape_batch = MAPE(pred, true, axis=(0, 1))
                smape_batch = SMAPE(pred, true, axis=(0, 1))

                total_mse += mse_batch
                total_mae += mae_batch
                total_rmse += rmse_batch
                total_mape += mape_batch
                total_smape += smape_batch

            
            total_mse = total_mse / len(test_loader)
            total_mae = total_mae / len(test_loader)
            total_rmse = total_rmse / len(test_loader)
            total_mape = total_mape / len(test_loader)
            total_smape = total_smape / len(test_loader)
        
        print(f'MSE : {total_mse.mean():.6f} | MAE : {total_mae.mean():.6f} | RMSE : {total_rmse.mean():.6f} | MAPE : {total_mape.mean():.6f} | SMAPE : {total_smape.mean():.6f}')
        print(f'MSE : {total_mse.mean():.6f} | MAE : {total_mae.mean():.6f} | RMSE : {total_rmse.mean():.6f} | MAPE : {total_mape.mean():.6f} | SMAPE : {total_smape.mean():.6f}', file=log_file)
        
        
        