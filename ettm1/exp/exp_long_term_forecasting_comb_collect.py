import pickle
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

from rich.progress import Progress
from itertools import combinations_with_replacement
def combine(temp_list):
    temp_list2 = []
    for n in (1,2):
        for c in combinations_with_replacement(temp_list, n):
            temp_list2.append(c)
    return temp_list2


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.selected_tasks = args.selected_tasks
        self.backup_model = self._build_backup_model().to(self.device)
        self.model = torch.compile(self.model)
        self.backup_model = torch.compile(self.backup_model)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _build_backup_model(self):
        backup_model = self.model_dict[self.args.model].Model(self.args).float()
        return backup_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        backup_optim = optim.Adam(self.backup_model.parameters(), lr=self.args.learning_rate)
        return model_optim, backup_optim

    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='none')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, self.selected_tasks]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.mean())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        tdg_task_groups = combine(self.selected_tasks)

        # result save
        self.result_path = './MS_collection/' + setting + '/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        
        checkpoint_path = os.path.join(self.result_path, 'checkpoints')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, backup_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_task_gains = {
            '|'.join(map(str, group)) : {t: [] for t in self.selected_tasks} for group in tdg_task_groups
        }

        epoch_train_loss = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            task_gains = {
                '|'.join(map(str, group)) : {t: [] for t in self.selected_tasks} for group in tdg_task_groups
            }
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            progress = Progress()
            progress.start()
            num_batches = len(train_loader)
            progress_task = progress.add_task("[red]Training...", total=num_batches)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                progress.update(progress_task, advance=1, description=f"[red]Training... {i}/{num_batches}")
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=batch_y.dtype, device=self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                # backup model and optimizer
                self.backup_model.load_state_dict(self.model.state_dict())
                backup_optim.load_state_dict(model_optim.state_dict())
                # compute task gains
                for group in tdg_task_groups:
                    # using backup model and optimizer
                    model_optim.zero_grad()
                    group_name = '|'.join(map(str, group))
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = outputs[:, -self.args.pred_len:, group]
                            out_batch_y = batch_y[:, -self.args.pred_len:, group]
                            loss = (outputs - out_batch_y).square().mean(0).mean(0)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len:, group]
                        out_batch_y = batch_y[:, -self.args.pred_len:, group]
                        loss = (outputs - out_batch_y).square().mean(0).mean(0)
                    group_loss = loss.mean()
                    if self.args.use_amp:
                        scaler.scale(group_loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        group_loss.backward()
                        model_optim.step()
                    # decoder input
                    g_dec_inp = dec_inp
                    if self.args.output_attention:
                        g_outputs = self.model(batch_x, batch_x_mark, g_dec_inp, batch_y_mark)[0]
                    else:
                        g_outputs = self.model(batch_x, batch_x_mark, g_dec_inp, batch_y_mark)

                    g_outputs = g_outputs[:, -self.args.pred_len:, :]
                    g_out_batch_y = batch_y[:, -self.args.pred_len:, :]
                    g_loss = (g_outputs - g_out_batch_y).square().mean(0).mean(0)
                    
                    for task in self.selected_tasks:
                        task_gains[group_name][task].append(g_loss[task].item())
                    # restore model and optimizer
                    self.model.load_state_dict(self.backup_model.state_dict())
                    model_optim.load_state_dict(backup_optim.state_dict())

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, self.selected_tasks]
                        loss = criterion(outputs, batch_y).mean(0).mean(0)
                        train_loss.append(loss.detach().cpu().numpy())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, self.selected_tasks]
                    loss = criterion(outputs, batch_y).mean(0).mean(0)
                    train_loss.append(loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.mean().item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss.mean()).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.mean().backward()
                    model_optim.step()
            progress.stop()

            train_loss = np.stack(train_loss)
            print(train_loss.shape)
            epoch_train_loss.append(train_loss)
            for group in tdg_task_groups:
                group_name = '|'.join(map(str, group))
                for task in self.selected_tasks:
                    epoch_task_gains[group_name][task].append(task_gains[group_name][task])
                    print('task {} gain: {}'.format(task, epoch_task_gains[group_name][task][-1]))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss.mean(), vali_loss))
            early_stopping(vali_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # save epoch_task_gains into result_path, using pickle
            with open(self.result_path + f'epoch_{epoch}_task_gains.pkl', 'wb') as f:
                pickle.dump(epoch_task_gains, f)
            # save epoch_train_loss into result_path, using pickle
            with open(self.result_path + f'epoch_{epoch}_train_loss.pkl', 'wb') as f:
                pickle.dump(epoch_train_loss, f)

            with open(self.result_path + 'epoch_lookahead_times.pkl', 'wb') as f:
                pickle.dump({
                    'fw': self.lookahead_fw_time,
                    'bp': self.lookahead_bp_time,
                    'total': self.lookahead_time
                }, f)

        best_model_path = checkpoint_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # save epoch_task_gains into result_path, using pickle
        with open(self.result_path + 'epoch_task_gains.pkl', 'wb') as f:
            pickle.dump(epoch_task_gains, f)
        # save epoch_train_loss into result_path, using pickle
        with open(self.result_path + 'epoch_train_loss.pkl', 'wb') as f:
            pickle.dump(epoch_train_loss, f)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = self.result_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs
                batch_y = batch_y[:, :, self.selected_tasks]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe, per_mae, per_mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('per mse:{}, per mae{}'.format(per_mse, per_mae))
        f = open(self.result_path + "result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}  \n'.format(mse, mae))
        f.write('per mse:{}, per mae{}  \n'.format(per_mse, per_mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'per_metrics.npy', np.array([per_mse, per_mae]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return