import copy
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

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.selected_tasks = args.selected_tasks

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
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

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        meta_val_data, meta_val_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        autol = AutoLambda(self.model, self.device, self.selected_tasks, self.args.pred_len, self.args.label_len, criterion)
        meta_weight_ls = np.zeros([self.args.train_epochs, len(self.selected_tasks)], dtype=np.float32)
        meta_optimizer = optim.Adam([autol.meta_weights], lr=self.args.autol_lr)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_time = []
        vali_time = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, ((batch_x, batch_y, batch_x_mark, batch_y_mark),
                (val_batch_x, val_batch_y, val_batch_x_mark, val_batch_y_mark)) in enumerate(zip(train_loader, meta_val_loader)):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                val_batch_x = val_batch_x.float().to(self.device)
                val_batch_y = val_batch_y.float().to(self.device)
                val_batch_x_mark = val_batch_x_mark.float().to(self.device)
                val_batch_y_mark = val_batch_y_mark.float().to(self.device)

                meta_optimizer.zero_grad()
                autol.unrolled_backward(batch_x, batch_y, batch_x_mark, batch_y_mark,
                                    val_batch_x, val_batch_y, val_batch_x_mark, val_batch_y_mark,
                                    self.args.learning_rate, model_optim)
                meta_optimizer.step()

                model_optim.zero_grad()


                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=batch_y.dtype, device=self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, self.selected_tasks]
                loss = [criterion(outputs[:, :, i], batch_y[:, :, i]) for i in range(len(self.selected_tasks))]
                loss = sum(w * loss[i] for i, w in enumerate(autol.meta_weights))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_time.append(time.time() - epoch_time)
            train_loss = np.average(train_loss)
            vali_start = time.time()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            vali_time.append(time.time() - vali_start)
            # test_loss = self.vali(test_data, test_loader, criterion)
            meta_weight_ls[epoch] = autol.meta_weights.detach().cpu()


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # save train_time and vali_time
        np.save(path + '/train_time.npy', np.array(train_time))
        np.save(path + '/vali_time.npy', np.array(vali_time))
        np.save(path + '/meta_weights.npy', meta_weight_ls)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
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
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, per_mae, per_mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('per mse:{}, per mae{}'.format(per_mse, per_mae))
        f = open("result_long_term_forecast.txt", 'a')
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


class AutoLambda:
    def __init__(self, model, device, train_tasks, pred_len, label_len, criterion, weight_init=0.1):
        self.model = model
        self.model_ = copy.deepcopy(model)
        self.device = device
        self.meta_weights = torch.tensor([weight_init] * len(train_tasks), requires_grad=True, device=device)
        self.train_tasks = train_tasks
        self.pred_len = pred_len
        self.label_len = label_len
        self.criterion = criterion


    def virtual_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        train_loss = self.model_fit(self.model, batch_x, batch_y, batch_x_mark, batch_y_mark)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

    def unrolled_backward(self, batch_x, batch_y, batch_x_mark, batch_y_mark,
        val_batch_x, val_batch_y, val_batch_x_mark, val_batch_y_mark, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(batch_x, batch_y, batch_x_mark, batch_y_mark, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            pri_weights += [1.0]

        # compute validation data loss on primary tasks
        val_loss = self.model_fit(self.model_, val_batch_x, val_batch_y, val_batch_x_mark, val_batch_y_mark)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, batch_x, batch_y, batch_x_mark, batch_y_mark)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = - alpha * h

    def compute_hessian(self, d_model, batch_x, batch_y, batch_x_mark, batch_y_mark):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        train_loss = self.model_fit(self.model, batch_x, batch_y, batch_x_mark, batch_y_mark)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        train_loss = self.model_fit(self.model, batch_x, batch_y, batch_x_mark, batch_y_mark)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def model_fit(self, model, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        define task specific losses
        """
        outputs, batch_y = self.model_forward(model, batch_x, batch_y, batch_x_mark, batch_y_mark)
        loss = [self.criterion(outputs[:, :, i], batch_y[:, :, i]) for i in range(len(self.train_tasks))]
        return loss

    def model_forward(self, model, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :], dtype=batch_y.dtype, device=self.device)
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1)

        # encoder - decoder
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        outputs = outputs[:, -self.pred_len:, :]
        batch_y = batch_y[:, -self.pred_len:, self.train_tasks]

        return outputs, batch_y