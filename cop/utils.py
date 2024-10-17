
"""
The MIT License

Copyright (c) 2021 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import time
import sys
import os
from datetime import datetime
import logging
import logging.config
import pytz
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import torch
import networkx as nx
from sklearn.linear_model import LinearRegression

process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'


def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))


def util_save_log_image_with_label(result_file_prefix,
                                   img_params,
                                   result_log: LogData,
                                   labels=None):
    dirname = os.path.dirname(result_file_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    _build_log_image_plt(img_params, result_log, labels)

    if labels is None:
        labels = result_log.get_keys()
    file_name = '_'.join(labels)
    fig = plt.gcf()
    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)


def _build_log_image_plt(img_params,
                         result_log: LogData,
                         labels=None):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    # Read json
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)

    with open(log_image_config_file, 'r') as f:
        config = json.load(f)

    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)

    if labels is None:
        labels = result_log.get_keys()
    for label in labels:
        plt.plot(*result_log.getXY(label), label=label)

    ylim_min = config['ylim']['min']
    ylim_max = config['ylim']['max']
    if ylim_min is None:
        ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None:
        ylim_max = plt.gca().dataLim.ymax
    plt.ylim(ylim_min, ylim_max)

    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None:
        xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None:
        xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)

    plt.rc('legend', **{'fontsize': 18})
    plt.legend()
    plt.grid(config["grid"])


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)


def read_score_loss(file_dir):

    info = torch.load(file_dir)

    trend = np.concatenate(info['trend_list'],axis=0)
    choices = np.array(info['choices']).reshape(-1,1)
    rewards = np.array(info['rewards']).reshape(-1,1)
    trend = np.concatenate([trend[-len(rewards):],rewards,choices],axis=-1)

    seen_tasks = file_dir.split('/')[-2].split('_')[-4].split('-')
    unseen_tasks = file_dir.split('/')[-2].split('_')[-2].split('-')[1:]

    res_dic = {}

    data = info['result_log'][1]
    score = np.concatenate([_[1].reshape(-1, 1) for _ in data['train_score']], axis=-1)
    loss = [_[1] for _ in data['train_loss']]

    count = 0
    for info in seen_tasks:
        temp_info = info.split('[')
        problem = temp_info[0]
        res_dic[problem] = {}
        res_dic[problem]['seen'] = {}
        scales = temp_info[1].strip(']').split(',')
        for scale in scales:
            res_dic[problem]['seen'][scale] = score[count, :]
            count += 1

    for info in unseen_tasks:
        temp_info = info.split('[')
        problem = temp_info[0]
        res_dic[problem]['unseen'] = {}
        scales = temp_info[1].strip(']').split(',')
        for scale in scales:
            res_dic[problem]['unseen'][scale] = score[count, :]
            count += 1
    assert count == score.shape[0]
    return res_dic, loss


def get_discouneted_average_data(data,gamma, shift):
    shift_data =[data.reshape(1,-1)]
    for i in range(shift):
        shift_data.append(np.roll(data,shift=i+1).reshape(1,-1)*gamma**(i+1))
    total_data = np.concatenate(shift_data,axis=0)
    return total_data.sum(0)[shift:]


def plot_fig(data_dic, log_flag=True):
    rows = len(data_dic.keys())
    col_leng = []
    for problem in data_dic.keys():
        seen_unseen_data = data_dic[problem]
        l = 0
        if 'seen' in seen_unseen_data.keys():
            l+=len(seen_unseen_data['seen'])
        if 'unseen' in seen_unseen_data.keys():
            l+=len(seen_unseen_data['unseen'])
        col_leng.append(l)
    cols = max(col_leng)

    plt.figure(figsize=(8 * rows, 5 * cols))
    count = 1
    for problem, datass in data_dic.items():
        temp_count = 1
        for task_type, datas in datass.items():
            for scale,data in datas.items():
                if log_flag:
                    if problem == 'TSP' or problem == 'CVRP':
                        norm_val = np.min([1e6, np.min(data)])
                        data = data - norm_val + 1e-3
                    else:
                        norm_val = np.max([0, np.max(data)])
                        data = -data + norm_val + 1e-3
                    data = np.log(data)
                ax = plt.subplot(rows,cols,count)
                ax.set_title('{}-{}-{}'.format(problem,scale,task_type),fontsize=15, fontweight='bold')

                roll_data = np.roll(data,1)
                judge = ((data<roll_data)[1:])
                dom = np.arange((len(judge)))+1
                ax.plot(np.arange(len(judge)),(np.cumsum(judge)/np.arange(len(judge))+1), label='approximated p for {}-{}-{}'.format(problem,scale, task_type))

                plt.legend(fontsize=10)


                # shift = 300
                # model = LinearRegression()
                # coefs = []
                # for _ in range(shift,len(data)):
                #     temp = data[max(0,_-shift):_]
                #     model.fit(np.arange(len(temp)).reshape(-1,1),temp)
                #     if problem == 'OP' or problem =='KP':
                #         coefs.append(-model.coef_*1e3)
                #     else:
                #         coefs.append(model.coef_*1e3)
                #
                # ax.plot(coefs, label='final slope {:.3f} with shift {}'.format(coefs[-1].item(),shift))
                # ax.plot(np.zeros(len(coefs)),'r')
                # plt.legend(fontsize=10)
                # plt.ylim(-10, 5)
                count+=1
                temp_count+=1

        count = count+cols+1-temp_count

    plt.tight_layout()
    plt.show()


def read_improvement_graph(file_dir):

    info = torch.load(file_dir)
    eval_res = info['eval_res']
    choices = info['choices']
    rewards = info['rewards']

    seen_node_idx = np.unique(choices)
    graph_size = len(eval_res[0])

    seen_task_infos = file_dir.split('/')[-2].split('_')[-4].split('-')
    unseen_task_infos = file_dir.split('/')[-2].split('_')[-2].split('-')[1:]

    seen_tasks = []
    total_tasks = []
    for info in seen_task_infos:
        temp_info = info.split('[')
        problem = temp_info[0]
        scales = temp_info[1].strip(']').replace(' ','').split(',')
        for scale in scales:
            seen_tasks.append(problem+'-'+scale)
            total_tasks.append(problem+'-'+scale)

    for info in unseen_task_infos:
        temp_info = info.split('[')
        problem = temp_info[0]
        scales = temp_info[1].strip(']').split(',')
        for scale in scales:
            total_tasks.append(problem+'-'+scale)

    unique_choice = np.unique(choices)
    cum_sum_list = []
    rew_list = []
    for choice in unique_choice:
        cum_sum = np.cumsum(np.array(choices)==choice)
        cum_sum_list.append(cum_sum)

        # rew = np.array(rewards)[np.array(choices[2:])==choice]
        # rew_list.append(np.mean(rew))

    # plt.barh(seen_tasks, rew_list)
    # plt.show()

    idxs = np.argsort([_[-1] for _ in cum_sum_list])[::-1]
    plt.figure()

    for choice in unique_choice[idxs]:
        data = cum_sum_list[choice]
        # plt.plot(cum_sum_list[choice],label='Train {}'.format(total_tasks[choice]))


        shift = 50
        model = LinearRegression()
        coefs = []
        for _ in range(shift, len(data)):
            temp = data[_ - shift:_]
            model.fit(np.arange(len(temp)).reshape(-1, 1), temp)
            if total_tasks[choice].split('-')[0] == 'KP' or total_tasks[choice].split('-')[0] == 'OP':
                coefs.append(-model.coef_ * 1e3)
            else:
                coefs.append(model.coef_ * 1e3)
        plt.title('{}'.format(total_tasks[choice]))
        plt.plot(coefs, label='maximal slope {:.3f}'.format(np.max(coefs)))
        plt.plot(np.zeros(len(coefs)), 'r')
        plt.legend(fontsize=10)
        plt.ylim(-10, 5)
        plt.show()

    # plt.legend()
    # plt.show()

    # compute improvement
    idx = []
    for i, t in enumerate(total_tasks):
        if t.split('-')[0] == 'OP' or t.split('-')[0] == 'KP':
            idx.append(i)
    improvement_res = []
    for i in range(1,len(eval_res)):
        improvement = 1 - eval_res[i] / eval_res[i-1]
        improvement[idx] = -improvement[idx]
        improvement_res.append(improvement.reshape(1,-1)*100)

    improvement_res = np.concatenate(improvement_res,axis=0)

    # for i, task in enumerate(total_tasks):
    #     data = get_discouneted_average_data(improvement_res[:,i], gamma=.8, shift=100)
    #
    #     plt.plot(data,label=task)
    #
    #     plt.legend()
    #     plt.show()

    weight_mat = np.zeros((len(seen_node_idx), graph_size))
    for i in range(len(choices)-1):
        choice = choices[i]
        weight_mat[choice] += improvement_res[i]

    # weight_mat/=len(choices)
    G = nx.DiGraph()
    edge_list = []
    for i in range(len(seen_node_idx)):
        for j in range(graph_size):
            edge_list.append((seen_tasks[i],total_tasks[j],weight_mat[i,j]))
    G.add_weighted_edges_from(edge_list)
    return G


def plt_rew(file_dir):
    info = torch.load(file_dir)
    eval_res = np.concatenate(info['eval_res'],axis=0)[:,:12]
    tasks = ['tsp20','tsp50','tsp100','cvrp20','cvrp50','cvrp100','op20','op50','op100','kp50','kp100','kp200','tsp200','tsp500','cvrp200','cvrp500','op200','op500','kp500','kp1000']
    reward = np.array(info['rewards'])
    choices = np.array(info['choices'])
    unique_choice = np.unique(choices)
    linestyles = ['-','-','-','-.','-.','-.',':',':',':','--','--','--']
    plt.figure(figsize=(8*5,5*5))
    for choice in unique_choice:
        # rew = judge.copy()
        # rew[choices!=choice] = 0
        plt.subplot(4,3,choice+1)
        rew = eval_res[choices==choice]
        for r in range(rew.shape[1]):
            plt.plot(np.cumsum(rew[:,r]), linestyle=linestyles[r],label='arm {} for task {}'.format(tasks[choice],tasks[r]))
            plt.legend()
    plt.show()


    pass


def plot_graph(G):
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.circular_layout(G)
    nx.draw_spring(G,with_labels=True)
    plt.show()


if __name__=="__main__":
    file_dirs = [
        '/home/wangchenguang/code/UniversalSolver/POMOSolver/result/_train_TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]_BanditAlg-cusumucb_unseen-TSP[200, 500]-CVRP[200, 500]-OP[200, 500]-KP[500, 1000]_desc-cusumucb_strategy/checkpoint-latest.pt',
        '/home/wangchenguang/code/UniversalSolver/POMOSolver/result/_train_TSP[20, 50]-CVRP[20, 50]-OP[20]-KP[50]_BanditAlg-cumucb_unseen-TSP[21]-CVRP[21]-OP[21]-KP[51]_desc-test-valid-nogap-debug/checkpoint-latest.pt',
        '/home/wangchenguang/code/UniversalSolver/POMOSolver/result/_train_TSP[20, 50]-CVRP[20, 50]_BanditAlg-cumucb_unseen-TSP[100, 200]-CVRP[100, 200]_desc-4gpu-cumucb/checkpoint-latest.pt',
    '/home/wangchenguang/code/UniversalSolver/POMOSolver/result/_train_TSP[20, 50]-CVRP[20, 50]_BanditAlg-cumucb_unseen-TSP[100, 200]-CVRP[100, 200]_desc-test-4gpu-gap/checkpoint-latest.pt',
        '/home/wangchenguang/code/UniversalSolver/POMOSolver/result/_train_TSP[20, 50]-CVRP[20, 50]_BanditAlg-phtucb_unseen-TSP[100, 200]-CVRP[100, 200]_desc-test-4gpu-gap/checkpoint-latest.pt'
    ]

    # info = torch.load(file_dirs[0])
    # reward = info['rewards']
    # eval_res = np.concatenate(info['eval_res'],axis=0)[:,:6]
    # con = np.concatenate([np.array(reward).reshape(-1,1),eval_res.sum(-1).reshape(-1,1)],axis=-1)
    # plt.figure()
    # # plt.plot(np.cumsum(reward),label='reward')
    # plt.plot((eval_res.sum(-1)),label='eval_res')
    # plt.legend()
    # plt.show()

    # for dir in file_dirs:
    #     res_dic, loss = read_score_loss(dir)
    #     plot_fig(res_dic,log_flag=False)
    # res_dic, loss = read_score_loss(file_dirs[0])
    # plot_fig(res_dic, log_flag=False)
    # G = read_improvement_graph(file_dirs[0])
    # plot_graph(G)
    plt_rew(file_dirs[0])
    pass


