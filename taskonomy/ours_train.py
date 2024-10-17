import argparse
import itertools
import os
import shutil
import time
import platform

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets


from taskonomy_losses import *
from taskonomy_loader import TaskonomyLoader

import torch.distributed as dist
import torch.multiprocessing as mp

import copy
import numpy as np
import signal
import sys
import math
from collections import defaultdict
import scipy.stats

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from rich.progress import track

import model_definitions as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Taskonomy Training')
parser.add_argument('--data_dir', '-d', dest='data_dir',required=True,
                    help='path to training set')
parser.add_argument('--arch', '-a', metavar='ARCH',required=True,
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (required)')
parser.add_argument('-b', '--batch_size', default=96, type=int,
                    help='mini-batch size (default: 96)')
parser.add_argument('--tasks', '-ts', default='sdnkt', dest='tasks',
                    help='which tasks to train on')
parser.add_argument('--model_dir', default='saved_models', dest='model_dir',
                    help='where to save models')
parser.add_argument('--image_size', default=256, type=int,
                    help='size of image side (images are square)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-pf', '--print_frequency', default=1, type=int,
                    help='how often to print output')
parser.add_argument('--epochs', default=100, type=int,
                    help='maximum number of epochs to run')
parser.add_argument('-mlr', '--minimum_learning_rate', default=3e-5, type=float,
                    metavar='LR', help='End trianing when learning rate falls below this value.')

parser.add_argument('-lr', '--learning_rate',dest='lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-ltw0', '--loss_tracking_window_initial', default=500000, type=int,
                    help='inital loss tracking window (default: 500000)')
parser.add_argument('-mltw', '--maximum_loss_tracking_window', default=2000000, type=int,
                    help='maximum loss tracking window (default: 2000000)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '-wd','--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume','--restart', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--start-epoch', default=0, type=int,
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('-n','--experiment_name', default='', type=str,
                    help='name to prepend to experiment saves.')
parser.add_argument('-v', '--validate', dest='validate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='evaluate model on test set')

parser.add_argument('-r', '--rotate_loss', dest='rotate_loss', action='store_true',
                    help='should loss rotation occur')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('-vb', '--virtual-batch-multiplier', default=1, type=int,
                    metavar='N', help='number of forward/backward passes per parameter update')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('-sbn', '--sync_batch_norm', action='store_true',
                    help='sync batch norm parameters accross gpus.')
parser.add_argument('-hs', '--half_sized_output', action='store_true',
                    help='output 128x128 rather than 256x256.')
parser.add_argument('-na','--no_augment', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('-ml', '--model_limit', default=None, type=int,
                    help='Limit the number of training instances from a single 3d building model.')
parser.add_argument('-tw', '--task_weights', default=None, type=str,
                    help='a comma separated list of numbers one for each task to multiply the loss by.')

cudnn.benchmark = False


def main():
    set_seed(42)
    args = parser.parse_args()
    config = ProjectConfiguration(project_dir=args.model_dir, logging_dir='logs')
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        log_with='aim',
        project_config=config
    )
    accelerator.init_trackers(project_name=args.model_dir, config=vars(args))
    global print
    print = accelerator.print
    print(args)
    print('starting on', platform.node())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('cuda gpus:',os.environ['CUDA_VISIBLE_DEVICES'])

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
        print('Got fp16!')
    
    taskonomy_loss, losses, criteria, taskonomy_tasks = get_losses_and_tasks(args)

    print("including the following tasks:", list(losses.keys()))

    criteria2={'Loss':taskonomy_loss}
    for key,value in criteria.items():
        criteria2[key]=value
    criteria = criteria2

    print('data_dir =',args.data_dir, len(args.data_dir))
    
    if args.no_augment:
        augment = False
    else:
        augment = True
    train_dataset = TaskonomyLoader(
        args.data_dir,
        label_set=taskonomy_tasks,
        model_whitelist='train_models.txt',
        model_limit=args.model_limit,
        output_size = (args.image_size,args.image_size),
        half_sized_output=args.half_sized_output,
        augment=augment)

    print('Found',len(train_dataset),'training instances.')

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](tasks=losses.keys(),half_sized_output=args.half_sized_output)
    backup_model = models.__dict__[args.arch](tasks=losses.keys(),half_sized_output=args.half_sized_output)

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            #print(p.size())
            nn=1
            for s in list(p.size()):
                
                nn = nn*s
            pp += nn
        return pp

    print("Model has", get_n_params(model), "parameters")
    try:
        print("Encoder has", get_n_params(model.encoder), "parameters")
        #flops, params=get_model_complexity_info(model.encoder,(3,256,256), as_strings=False, print_per_layer_stat=False)
        #print("Encoder has", flops, "Flops and", params, "parameters,")
    except:
        print("Each encoder has", get_n_params(model.encoders[0]), "parameters")
    for decoder in model.task_to_decoder.values():
        print("Decoder has", get_n_params(decoder), "parameters")


    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    backup_optimizer = torch.optim.SGD(backup_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.fp16:
        raise NotImplementedError


    # optionally resume from a checkpoint
    checkpoint=None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.pretrained != '':
        print('loading pretrained weights for '+args.arch+' ('+args.pretrained+')')
        model.encoder.load_state_dict(torch.load(args.pretrained))

    if args.resume:
        if os.path.isfile(args.resume) and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = get_eval_loader(args.data_dir, taskonomy_tasks, args)

    model, optimizer, backup_model, backup_optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, backup_model, backup_optimizer, train_loader, val_loader
    )
    
    trainer=Trainer(accelerator,train_loader,val_loader,model,optimizer,backup_model,backup_optimizer,criteria,args,losses.keys(),checkpoint)
    if args.validate:
        trainer.progress_table=[]
        trainer.validate([{}])
        print()
        return
    

    if args.test:
        trainer.progress_table=[]
        # replace val loader with a loader that loads test data
        trainer.val_loader=get_eval_loader(args.data_dir, taskonomy_tasks, args, model_limit=(1000,2000))
        trainer.validate([{}])
        return
    
    trainer.train()
   

def get_eval_loader(datadir, label_set, args,model_limit=1000):
    print(datadir)

    val_dataset = TaskonomyLoader(datadir,
                                  label_set=label_set,
                                  model_whitelist='val_models.txt',
                                  model_limit=model_limit,
                                  output_size = (args.image_size,args.image_size),
                                  half_sized_output=args.half_sized_output,
                                  augment=False)
    print('Found',len(val_dataset),'validation instances.')
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(args.batch_size//2,1), shuffle=False,
        num_workers=args.workers, pin_memory=True,sampler=None)
    return val_loader

program_start_time = time.time()

def on_keyboared_interrupt(x,y):
    #print()
    sys.exit(1)
signal.signal(signal.SIGINT, on_keyboared_interrupt)

def get_average_learning_rate(optimizer):
    try:
        return optimizer.learning_rate
    except:
        s = 0
        for param_group in optimizer.param_groups:
            s+=param_group['lr']
        return s/len(optimizer.param_groups)

def get_combinations(tasks):
    combinations = []
    for num_tasks in [1,2]:
        for c in itertools.combinations_with_replacement(tasks, num_tasks):
            combinations.append(c)
    return combinations

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def print_table(table_list, go_back=True):
    if len(table_list)==0:
        print()
        print()
        return
    if go_back:
        print("\033[F",end='')
        print("\033[K",end='')
        for i in range(len(table_list)):
            print("\033[F",end='')
            print("\033[K",end='')


    lens = defaultdict(int)
    for i in table_list:
        for ii,to_print in enumerate(i):
            for title,val in to_print.items():
                lens[(title,ii)]=max(lens[(title,ii)],max(len(title),len(val)))
    

    # printed_table_list_header = []
    for ii,to_print in enumerate(table_list[0]):
        for title,val in to_print.items():

            print('{0:^{1}}'.format(title,lens[(title,ii)]),end=" ")
    for i in table_list:
        print()
        for ii,to_print in enumerate(i):
            for title,val in to_print.items():
                print('{0:^{1}}'.format(val,lens[(title,ii)]),end=" ",flush=True)
    print()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.std= 0
        self.sum = 0
        self.sumsq = 0
        self.count = 0
        self.lst = []

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        #self.sumsq += float(val)**2
        self.count += n
        self.avg = self.sum / self.count
        self.lst.append(self.val)
        self.std=np.std(self.lst)


class Trainer:
    def __init__(self,accelerator:Accelerator,train_loader,val_loader,model,optimizer,backup_model,backup_optimizer,criteria,args,task_keys,checkpoint=None):
        self.accelerator = accelerator
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.model=model
        self.optimizer=optimizer
        self.backup_model=backup_model
        self.backup_optimizer=backup_optimizer
        self.criteria=criteria
        self.args = args
        self.task_keys=task_keys
        self.combined_tasks = get_combinations(task_keys)
        self.code_archive=self.get_code_archive()
        if checkpoint:
            if 'progress_table' in checkpoint:
                self.progress_table = checkpoint['progress_table']
            else:
                self.progress_table=[]    
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch']+1
            else:
                self.start_epoch = 0
            if 'best_loss' in checkpoint:
                self.best_loss = checkpoint['best_loss']
            else:
                self.best_loss = 9e9
            if 'stats' in checkpoint:
                self.stats = checkpoint['stats']
            else:
                self.stats=[]
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
            else:
                self.loss_history=[]
            if 'transfer_gains' in checkpoint:
                self.transfer_gains = checkpoint['transfer_gains']
            else:
                self.transfer_gains = {
                    combined_task: {task: [] for task in task_keys} for combined_task in self.combined_tasks
                }
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            else:
                self.train_losses = {task: [] for task in task_keys}

        else:
            self.progress_table=[]
            self.best_loss = 9e9
            self.stats = []
            self.start_epoch = 0
            self.loss_history=[]
            self.transfer_gains = {
                combined_task: {task: [] for task in task_keys} for combined_task in self.combined_tasks
            }
            self.train_losses = {task: [] for task in task_keys}
        
        self.lr0 = get_average_learning_rate(optimizer)
            
        print_table(self.progress_table,False)
        self.ticks=0
        self.last_tick=0
        self.loss_tracking_window = args.loss_tracking_window_initial

    def get_code_archive(self):
        file_contents={}
        for i in os.listdir('.'):
            if i[-3:]=='.py':
                with open(i,'r') as file:
                    file_contents[i]=file.read()
        return file_contents

    def train(self):
        for self.epoch in range(self.start_epoch,self.args.epochs):
            current_learning_rate = get_average_learning_rate(self.optimizer)
            if current_learning_rate < self.args.minimum_learning_rate:
                break
            # train for one epoch
            train_string, train_stats = self.train_epoch()

            # evaluate on validation set
            progress_string=train_string
            loss, progress_string, val_stats = self.validate(progress_string)
            print()

            self.progress_table.append(progress_string)

            self.stats.append((train_stats,val_stats))
            if self.accelerator.is_main_process:
                self.checkpoint(loss)

    def checkpoint(self, loss):
        is_best = loss < self.best_loss
        self.best_loss = min(loss, self.best_loss)
        save_filename = str(self.epoch) + '_' + self.args.experiment_name+'_'+self.args.arch+'_'+('p' if self.args.pretrained != '' else 'np')+'_'+self.args.tasks+'_checkpoint.pth.tar'

        try:
            to_save = self.model
            if torch.cuda.device_count() >1:
                to_save=to_save.module
            gpus='all'
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpus=os.environ['CUDA_VISIBLE_DEVICES']
            self.save_checkpoint({
                'epoch': self.epoch,
                'info':{'machine':platform.node(), 'GPUS':gpus},
                'args': self.args,
                'arch': self.args.arch,
                'state_dict': to_save.state_dict(),
                'best_loss': self.best_loss,
                'optimizer' : self.optimizer.state_dict(),
                'progress_table' : self.progress_table,
                'stats': self.stats,
                'loss_history': self.loss_history,
                'code_archive':self.code_archive,
                'transfer_gains':self.transfer_gains,
                'train_losses':self.train_losses
            }, False, self.args.model_dir, save_filename)

            if is_best:
                self.save_checkpoint(None, True,self.args.model_dir, save_filename)
        except:
            print('save checkpoint failed...')



    def save_checkpoint(self,state, is_best,directory='', filename='checkpoint.pth.tar'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory,filename)
        if is_best:
            best_path = os.path.join(directory,'best_'+filename)
            shutil.copyfile(path, best_path)
        else:
            torch.save(state, path)

    def learning_rate_schedule(self):
        ttest_p=0
        z_diff=0

        #don't reduce learning rate until the second epoch has ended
        if self.epoch < 2:
            return 0,0
        
        wind=self.loss_tracking_window//(self.args.batch_size)
        if len(self.loss_history)-self.last_tick > wind:
            a = self.loss_history[-wind:-wind*5//8]
            b = self.loss_history[-wind*3//8:]
            #remove outliers
            a = sorted(a)
            b = sorted(b)
            a = a[int(len(a)*.05):int(len(a)*.95)]
            b = b[int(len(b)*.05):int(len(b)*.95)]
            length_=min(len(a),len(b))
            a=a[:length_]
            b=b[:length_]
            z_diff,ttest_p = scipy.stats.ttest_rel(a,b,nan_policy='omit')

            if z_diff < 0 or ttest_p > .99:
                self.ticks+=1
                self.last_tick=len(self.loss_history)
                self.adjust_learning_rate()
                self.loss_tracking_window = min(self.args.maximum_loss_tracking_window,self.loss_tracking_window*2)
        return ttest_p, z_diff

    def train_epoch(self):
        global program_start_time
        average_meters = defaultdict(AverageMeter)
        display_values = []
        for name,func in self.criteria.items():
            display_values.append(name)

        # switch to train mode
        self.model.train()

        end = time.time()
        epoch_start_time = time.time()
        epoch_start_time2=time.time()

        batch_num = 0
        num_data_points=len(self.train_loader)
        if num_data_points > 10000:
            num_data_points = num_data_points//5
            
        starting_learning_rate=get_average_learning_rate(self.optimizer)
        for input, target in self.train_loader:
            if batch_num ==0:
                epoch_start_time2=time.time()
            if num_data_points==batch_num:
                break
            self.percent = batch_num/num_data_points

            data_start = time.time()

            average_meters['data_time'].update(time.time() - data_start)
            loss_dict, loss = self.train_batch(input,target)
            # do the weight updates and set gradients back to zero
            self.update()

            self.loss_history.append(float(loss))
            ttest_p, z_diff = self.learning_rate_schedule()

            for name,value in loss_dict.items():
                try:
                    average_meters[name].update(value.data)
                except:
                    average_meters[name].update(value)

            elapsed_time_for_epoch = (time.time()-epoch_start_time2)
            eta = (elapsed_time_for_epoch/(batch_num+.2))*(num_data_points-batch_num)
            if eta >= 24*3600:
                eta = 24*3600-1

            batch_num+=1
            current_learning_rate= get_average_learning_rate(self.optimizer)
            if True:
                to_print = {}
                to_print['ep']= ('{0}:').format(self.epoch)
                to_print['#/{0}'.format(num_data_points)]= ('{0}').format(batch_num)
                to_print['lr']= ('{0:0.3g}-{1:0.3g}').format(starting_learning_rate,current_learning_rate)
                to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
                
                to_print['d%']=('{0:0.2g}').format(100*average_meters['data_time'].sum/elapsed_time_for_epoch)
                for name in display_values:
                    meter = average_meters[name]
                    to_print[name]= ('{meter.avg:.4g}').format(meter=meter)
                if batch_num < num_data_points-1:
                    to_print['ETA']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta+elapsed_time_for_epoch))))
                    to_print['ttest']= ('{0:0.3g},{1:0.3g}').format(z_diff,ttest_p)
                if batch_num % self.args.print_frequency == 0:
                        print_table(self.progress_table+[[to_print]])

            log_dict = {}
            for name in display_values:
                log_dict["train/"+name] = loss_dict[name].mean().clone().detach().cpu().numpy()
            log_dict['train/Loss'] = loss.clone().detach().cpu().numpy()
            log_dict['learning_rate'] = current_learning_rate
            log_dict['ttest_p'] = ttest_p
            log_dict['z_diff'] = z_diff
            self.accelerator.log(log_dict)

        epoch_time = time.time()-epoch_start_time
        stats={'batches':num_data_points,
            'learning_rate':current_learning_rate,
            'Epoch time':epoch_time,
            }
        for name in display_values:
            meter = average_meters[name]
            stats[name] = meter.avg
        log_dict = {}
        for k, v in stats.items():
            log_dict["train/epoch/"+k] = v
        self.accelerator.log(log_dict)
        data_time = average_meters['data_time'].sum

        to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
        
        return [to_print], stats



    def train_batch(self, input, target):
        self.lookahead_batch(input, target)

        loss_dict = {}
        
        input = input.float()
        output = self.model(input)
        first_loss=None
        for c_name,criterion_fun in self.criteria.items():
            if first_loss is None:first_loss=c_name
            loss_dict[c_name]=criterion_fun(output, target).clone()
        loss = loss_dict[first_loss].clone()
        loss = loss
        
        reduced_loss_dict = self.accelerator.reduce(loss_dict, reduction='mean')
        
        if self.accelerator.is_main_process:
            for task in self.task_keys:
                # record the loss on the main device
                self.train_losses[task].append(reduced_loss_dict[task].mean().clone().detach().cpu().numpy())

        if self.args.fp16:
            raise NotImplementedError
        else:
            self.accelerator.backward(loss)

        return loss_dict, loss
    
    def assign_grad(self, model, grads):
        for param, grad in zip(model.parameters(), grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad

    def lookahead_batch(self, input, target):
        # Backup the model and optimizer
        self.backup_model.load_state_dict(self.model.state_dict())
        self.backup_model.train()
        loss_dict = {}

        input = input.float()
        output = self.backup_model(input)
        for c_name,criterion_fun in self.criteria.items():
            loss_dict[c_name]=criterion_fun(output, target).clone()
        reduced_loss_dict = self.accelerator.reduce(loss_dict, reduction='mean')
        grads_dict = {}
        for task in self.task_keys:
            grads_dict[task] = torch.autograd.grad(reduced_loss_dict[task], self.backup_model.parameters(), retain_graph=True, materialize_grads=True)

        for combined_task in self.combined_tasks:
            # update model
            self.backup_model.load_state_dict(self.model.state_dict())
            self.backup_optimizer.load_state_dict(self.optimizer.state_dict())
            self.backup_optimizer.zero_grad()
            for task in combined_task:
                self.assign_grad(self.backup_model, grads_dict[task])
            self.backup_optimizer.step()
            # loss_dict = {}
            # input = input.float()
            # output = self.backup_model(input)
            # for c_name,criterion_fun in self.criteria.items():
            #     loss_dict[c_name]=criterion_fun(output, target).clone()

            # # Get the loss for the combined task
            # loss = torch.sum(torch.stack([loss_dict[task] for task in combined_task]))
                
            # if self.args.fp16:
            #     raise NotImplementedError
            # else:
            #     self.accelerator.backward(loss)
            # self.backup_optimizer.step()
            # self.backup_optimizer.zero_grad()
            # Calculate the loss after updating the model
            new_loss_dict = {}
            output = self.backup_model(input)
            for c_name,criterion_fun in self.criteria.items():
                new_loss_dict[c_name]=criterion_fun(output, target).clone()

            reduced_loss_dict = self.accelerator.reduce(new_loss_dict, reduction='mean')

            if self.accelerator.is_main_process:
                for task in self.task_keys:
                    # Calculate the average loss and record it
                    self.transfer_gains[combined_task][task].append(reduced_loss_dict[task].mean().clone().detach().cpu().numpy())
        

        return


    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


    def validate(self, train_table):
        average_meters = defaultdict(AverageMeter)
        self.model.eval()
        epoch_start_time = time.time()
        batch_num=0
        num_data_points=len(self.val_loader)

        torch.cuda.empty_cache()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_loader):
                if batch_num ==0:
                    epoch_start_time2=time.time()

                output = self.model(input)
                

                loss_dict = {}
                
                for c_name,criterion_fun in self.criteria.items():
                    loss_dict[c_name]=criterion_fun(output, target)
                
                batch_num=i+1

                for name,value in loss_dict.items():    
                    try:
                        average_meters[name].update(value.data)
                    except:
                        average_meters[name].update(value)
                eta = ((time.time()-epoch_start_time2)/(batch_num+.2))*(len(self.val_loader)-batch_num)

                to_print = {}
                to_print['#/{0}'.format(num_data_points)]= ('{0}').format(batch_num)
                to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
                for name in self.criteria.keys():
                    meter = average_meters[name]
                    to_print[name]= ('{meter.avg:.4g}').format(meter=meter)
                progress=train_table+[to_print]
                if batch_num % self.args.print_frequency == 0:
                    print_table(self.progress_table+[progress])
                
                log_dict = {}
                for name,value in loss_dict.items():
                    log_dict["val/"+name] = value.mean().clone().detach().cpu().numpy()
                self.accelerator.log(log_dict)

        epoch_time = time.time()-epoch_start_time

        stats={'batches':len(self.val_loader),
            'Epoch time':epoch_time,
            }
        ultimate_loss = None
        for name in self.criteria.keys():
            meter = average_meters[name]
            stats[name]=meter.avg
        ultimate_loss = stats['Loss']
        to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
        torch.cuda.empty_cache()
        log_dict = {}
        for k, v in stats.items():
            log_dict["val/epoch/"+k] = v
        self.accelerator.log(log_dict)
        return float(ultimate_loss), progress , stats

    def adjust_learning_rate(self):
        self.lr = self.lr0 * (0.50 ** (self.ticks))
        self.set_learning_rate(self.lr)

    def set_learning_rate(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    #mp.set_start_method('forkserver')
    main()
