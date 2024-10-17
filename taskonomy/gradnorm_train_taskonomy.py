from collections import defaultdict
import copy
import json
import os
import platform
import tempfile
import shutil
import signal
import sys
import time

from google.cloud import storage
from absl import app
from absl import flags
from absl import logging
import model_definitions as models
import numpy as np
import scipy.stats
from taskonomy_loader import TaskonomyLoader
from taskonomy_losses import *
import torch
import torch.backends.cudnn as cudnn

# from ptflops import get_model_complexity_info


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__') and
                     callable(models.__dict__[name]))

FLAGS = flags.FLAGS
flags.DEFINE_bool('fp16', True, 'Run model fp16 mode.')
flags.DEFINE_string('tasks', 'sdnkt', 'which tasks to train on')
flags.DEFINE_bool('rotate_loss', True, 'should loss rotation occur')
flags.DEFINE_string('data_dir', './taskonomy_data', 'data directory')
flags.DEFINE_string('arch', 'xception_taskonomy_new', 'model architecture')
flags.DEFINE_float('lr', 0.1, 'initial learning rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_float('weight_decay', 1e-4, 'weight decay (default: 1e-4)')
flags.DEFINE_string('resume', '', 'path to latest checkpoint (default: none)')
flags.DEFINE_string('pretrained', '', 'use pre-trained model')
flags.DEFINE_bool('sbn', True, 'sync batch norm paramters across gpus.')
flags.DEFINE_integer('virtual_batch_multiplier', 1, 'number of forward/backward passes per parameter update')
flags.DEFINE_integer('workers', 16, 'number of data loading workers')
flags.DEFINE_integer('batch_size', 96, 'batch size')
flags.DEFINE_string('model_dir', 'saved_models', 'where to save models')
flags.DEFINE_integer('epochs', 100, 'maximum number of epochs to train')
flags.DEFINE_float('minimum_learning_rate', 3e-5, 'end training when the learning rate falls below this value.')
flags.DEFINE_string('experiment_name', '', 'name to prepend to experiment saves.')
flags.DEFINE_integer('maximum_loss_tracking_window', 2000000, 'maximum loss tracking window (default: 2000000)')
flags.DEFINE_integer('print_frequency', 1, 'print frequency')
flags.DEFINE_integer('loss_tracking_window_initial', 500000, 'initial loss tracking window')
flags.DEFINE_bool('validate', False, 'evaluate model on validation set')
flags.DEFINE_bool('test', False, 'evaluate model on test set')

cudnn.benchmark = False


def main(_):
  logging.info('starting on %s', platform.node())
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    logging.info('cuda gpus: %s', os.environ['CUDA_VISIBLE_DEVICES'])

  main_stream = torch.cuda.Stream()

  if FLAGS.fp16:
    assert torch.backends.cudnn.enabled, ('fp16 mode requires cudnn backend to '
                                          'be enabled.')
    logging.info('Got fp16!')

  taskonomy_loss, losses, criteria, taskonomy_tasks = get_losses_and_tasks(FLAGS)

  logging.info('including the following tasks: %s', list(losses.keys()))

  criteria2 = {'Loss': taskonomy_loss}
  for key, value in criteria.items():
    criteria2[key] = value
  criteria = criteria2

  logging.info('data_dir =%s %d', FLAGS.data_dir, len(FLAGS.data_dir))
  augment = True

  train_dataset = TaskonomyLoader(
      FLAGS.data_dir,
      label_set=taskonomy_tasks,
      model_whitelist='train_models.txt',
      model_limit=None,
      output_size=(256, 256),
      half_sized_output=False,
      augment=augment)

  logging.info('Found %s training instances.', len(train_dataset))

  logging.info("=> creating model '%s'", FLAGS.arch)
  model = models.__dict__[FLAGS.arch](
      tasks=losses.keys(), half_sized_output=False, ozan='gradnorm')

  model = model.cuda()

  optimizer = torch.optim.SGD(
      model.parameters(),
      FLAGS.lr,
      momentum=FLAGS.momentum,
      weight_decay=FLAGS.weight_decay)

  # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
  # for convenient interoperation with argparse.
  if FLAGS.fp16:
    model, optimizer = amp.initialize(
        model, optimizer, opt_level='O1', loss_scale='dynamic', verbosity=0)
    logging.info('Got fp16!')

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model).cuda()
    if FLAGS.sbn:
      from taskonomy_gpu_pytorch.sync_batchnorm import patch_replication_callback
      patch_replication_callback(model)

  logging.info('Virtual batch size = %d', FLAGS.batch_size * FLAGS.virtual_batch_multiplier)

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      num_workers=FLAGS.workers,
      pin_memory=True,
      sampler=None)
  val_loader = get_eval_loader(FLAGS.data_dir, taskonomy_tasks)
  test_loader = get_test_loader(FLAGS.data_dir, taskonomy_tasks)

  trainer = Trainer(train_loader, val_loader, test_loader, model, optimizer,
                    criteria, None)
  trainer.train()


def get_eval_loader(datadir, label_set, model_limit=1000):
  logging.info(datadir)
  val_dataset = TaskonomyLoader(
      datadir,
      label_set=label_set,
      model_whitelist='val_models.txt',
      model_limit=model_limit,
      output_size=(256, 256),
      half_sized_output=False,
      augment=False)
  logging.info('Found %d validation instances', len(val_dataset))

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=max(FLAGS.batch_size // 2, 1),
      shuffle=False,
      num_workers=FLAGS.workers,
      pin_memory=True,
      sampler=None)
  return val_loader


def get_test_loader(datadir, label_set, model_limit=1000):
  logging.info(datadir)
  val_dataset = TaskonomyLoader(
      datadir,
      label_set=label_set,
      model_whitelist='test_models.txt',
      model_limit=model_limit,
      output_size=(256, 256),
      half_sized_output=False,
      augment=False)
  logging.info('Found %d test instances', len(val_dataset))

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=max(FLAGS.batch_size // 2, 1),
      shuffle=False,
      num_workers=FLAGS.workers,
      pin_memory=True,
      sampler=None)
  return val_loader


program_start_time = time.time()


def on_keyboared_interrupt(x, y):
  sys.exit(1)


signal.signal(signal.SIGINT, on_keyboared_interrupt)


def get_average_learning_rate(optimizer):
  try:
    return optimizer.learning_rate
  except:
    s = 0
    for param_group in optimizer.param_groups:
      s += param_group['lr']
    return s / len(optimizer.param_groups)


class data_prefetcher():

  def __init__(self, loader):
    self.inital_loader = loader
    self.loader = iter(loader)
    self.stream = torch.cuda.Stream()
    self.preload()

  def preload(self):
    try:
      self.next_input, self.next_target = next(self.loader)
    except StopIteration:
      # self.next_input = None
      # self.next_target = None
      self.loader = iter(self.inital_loader)
      self.preload()
      return
    with torch.cuda.stream(self.stream):
      self.next_input = self.next_input.cuda(non_blocking=True)
      # self.next_target = self.next_target.cuda(async=True)
      self.next_target = {
          key: val.cuda(non_blocking=True)
          for (key, val) in self.next_target.items()
      }

  def next(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    input = self.next_input
    target = self.next_target
    self.preload()
    return input, target


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
  if len(table_list) == 0:
    return
  if go_back:
    print_str = ''
    print_str += '\033[F'
    print_str += '\033[K'
    for i in range(len(table_list)):
      print_str += '\033[F'
      print_str += '\033[K'

  lens = defaultdict(int)
  for i in table_list:
    for ii, to_print in enumerate(i):
      for title, val in to_print.items():
        lens[(title, ii)] = max(lens[(title, ii)], max(len(title), len(val)))

  # printed_table_list_header = []
  for ii, to_print in enumerate(table_list[0]):
    for title, val in to_print.items():
      tmp_str = '{0:^{1}} '.format(title, lens[(title, ii)])
      print_str += tmp_str
  for i in table_list:
    for ii, to_print in enumerate(i):
      for title, val in to_print.items():
        tmp_str = '{0:^{1}} '.format(val, lens[(title, ii)])
        print_str += tmp_str
  logging.info(print_str)


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.std = 0
    self.sum = 0
    self.sumsq = 0
    self.count = 0
    self.lst = []

  def update(self, val, n=1):
    self.val = float(val)
    self.sum += float(val) * n
    # self.sumsq += float(val)**2
    self.count += n
    self.avg = self.sum / self.count
    self.lst.append(self.val)
    self.std = np.std(self.lst)


class Trainer:

  def __init__(self,
               train_loader,
               val_loader,
               test_loader,
               model,
               optimizer,
               criteria,
               checkpoint=None):
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.train_prefetcher = data_prefetcher(self.train_loader)
    self.model = model
    self.optimizer = optimizer
    self.criteria = criteria
    self.fp16 = FLAGS.fp16
    self.code_archive = self.get_code_archive()
    self.loss_keys = ['ss_l', 'norm_l', 'depth_l', 'key_l', 'edge2d_l']
    self.progress_table = []
    self.best_loss = 9e9
    self.stats = []
    self.start_epoch = 0
    self.loss_history = []
    self.lr0 = get_average_learning_rate(optimizer)
    print_table(self.progress_table, False)
    self.ticks = 0
    self.last_tick = 0
    self.loss_tracking_window = FLAGS.loss_tracking_window_initial

  def get_code_archive(self):
    file_contents = {}
    for i in os.listdir('.'):
      if i[-3:] == '.py':
        with open(i, 'r') as file:
          file_contents[i] = file.read()
    return file_contents

  def train(self):
    best_val = 1000.
    train_dict = {epoch: {} for epoch in range(self.start_epoch, FLAGS.epochs)}
    valid_dict = {epoch: {} for epoch in range(self.start_epoch, FLAGS.epochs)}
    test_dict = {epoch: {} for epoch in range(self.start_epoch, FLAGS.epochs)}
    for self.epoch in range(self.start_epoch, FLAGS.epochs):
      current_learning_rate = get_average_learning_rate(self.optimizer)
      if current_learning_rate < FLAGS.minimum_learning_rate:
        break
      # train for one epoch and record training stats in train_dict
      train_string, train_stats = self.train_epoch()

      # evaluate on validation set
      train_progress_string = train_string
      loss, progress_string, valid_stats = self.validate(train_progress_string)

      loss_names = ['ss_l', 'depth_l', 'norm_l', 'key_l', 'edge2d_l']
      for loss_name in loss_names:
        if loss_name in train_stats:
          train_dict[self.epoch][loss_name] = valid_stats[loss_name]
          valid_dict[self.epoch][loss_name] = valid_stats[loss_name]
        else:
          train_dict[self.epoch][loss_name] = 0.
          valid_dict[self.epoch][loss_name] = 0.

      valid_total = sum(
          [valid_dict[self.epoch][loss_name] for loss_name in loss_names])
      valid_loss_str = (
          'Valid Metrics: epoch: {} || total: {:.4f} || ss_l: {:.4f} || '
          'depth_l: {:.4f} || norm_l: {:.4f} || key_l: {:.4f} || '
          'edge2d_l: {:.4f}'.format(self.epoch, valid_total,
                                    valid_dict[self.epoch]['ss_l'],
                                    valid_dict[self.epoch]['depth_l'],
                                    valid_dict[self.epoch]['norm_l'],
                                    valid_dict[self.epoch]['key_l'],
                                    valid_dict[self.epoch]['edge2d_l']))
      logging.info(valid_loss_str)

      if valid_total < best_val:
        best_val = valid_total
        _, _, test_stats = self.test(train_progress_string)
        for loss_name in loss_names:
          if loss_name in test_stats:
            test_dict[self.epoch][loss_name] = test_stats[loss_name]
          else:
            test_dict[self.epoch][loss_name] = 0.
        test_total = sum(
            [test_dict[self.epoch][loss_name] for loss_name in loss_names])
        test_loss_str = (
            'Test Metrics: epoch: {} || total: {:.4f} || ss_l: {:.4f} || '
            'depth_l: {:.4f} || norm_l: {:.4f} || key_l: {:.4f} || '
            'edge2d_l: {:.4f}'.format(self.epoch, test_total,
                                      test_dict[self.epoch]['ss_l'],
                                      test_dict[self.epoch]['depth_l'],
                                      test_dict[self.epoch]['norm_l'],
                                      test_dict[self.epoch]['key_l'],
                                      test_dict[self.epoch]['edge2d_l']))
        logging.info(test_loss_str)

      self.progress_table.append(progress_string)
      self.stats.append((train_stats, valid_stats))

    tmp_path = tempfile.mkdtemp()

    # Upload train/valid/test stats to gcs.
    with open(os.path.join(tmp_path, 'train_dict.json'), 'w') as f:
      f.write(json.dumps(train_dict))
    with open(os.path.join(tmp_path, 'valid_dict.json'), 'w') as f:
      f.write(json.dumps(valid_dict))
    with open(os.path.join(tmp_path, 'test_dict.json'), 'w') as f:
      f.write(json.dumps(test_dict))

    # Saves the pytorch model.
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
    }, os.path.join(tmp_path, 'final_model.tar'))

    # Upload saved model and training stats to gcs.
    storage_client = storage.Client()
    gcs_bucket = 'xcloud_public_bucket'
    output_dir = f'cfifty/taskonomy_data/gradnorm_{FLAGS.tasks}_taskonomy'
    bucket = storage_client.bucket(gcs_bucket)
    for dirpath, _, filenames in os.walk(tmp_path):
      for name in filenames:
        filename = os.path.join(dirpath, name)
        blob = storage.Blob(os.path.join(output_dir, name), bucket)
        with open(filename, 'rb') as f:
          blob.upload_from_file(f)
        logging.info('blob path: %s', blob.path)
        logging.info('bucket path: gs://%s/%s', gcs_bucket, output_dir)

  def learning_rate_schedule(self):
    ttest_p = 0
    z_diff = 0

    # don't reduce learning rate until the second epoch has ended
    if self.epoch < 2:
      return 0, 0

    wind = self.loss_tracking_window // (
        FLAGS.batch_size * FLAGS.virtual_batch_multiplier)
    if len(self.loss_history) - self.last_tick > wind:
      a = self.loss_history[-wind:-wind * 5 // 8]
      b = self.loss_history[-wind * 3 // 8:]
      # remove outliers
      a = sorted(a)
      b = sorted(b)
      a = a[int(len(a) * .05):int(len(a) * .95)]
      b = b[int(len(b) * .05):int(len(b) * .95)]
      length_ = min(len(a), len(b))
      a = a[:length_]
      b = b[:length_]
      z_diff, ttest_p = scipy.stats.ttest_rel(a, b, nan_policy='omit')

      if z_diff < 0 or ttest_p > .99:
        self.ticks += 1
        self.last_tick = len(self.loss_history)
        self.adjust_learning_rate()
        self.loss_tracking_window = min(FLAGS.maximum_loss_tracking_window,
                                        self.loss_tracking_window * 2)
    return ttest_p, z_diff

  def train_epoch(self):
    global program_start_time
    average_meters = defaultdict(AverageMeter)
    display_values = []
    for name, func in self.criteria.items():
      display_values.append(name)

    # switch to train mode
    self.model.train()

    end = time.time()
    epoch_start_time = time.time()
    epoch_start_time2 = time.time()

    batch_num = 0
    num_data_points = len(
        self.train_loader) // FLAGS.virtual_batch_multiplier
    if num_data_points > 10000:
      num_data_points = num_data_points // 5

    starting_learning_rate = get_average_learning_rate(self.optimizer)
    while True:
      if batch_num == 0:
        end = time.time()
        epoch_start_time2 = time.time()
      if num_data_points == batch_num:
        break
      self.percent = batch_num / num_data_points
      loss_dict = None
      loss = 0

      # accumulate gradients over multiple runs of input
      for _ in range(FLAGS.virtual_batch_multiplier):
        data_start = time.time()
        input, target = self.train_prefetcher.next()
        average_meters['data_time'].update(time.time() - data_start)
        loss_dict2, loss2 = self.train_batch(input, target)
        loss += loss2
        if loss_dict is None:
          loss_dict = loss_dict2
        else:
          for key, value in loss_dict2.items():
            loss_dict[key] += value

      # divide by the number of accumulations
      loss /= FLAGS.virtual_batch_multiplier
      for key, value in loss_dict.items():
        loss_dict[key] = value / FLAGS.virtual_batch_multiplier

      # do the weight updates and set gradients back to zero
      self.update()

      self.loss_history.append(float(loss))
      ttest_p, z_diff = self.learning_rate_schedule()

      for name, value in loss_dict.items():
        try:
          average_meters[name].update(value.data)
        except:
          average_meters[name].update(value)

      elapsed_time_for_epoch = (time.time() - epoch_start_time2)
      eta = (elapsed_time_for_epoch / (batch_num + .2)) * (
          num_data_points - batch_num)
      if eta >= 24 * 3600:
        eta = 24 * 3600 - 1

      batch_num += 1
      current_learning_rate = get_average_learning_rate(self.optimizer)
      if True:
        to_print = {}
        to_print['ep'] = ('{0}:').format(self.epoch)
        to_print['#/{0}'.format(num_data_points)] = ('{0}').format(batch_num)
        to_print['lr'] = ('{0:0.3g}-{1:0.3g}').format(starting_learning_rate,
                                                      current_learning_rate)
        to_print['eta'] = ('{0}').format(
            time.strftime('%H:%M:%S', time.gmtime(int(eta))))

        to_print['d%'] = ('{0:0.2g}').format(
            100 * average_meters['data_time'].sum / elapsed_time_for_epoch)
        for name in display_values:
          meter = average_meters[name]
          to_print[name] = ('{meter.avg:.4g}').format(meter=meter)
        if batch_num < num_data_points - 1:
          to_print['ETA'] = ('{0}').format(
              time.strftime('%H:%M:%S',
                            time.gmtime(int(eta + elapsed_time_for_epoch))))
          to_print['ttest'] = ('{0:0.3g},{1:0.3g}').format(z_diff, ttest_p)
        # if batch_num % FLAGS.print_frequency == 0:
        #   print_table(self.progress_table + [[to_print]])

    epoch_time = time.time() - epoch_start_time
    stats = {
        'batches': num_data_points,
        'learning_rate': current_learning_rate,
        'Epoch time': epoch_time,
    }
    for name in display_values:
      meter = average_meters[name]
      stats[name] = meter.avg

    to_print['eta'] = ('{0}').format(
        time.strftime('%H:%M:%S', time.gmtime(int(epoch_time))))

    return [to_print], stats

  def train_batch(self, input, target):
    loss_dict = {}
    input = input.float()
    output = self.model(input)
    first_loss = None
    for c_name, criterion_fun in self.criteria.items():
      if first_loss is None:
        first_loss = c_name
      loss_dict[c_name] = criterion_fun(output, target)

    loss = loss_dict[first_loss].clone()
    loss = loss / FLAGS.virtual_batch_multiplier

    losses = [loss_dict[task] for task in loss_dict if task != 'Loss']
    self.model.module.input_per_task_losses(losses)

    if FLAGS.fp16:
      with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        scaled_loss.backward()
    else:
      loss.backward()

    return loss_dict, loss

  def update(self):
    self.optimizer.step()
    self.optimizer.zero_grad()

  def validate(self, train_table):
    average_meters = defaultdict(AverageMeter)
    self.model.eval()
    epoch_start_time = time.time()
    batch_num = 0
    num_data_points = len(self.val_loader)

    prefetcher = data_prefetcher(self.val_loader)
    torch.cuda.empty_cache()
    with torch.no_grad():
      for i in range(len(self.val_loader)):
        input, target = prefetcher.next()

        if batch_num == 0:
          epoch_start_time2 = time.time()

        output = self.model(input)

        loss_dict = {}

        for c_name, criterion_fun in self.criteria.items():
          loss_dict[c_name] = criterion_fun(output, target)

        batch_num = i + 1

        for name, value in loss_dict.items():
          try:
            average_meters[name].update(value.data)
          except:
            average_meters[name].update(value)
        eta = ((time.time() - epoch_start_time2) / (batch_num + .2)) * (
            len(self.val_loader) - batch_num)

        to_print = {}
        to_print['#/{0}'.format(num_data_points)] = ('{0}').format(batch_num)
        to_print['eta'] = ('{0}').format(
            time.strftime('%H:%M:%S', time.gmtime(int(eta))))
        for name in self.criteria.keys():
          meter = average_meters[name]
          to_print[name] = ('{meter.avg:.4g}').format(meter=meter)
        progress = train_table + [to_print]
        # if batch_num % FLAGS.print_frequency == 0:
        #   print_table(self.progress_table + [progress])

    epoch_time = time.time() - epoch_start_time

    stats = {
        'batches': len(self.val_loader),
        'Epoch time': epoch_time,
    }
    ultimate_loss = None
    for name in self.criteria.keys():
      meter = average_meters[name]
      stats[name] = meter.avg
    ultimate_loss = stats['Loss']
    to_print['eta'] = ('{0}').format(
        time.strftime('%H:%M:%S', time.gmtime(int(epoch_time))))
    torch.cuda.empty_cache()
    return float(ultimate_loss), progress, stats

  def test(self, train_table):
    average_meters = defaultdict(AverageMeter)
    self.model.eval()
    epoch_start_time = time.time()
    batch_num = 0
    num_data_points = len(self.test_loader)

    prefetcher = data_prefetcher(self.test_loader)
    torch.cuda.empty_cache()
    with torch.no_grad():
      for i in range(len(self.test_loader)):
        input, target = prefetcher.next()

        if batch_num == 0:
          epoch_start_time2 = time.time()

        output = self.model(input)

        loss_dict = {}

        for c_name, criterion_fun in self.criteria.items():
          loss_dict[c_name] = criterion_fun(output, target)

        batch_num = i + 1

        for name, value in loss_dict.items():
          try:
            average_meters[name].update(value.data)
          except:
            average_meters[name].update(value)
        eta = ((time.time() - epoch_start_time2) / (batch_num + .2)) * (
            len(self.test_loader) - batch_num)

        to_print = {}
        to_print['#/{0}'.format(num_data_points)] = ('{0}').format(batch_num)
        to_print['eta'] = ('{0}').format(
            time.strftime('%H:%M:%S', time.gmtime(int(eta))))
        for name in self.criteria.keys():
          meter = average_meters[name]
          to_print[name] = ('{meter.avg:.4g}').format(meter=meter)
        progress = train_table + [to_print]
        # if batch_num % FLAGS.print_frequency == 0:
        #   print_table(self.progress_table + [progress])

    epoch_time = time.time() - epoch_start_time

    stats = {
        'batches': len(self.test_loader),
        'Epoch time': epoch_time,
    }
    ultimate_loss = None
    for name in self.criteria.keys():
      meter = average_meters[name]
      stats[name] = meter.avg
    ultimate_loss = stats['Loss']
    to_print['eta'] = ('{0}').format(
        time.strftime('%H:%M:%S', time.gmtime(int(epoch_time))))
    torch.cuda.empty_cache()
    return float(ultimate_loss), progress, stats

  def adjust_learning_rate(self):
    self.lr = self.lr0 * (0.50**(self.ticks))
    self.set_learning_rate(self.lr)

  def set_learning_rate(self, lr):
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr


if __name__ == '__main__':
  app.run(main)
