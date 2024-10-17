# %% [markdown]
# ### CelebA Experiments for...
# ## Towards Principled Task Grouping for Multi-Task Learning
#
# Licensed under the Apache License, Version 2.0
#

# %%
import itertools
import pickle
import time
import copy
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from collections import namedtuple, OrderedDict
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    Lambda,
)

import scipy.integrate as it
from absl import flags

# %%
# Adapted from https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py
GATE_OP = 1


class PCGrad(tf.compat.v1.train.Optimizer):
    """PCGrad. https://arxiv.org/pdf/2001.06782.pdf."""

    def __init__(self, opt, use_locking=False, name="PCGrad"):
        """optimizer: the optimizer being wrapped."""
        super(PCGrad, self).__init__(use_locking, name)
        self.optimizer = opt

    def compute_gradients(
        self,
        loss,
        var_list=None,
        gate_gradients=GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        grad_loss=None,
    ):
        assert isinstance(loss, list)
        num_tasks = len(loss)
        loss = tf.stack(loss)
        tf.random.shuffle(loss)

        # Compute per-task gradients.
        grads_task = tf.vectorized_map(
            lambda x: tf.concat(
                [
                    tf.reshape(
                        grad,
                        [
                            -1,
                        ],
                    )
                    for grad in tf.gradients(x, var_list)
                    if grad is not None
                ],
                axis=0,
            ),
            loss,
        )

        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task * grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(
                    grads_task[k] * grads_task[k]
                )
                grad_task = grad_task - tf.minimum(proj_direction, 0.0) * grads_task[k]
            return grad_task

        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

        # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = np.prod(
                    [grad_shape.dims[i].value for i in range(len(grad_shape.dims))]
                )
                proj_grad = proj_grads_flatten[j][start_idx : start_idx + flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))
        return grads_and_vars

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _prepare(self):
        self.optimizer._prepare()

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        return self.optimizer._finish(update_ops, name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param


# %%
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


del_all_flags(flags.FLAGS)

FLAGS = flags.FLAGS

flags.DEFINE_integer("steps", 100, "Number of epoch to train.")
flags.DEFINE_integer("batch_size", 256, "Number of examples in a minibatch.")
flags.DEFINE_integer("order", -1, "Order of permutations to consider.")
flags.DEFINE_enum("eval", "test", ["valid", "test"], "The eval dataset.")
flags.DEFINE_enum(
    "method", "fast", ["fast", "ours"], "Multitask Training Method."
)
flags.DEFINE_list(
    "tasks",
    [
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ],
    "The attributes to predict in CelebA.",
)
flags.DEFINE_boolean("fast", False, "Whether to use fast training paradigm.")

# %%
SEED = 0
METRICS_AVERAGE = 1
EPSILON = 0.001
TRAIN_SIZE = 162770
VALID_SIZE = 19867
TEST_SIZE = 19962


# %%
class ResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, name):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(
            filters=filters[0],
            kernel_size=kernel_size[0],
            strides=strides,
            name="conv{}_1".format(name),
            kernel_initializer=glorot_uniform(seed=SEED),
            padding="same",
            use_bias=False,
        )
        self.bn1 = BatchNormalization(axis=3, name="bn{}_1".format(name))
        self.conv2 = Conv2D(
            filters=filters[1],
            kernel_size=kernel_size[1],
            strides=(1, 1),
            name="conv{}_2".format(name),
            kernel_initializer=glorot_uniform(seed=SEED),
            padding="same",
            use_bias=False,
        )
        self.bn2 = BatchNormalization(axis=3, name="bn{}_2".format(name))

        if strides == (1, 1):
            self.shortcut = Lambda(lambda x: x)
        else:
            self.shortcut = tf.keras.Sequential()
            shortcut_conv = Conv2D(
                filters=filters[1],
                kernel_size=1,
                strides=(2, 2),
                name="skip_conv{}_1".format(name),
                kernel_initializer=glorot_uniform(seed=SEED),
                padding="valid",
                use_bias=False,
            )
            shortcut_bn = BatchNormalization(axis=3, name="skip_bn{}_1".format(name))
            self.shortcut.add(shortcut_conv)
            self.shortcut.add(shortcut_bn)

    def call(self, inputs):
        x = inputs
        x = Activation("relu")(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = Add()([x, self.shortcut(inputs)])
        return Activation("relu")(x)


class ResNet18(tf.keras.Model):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1_1 = Conv2D(
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            name="conv1_1",
            kernel_initializer=glorot_uniform(seed=SEED),
            padding="same",
            use_bias=False,
        )
        self.bn1_1 = BatchNormalization(axis=3, name="bn1_1")
        self.resblock_2 = ResBlock([64, 64], [3, 3], (1, 1), "1")

    def call(self, inputs):
        x = inputs
        x = Activation("relu")(self.bn1_1(self.conv1_1(x)))
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = self.resblock_2(x)
        x = AveragePooling2D((2, 2), name="avg_pool")(x)
        x = Flatten()(x)
        return x


class AttributeDecoder(tf.keras.Model):
    def __init__(self):
        super(AttributeDecoder, self).__init__()
        self.fc1 = Dense(2, kernel_initializer=glorot_uniform(seed=SEED))

    def call(self, inputs):
        x = inputs
        x = self.fc1(x)
        return x


# %%
def res_block_step(inputs, base_updated):
    conv1 = tf.nn.conv2d(inputs, base_updated[0], strides=(2, 2), padding="SAME")
    mean1, variance1 = tf.nn.moments(conv1, axes=[0, 1, 2])
    gamma1, beta1 = base_updated[1], base_updated[2]
    bn_conv1 = tf.nn.batch_normalization(
        conv1, mean1, variance1, offset=beta1, scale=gamma1, variance_epsilon=EPSILON
    )
    relu1 = tf.nn.relu(bn_conv1)

    conv2 = tf.nn.conv2d(relu1, base_updated[3], strides=(1, 1), padding="SAME")
    mean2, variance2 = tf.nn.moments(conv2, axes=[0, 1, 2])
    gamma2, beta2 = base_updated[4], base_updated[5]
    bn_conv2 = tf.nn.batch_normalization(
        conv2, mean2, variance2, offset=beta2, scale=gamma2, variance_epsilon=EPSILON
    )

    skip_conv = tf.nn.conv2d(inputs, base_updated[6], strides=(2, 2), padding="VALID")
    skip_mean, skip_variance = tf.nn.moments(skip_conv, axes=[0, 1, 2])
    skip_gamma, skip_beta = base_updated[7], base_updated[8]
    skip_bn = tf.nn.batch_normalization(
        skip_conv,
        skip_mean,
        skip_variance,
        offset=skip_beta,
        scale=skip_gamma,
        variance_epsilon=EPSILON,
    )

    res_block = tf.nn.relu(bn_conv2 + skip_bn)
    return res_block


def base_step(inputs, base_updated):
    # ResNet Block 1 Output.
    conv1_1 = tf.nn.conv2d(inputs, base_updated[0], strides=(1, 1), padding="SAME")
    mean1_1, variance1_1 = tf.nn.moments(
        conv1_1, axes=[0, 1, 2], keepdims=True
    )  # normalize across the channel dimension for spacial batch norm.
    gamma1_1, beta1_1 = base_updated[1], base_updated[2]
    bn_conv1_1 = tf.nn.batch_normalization(
        conv1_1,
        mean1_1,
        variance1_1,
        offset=beta1_1,
        scale=gamma1_1,
        variance_epsilon=EPSILON,
    )
    res_block_1 = tf.nn.max_pool2d(
        tf.nn.relu(bn_conv1_1),
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
    )

    # ResNet Block 2
    conv2_1 = tf.nn.conv2d(res_block_1, base_updated[3], strides=(1, 1), padding="SAME")
    mean2_1, variance2_1 = tf.nn.moments(conv2_1, axes=[0, 1, 2])
    gamma2_1, beta2_1 = base_updated[4], base_updated[5]
    bn_conv2_1 = tf.nn.batch_normalization(
        conv2_1,
        mean2_1,
        variance2_1,
        offset=beta2_1,
        scale=gamma2_1,
        variance_epsilon=EPSILON,
    )
    res_block2_1 = tf.nn.relu(bn_conv2_1)

    conv2_2 = tf.nn.conv2d(
        res_block2_1, base_updated[6], strides=(1, 1), padding="SAME"
    )
    mean2_2, variance2_2 = tf.nn.moments(conv2_2, axes=[0, 1, 2])
    gamma2_2, beta2_2 = base_updated[7], base_updated[8]
    bn_conv2_2 = tf.nn.batch_normalization(
        conv2_2,
        mean2_2,
        variance2_2,
        offset=beta2_2,
        scale=gamma2_2,
        variance_epsilon=EPSILON,
    )
    res_block_2 = tf.nn.relu(bn_conv2_2 + res_block_1)

    avg_pool = tf.nn.avg_pool2d(
        res_block_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )
    resnet_out = tf.reshape(avg_pool, [inputs.shape[0], -1])
    return resnet_out


def decoder_step(inputs, decoder_updated):
    fc1 = tf.matmul(inputs, decoder_updated[0])
    return fc1


# %%
def permute(losses):
    """Returns all combinations of losses in the loss dictionary."""
    losses = OrderedDict(sorted(losses.items()))
    rtn = {}
    keys = list(losses.keys())
    for n in range(1, FLAGS.order + 1):
        for tasks in itertools.combinations_with_replacement(keys, n):
            rtn["|".join(tasks)] = sum([losses[task] for task in tasks])

    return rtn


def permute_list(lst):
    """Returns all combinations of tasks in the task list."""
    lst.sort()
    rtn = []
    for n in range(1, FLAGS.order + 1):
        for tasks in itertools.combinations_with_replacement(lst, n):
            rtn.append("|".join(tasks))

    return rtn


def decay_lr(step, optimizer):
    if (step + 1) % 15 == 0:
        optimizer.lr = optimizer.lr / 2.0
        print(
            "Decreasing the learning rate by 1/2. New Learning Rate: {}".format(
                optimizer.lr
            )
        )


def decay_pcgrad_lr(step, lr_var):
    if (step + 1) % 15 == 0:
        lr_var.assign(lr_var / 2.0)
        print("Decreasing the learning rate by 1/2.")


def add_average(lst, metrics_dict, n):
    if len(lst) < n:
        lst.append(metrics_dict)
    elif len(lst) == n:
        lst.pop(0)
        lst.append(metrics_dict)
    elif len(lst) > n:
        raise Exception("List size is greater than n. This should never happen.")


def compute_average(metrics_list, n):
    if not metrics_list:
        return {}
    rtn = {task: 0.0 for task in metrics_list[0]}
    for metric in metrics_list:
        for task in metric:
            rtn[task] += metric[task] / float(n)
    return rtn


def load_dataset(batch_size):
    train = tfds.load("celeb_a", split="train")
    resized_train = train.map(
        lambda d: (
            d["attributes"],
            tf.image.resize(
                tf.image.convert_image_dtype(d["image"], tf.float32), [64, 64]
            ),
        )
    )
    final_train = resized_train.shuffle(
        buffer_size=TRAIN_SIZE, seed=SEED, reshuffle_each_iteration=True
    ).batch(batch_size)

    valid = tfds.load("celeb_a", split="validation")
    resized_valid = valid.map(
        lambda d: (
            d["attributes"],
            tf.image.resize(
                tf.image.convert_image_dtype(d["image"], tf.float32), [64, 64]
            ),
        )
    )
    final_valid = resized_valid.batch(batch_size)

    test = tfds.load("celeb_a", split="test")
    resized_test = test.map(
        lambda d: (
            d["attributes"],
            tf.image.resize(
                tf.image.convert_image_dtype(d["image"], tf.float32), [64, 64]
            ),
        )
    )
    final_test = resized_test.batch(batch_size)

    Dataset = namedtuple("Dataset", ["train", "valid", "test"])
    return Dataset(final_train, final_valid, final_test)


def get_uncertainty_weights():
    uncertainty_weights = {}
    global shadow_uncertainty
    if shadow_uncertainty is None:
        shadow_uncertainty = tf.Variable(1.0)
    uncertainty_weights["5_o_Clock_Shadow"] = shadow_uncertainty
    global black_hair_uncertainty
    if black_hair_uncertainty is None:
        black_hair_uncertainty = tf.Variable(1.0)
    uncertainty_weights["Black_Hair"] = black_hair_uncertainty
    global blond_hair_uncertainty
    if blond_hair_uncertainty is None:
        blond_hair_uncertainty = tf.Variable(1.0)
    uncertainty_weights["Blond_Hair"] = blond_hair_uncertainty
    global brown_hair_uncertainty
    if brown_hair_uncertainty is None:
        brown_hair_uncertainty = tf.Variable(1.0)
    uncertainty_weights["Brown_Hair"] = brown_hair_uncertainty
    global goatee_uncertainty
    if goatee_uncertainty is None:
        goatee_uncertainty = tf.Variable(1.0)
    uncertainty_weights["Goatee"] = goatee_uncertainty
    global mustache_uncertainty
    if mustache_uncertainty is None:
        mustache_uncertainty = tf.Variable(1.0)
    uncertainty_weights["Mustache"] = mustache_uncertainty
    global no_beard_uncertainty
    if no_beard_uncertainty is None:
        no_beard_uncertainty = tf.Variable(1.0)
    uncertainty_weights["No_Beard"] = no_beard_uncertainty
    global rosy_cheeks_uncertainty
    if rosy_cheeks_uncertainty is None:
        rosy_cheeks_uncertainty = tf.Variable(1.0)
    uncertainty_weights["Rosy_Cheeks"] = rosy_cheeks_uncertainty
    global wearing_hat_uncertainty
    if wearing_hat_uncertainty is None:
        wearing_hat_uncertainty = tf.Variable(1.0)
    uncertainty_weights["Wearing_Hat"] = wearing_hat_uncertainty
    return uncertainty_weights


def init_uncertainty_weights():
    global shadow_uncertainty
    shadow_uncertainty = None
    global black_hair_uncertainty
    black_hair_uncertainty = None
    global blond_hair_uncertainty
    blond_hair_uncertainty = None
    global brown_hair_uncertainty
    brown_hair_uncertainty = None
    global goatee_uncertainty
    goatee_uncertainty = None
    global mustache_uncertainty
    mustache_uncertainty = None
    global no_beard_uncertainty
    no_beard_uncertainty = None
    global rosy_cheeks_uncertainty
    rosy_cheeks_uncertainty = None
    global wearing_hat_uncertainty
    wearing_hat_uncertainty = None


def init_gradnorm_weights():
    global shadow_gradnorm
    shadow_gradnorm = None
    global black_hair_gradnorm
    black_hair_gradnorm = None
    global blond_hair_gradnorm
    blond_hair_gradnorm = None
    global brown_hair_gradnorm
    brown_hair_gradnorm = None
    global goatee_gradnorm
    goatee_gradnorm = None
    global mustache_gradnorm
    mustache_gradnorm = None
    global no_beard_gradnorm
    no_beard_gradnorm = None
    global rosy_cheeks_gradnorm
    rosy_cheeks_gradnorm = None
    global wearing_hat_gradnorm
    wearing_hat_gradnorm = None


def init_gradnorm_weights():
    global shadow_gradnorm
    shadow_gradnorm = None
    global black_hair_gradnorm
    black_hair_gradnorm = None
    global blond_hair_gradnorm
    blond_hair_gradnorm = None
    global brown_hair_gradnorm
    brown_hair_gradnorm = None
    global goatee_gradnorm
    goatee_gradnorm = None
    global mustache_gradnorm
    mustache_gradnorm = None
    global no_beard_gradnorm
    no_beard_gradnorm = None
    global rosy_cheeks_gradnorm
    rosy_cheeks_gradnorm = None
    global wearing_hat_gradnorm
    wearing_hat_gradnorm = None


def fetch_gradnorm_weights():
    gradnorm_weights = {}
    global shadow_gradnorm
    if shadow_gradnorm is None:
        shadow_gradnorm = tf.Variable(1.0)
    gradnorm_weights["5_o_Clock_Shadow"] = shadow_gradnorm
    global black_hair_gradnorm
    if black_hair_gradnorm is None:
        black_hair_gradnorm = tf.Variable(1.0)
    gradnorm_weights["Black_Hair"] = black_hair_gradnorm
    global blond_hair_gradnorm
    if blond_hair_gradnorm is None:
        blond_hair_gradnorm = tf.Variable(1.0)
    gradnorm_weights["Blond_Hair"] = blond_hair_gradnorm
    global brown_hair_gradnorm
    if brown_hair_gradnorm is None:
        brown_hair_gradnorm = tf.Variable(1.0)
    gradnorm_weights["Brown_Hair"] = brown_hair_gradnorm
    global goatee_gradnorm
    if goatee_gradnorm is None:
        goatee_gradnorm = tf.Variable(1.0)
    gradnorm_weights["Goatee"] = goatee_gradnorm
    global mustache_gradnorm
    if mustache_gradnorm is None:
        mustache_gradnorm = tf.Variable(1.0)
    gradnorm_weights["Mustache"] = mustache_gradnorm
    global no_beard_gradnorm
    if no_beard_gradnorm is None:
        no_beard_gradnorm = tf.Variable(1.0)
    gradnorm_weights["No_Beard"] = no_beard_gradnorm
    global rosy_cheeks_gradnorm
    if rosy_cheeks_gradnorm is None:
        rosy_cheeks_gradnorm = tf.Variable(1.0)
    gradnorm_weights["Rosy_Cheeks"] = rosy_cheeks_gradnorm
    global wearing_hat_gradnorm
    if wearing_hat_gradnorm is None:
        wearing_hat_gradnorm = tf.Variable(1.0)
    gradnorm_weights["Wearing_Hat"] = wearing_hat_gradnorm
    return gradnorm_weights


def init_gradnorm_l0():
    global shadow_loss
    shadow_loss = None
    global black_hair_loss
    black_hair_loss = None
    global blond_hair_loss
    blond_hair_loss = None
    global brown_hair_loss
    brown_hair_loss = None
    global goatee_loss
    goatee_loss = None
    global mustache_loss
    mustache_loss = None
    global no_beard_loss
    no_beard_loss = None
    global rosy_cheeks_loss
    rosy_cheeks_loss = None
    global wearing_hat_loss
    wearing_hat_loss = None


def fetch_gradnorm_l0(losses):
    gradnorm_l0 = {}
    global shadow_loss
    if shadow_loss is None:
        if "5_o_Clock_Shadow" in losses:
            loss_val = losses["5_o_Clock_Shadow"]
        else:
            loss_val = 0.0
        shadow_loss = tf.Variable(loss_val)
    gradnorm_l0["5_o_Clock_Shadow"] = shadow_loss
    global black_hair_loss
    if black_hair_loss is None:
        if "Black_Hair" in losses:
            loss_val = losses["Black_Hair"]
        else:
            loss_val = 0.0
        black_hair_loss = tf.Variable(loss_val)
    gradnorm_l0["Black_Hair"] = black_hair_loss
    global blond_hair_loss
    if blond_hair_loss is None:
        if "Blond_Hair" in losses:
            loss_val = losses["Blond_Hair"]
        else:
            loss_val = 0.0
        blond_hair_loss = tf.Variable(loss_val)
    gradnorm_l0["Blond_Hair"] = blond_hair_loss
    global brown_hair_loss
    if brown_hair_loss is None:
        if "Brown_Hair" in losses:
            loss_val = losses["Brown_Hair"]
        else:
            loss_val = 0.0
        brown_hair_loss = tf.Variable(loss_val)
    gradnorm_l0["Brown_Hair"] = brown_hair_loss
    global goatee_loss
    if goatee_loss is None:
        if "Goatee" in losses:
            loss_val = losses["Goatee"]
        else:
            loss_val = 0.0
        goatee_loss = tf.Variable(loss_val)
    gradnorm_l0["Goatee"] = goatee_loss
    global mustache_loss
    if mustache_loss is None:
        if "Mustache" in losses:
            loss_val = losses["Mustache"]
        else:
            loss_val = 0.0
        mustache_loss = tf.Variable(loss_val)
    gradnorm_l0["Mustache"] = mustache_loss
    global no_beard_loss
    if no_beard_loss is None:
        if "No_Beard" in losses:
            loss_val = losses["No_Beard"]
        else:
            loss_val = 0.0
        no_beard_loss = tf.Variable(loss_val)
    gradnorm_l0["No_Beard"] = no_beard_loss
    global rosy_cheeks_loss
    if rosy_cheeks_loss is None:
        if "Rosy_Cheeks" in losses:
            loss_val = losses["Rosy_Cheeks"]
        else:
            loss_val = 0.0
        rosy_cheeks_loss = tf.Variable(loss_val)
    gradnorm_l0["Rosy_Cheeks"] = rosy_cheeks_loss
    global wearing_hat_loss
    if wearing_hat_loss is None:
        if "Wearing_Hat" in losses:
            loss_val = losses["Wearing_Hat"]
        else:
            loss_val = 0.0
        wearing_hat_loss = tf.Variable(loss_val)
    gradnorm_l0["Wearing_Hat"] = wearing_hat_loss
    return gradnorm_l0


def compute_gradnorm_losses(losses, gradnorm_l0, gradnorms, expected_gradnorm):
    task_li = {}
    for task in FLAGS.tasks:
        task_li[task] = losses[task] / gradnorm_l0[task]
    li_expected = tf.reduce_mean(list(task_li.values()))

    gradnorm_loss = {}
    for task in FLAGS.tasks:
        task_ri = tf.math.pow(task_li[task] / li_expected, params.alpha)
        gradnorm_loss[task] = tf.norm(
            gradnorms[task] - tf.stop_gradient(expected_gradnorm * task_ri), ord=1
        )
    total_gradnorm_loss = tf.reduce_sum(list(gradnorm_loss.values()))
    return total_gradnorm_loss


# %%
def train(params):
    print(params)

    ResBase = ResNet18()
    ResTowers = {task: AttributeDecoder() for task in FLAGS.tasks}

    dataset = load_dataset(FLAGS.batch_size)
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.keras.optimizers.SGD(params.lr, momentum=0.9)

    # Initialize the weights.
    dummy_input = tf.zeros((1, 64, 64, 3))
    dummy_rep = ResBase(dummy_input)
    dummy_preds = {task: model(dummy_rep) for (task, model) in ResTowers.items()}

    all_trainable_variables = ResBase.trainable_variables
    for task in FLAGS.tasks:
        all_trainable_variables.extend(ResTowers[task].trainable_variables)
    optimizer.build(all_trainable_variables)

    base_backup = [tf.Variable(param) for param in ResBase.variables]
    decoders_backup = {
        task: [tf.Variable(param) for param in ResTowers[task].variables]
        for task in FLAGS.tasks
    }
    opt_backup = [tf.Variable(param) for param in optimizer.variables()]

    @tf.function()
    def train_step_ours(input, labels, first_step=False):
        """Modified for Ours Method."""
        with tf.GradientTape(persistent=True) as tape:
            rep = ResBase(input, training=True)
            preds = {
                task: model(rep, training=True) for (task, model) in ResTowers.items()
            }
            losses = {
                task: tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels[task], logits=preds[task]
                    )
                )
                for task in labels
            }
            loss = tf.add_n(list(losses.values()))

            # Compute the gradient of the task-specific loss w.r.t. the shared base.
            task_losses = {}
            task_losses["train_loss"] = losses
            task_permutations = permute(losses)

        # Backup the original weights.
        for original, backup in zip(optimizer.variables(), opt_backup):
            backup.assign(original)
        for original, backup in zip(ResBase.variables, base_backup):
            backup.assign(original)
        for task in FLAGS.tasks:
            for original, backup in zip(
                ResTowers[task].variables, decoders_backup[task]
            ):
                backup.assign(original)
        # Compute the gradients for all the permutations.
        combined_task_gradients = [
            (
                combined_task,
                tape.gradient(
                    task_permutations[combined_task], ResBase.trainable_weights
                ),
            )
            for combined_task in task_permutations
        ]
        
        for combined_task, task_gradient in combined_task_gradients:
            optimizer.apply_gradients(zip(task_gradient, ResBase.trainable_variables))
            # Update decoder weights.
            for task in combined_task.split("|"):
                decoder_grads = tape.gradient(
                    losses[task], ResTowers[task].trainable_weights
                )
                optimizer.apply_gradients(
                    zip(decoder_grads, ResTowers[task].trainable_weights)
                )
            task_update_rep = ResBase(input, training=False)
            if "|" in combined_task:
                task_update_preds = {
                    task: ResTowers[task](task_update_rep, training=False)
                    for task in combined_task.split("|")
                }
                task_update_losses = {
                    task: tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels[task], logits=task_update_preds[task]
                        )
                    )
                    for task in combined_task.split("|")
                }
            else:
                task_update_preds = {
                    task: ResTowers[task](task_update_rep, training=False)
                    for task in labels
                }
                task_update_losses = {
                    task: tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels[task], logits=task_update_preds[task]
                        )
                    )
                    for task in labels
                }
            task_losses[combined_task] = task_update_losses
            # Restore the original weights for base and optimizer.
            for original, backup in zip(optimizer.variables(), opt_backup):
                original.assign(backup)
            for original, backup in zip(ResBase.variables, base_backup):
                original.assign(backup)
            # Restore the original weights for decoders.
            for task in FLAGS.tasks:
                for original, backup in zip(
                    ResTowers[task].variables, decoders_backup[task]
                ):
                    original.assign(backup)

        # DO NOT apply Nesterov in normal mtl training.
        for task, model in ResTowers.items():
            task_grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, model.trainable_weights))

        # Apply the traditional MTL update since this is a normal train step.
        base_grads = tape.gradient(loss, ResBase.trainable_weights)
        optimizer.apply_gradients(zip(base_grads, ResBase.trainable_weights))

        global_step.assign_add(1)
        return losses, task_losses

    @tf.function()
    def train_step_tag(input, labels, first_step=False):
        """This is TAG."""
        with tf.GradientTape(persistent=True) as tape:
            rep = ResBase(input, training=True)
            preds = {
                task: model(rep, training=True) for (task, model) in ResTowers.items()
            }
            losses = {
                task: tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels[task], logits=preds[task]
                    )
                )
                for task in labels
            }
            loss = tf.add_n(list(losses.values()))

            # Compute the gradient of the task-specific loss w.r.t. the shared base.
            task_gains = {}
            task_permutations = permute(losses)
            combined_task_gradients = [
                (
                    combined_task,
                    tape.gradient(
                        task_permutations[combined_task], ResBase.trainable_weights
                    ),
                )
                for combined_task in task_permutations
            ]

        for combined_task, task_gradient in combined_task_gradients:
            if first_step:
                base_update = [optimizer.lr * grad for grad in task_gradient]
                base_updated = [
                    param - update
                    for param, update in zip(ResBase.trainable_weights, base_update)
                ]
            else:
                base_update = [
                    (
                        optimizer._momentum * optimizer.get_slot(param, "momentum")
                        - optimizer.lr * grad
                    )
                    for param, grad in zip(ResBase.trainable_weights, task_gradient)
                ]
                base_updated = [
                    param + update
                    for param, update in zip(ResBase.trainable_weights, base_update)
                ]
            task_update_rep = base_step(input, base_updated)
            task_update_preds = {
                task: model(task_update_rep, training=True)
                for (task, model) in ResTowers.items()
            }
            task_update_losses = {
                task: tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels[task], logits=task_update_preds[task]
                    )
                )
                for task in labels
            }
            task_gain = {
                task: (1.0 - task_update_losses[task] / losses[task]) / optimizer.lr
                for task in FLAGS.tasks
            }
            task_gains[combined_task] = task_gain

        # DO NOT apply Nesterov in normal mtl training.
        for task, model in ResTowers.items():
            task_grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, model.trainable_weights))

        # Apply the traditional MTL update since this is a normal train step.
        base_grads = tape.gradient(loss, ResBase.trainable_weights)
        optimizer.apply_gradients(zip(base_grads, ResBase.trainable_weights))

        global_step.assign_add(1)
        return losses, task_gains

    @tf.function()
    def train_fast_step(input, labels, first_step=False):
        """Call this function to evaluate task groupings. It's faster."""
        with tf.GradientTape(persistent=True) as tape:
            rep = ResBase(input, training=True)
            preds = {
                task: model(rep, training=True) for (task, model) in ResTowers.items()
            }
            losses = {
                task: tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels[task], logits=preds[task]
                    )
                )
                for task in labels
            }
            loss = tf.add_n(list(losses.values()))

        # DO NOT apply Nesterov in normal mtl training.
        for task, model in ResTowers.items():
            task_grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, model.trainable_weights))

        # Apply the traditional MTL update since this is a normal train step.
        base_grads = tape.gradient(loss, ResBase.trainable_weights)
        optimizer.apply_gradients(zip(base_grads, ResBase.trainable_weights))

        global_step.assign_add(1)
        return losses, {}

    @tf.function()
    def eval_step(input, labels):
        rep = ResBase(input)
        preds = {task: ResTowers[task](rep) for (task, model) in ResTowers.items()}
        int_preds = {
            task: tf.math.argmax(preds[task], 1, tf.dtypes.int32) for task in labels
        }
        int_labels = {
            task: tf.math.argmax(labels[task], 1, tf.dtypes.int32) for task in labels
        }
        losses = {
            task: tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.cast(labels[task], tf.float32), logits=preds[task]
                )
            )
            for task in labels
        }
        accuracies = {
            task: tf.math.count_nonzero(tf.equal(int_preds[task], int_labels[task]))
            for task in labels
        }
        Eval = namedtuple("Eval", ["losses", "accuracies"])
        return Eval(losses, accuracies)

    # Training Loop.
    metrics = {"train_loss": [], "eval_loss": [], "eval_acc": []}
    gradient_metrics = {task: [] for task in permute_list(FLAGS.tasks)}
    gradient_metrics["train_loss"] = []
    final_metrics = {"train_loss": [], "eval_loss": [], "eval_acc": [], 'test_loss': [], 'test_acc': []}
    model_params = []

    train_end = None
    eval_end = None
    end = None
    for step in range(FLAGS.steps):
        if end:
            print(f"Differnece in time: {end - start}")
        start = time.time()
        print("epoch: {}".format(step))
        decay_lr(step, optimizer)
        batch_train_loss = {task: 0.0 for task in FLAGS.tasks}
        batch_grad_metrics = {
            combined_task: {task: [] for task in FLAGS.tasks}
            for combined_task in gradient_metrics
        }
        batch_grad_metrics["train_loss"] = {task: [] for task in FLAGS.tasks}
        iteration = 0
        progbar = tf.keras.utils.Progbar(len(dataset.train))
        for labels, img in dataset.train:
            iteration += 1
            progbar.update(iteration)
            labels = {
                task: tf.keras.utils.to_categorical(labels[task], num_classes=2)
                for task in labels
                if task in FLAGS.tasks
            }
            if FLAGS.method == "fast":
                losses, task_gains = train_fast_step(
                    img, labels, first_step=(len(optimizer.variables()) == 0)
                )
            elif FLAGS.method == "ours":
                losses, task_gains = train_step_ours(
                    img, labels, first_step=(len(optimizer.variables()) == 1)
                )
            else:
                raise Exception("Unrecognized Method Selected.")

            # Record batch-level training and gradient metrics.
            for combined_task, task_gain_map in task_gains.items():
                for task, gain in task_gain_map.items():
                    batch_grad_metrics[combined_task][task].append(gain.numpy())
            for task, loss in losses.items():
                batch_train_loss[task] += loss.numpy() / (
                    math.ceil(TRAIN_SIZE / FLAGS.batch_size)
                )

        train_end = time.time()
        print(f"Time to train: {train_end - start}")
        # Record epoch-level training and gradient metrics.
        add_average(metrics["train_loss"], batch_train_loss, METRICS_AVERAGE)
        for combined_task, task_gain_map in batch_grad_metrics.items():
            gradient_metrics[combined_task].append(task_gain_map)

        batch_eval_loss = {task: 0.0 for task in FLAGS.tasks}
        batch_eval_acc = {task: 0.0 for task in FLAGS.tasks}
        for labels, img in dataset.test if FLAGS.eval == "test" else dataset.valid:
            labels = {
                task: tf.keras.utils.to_categorical(labels[task], num_classes=2)
                for task in labels
                if task in FLAGS.tasks
            }
            eval_metrics = eval_step(img, labels)
            for task in FLAGS.tasks:
                EVAL_SIZE = TEST_SIZE if FLAGS.eval == "test" else VALID_SIZE
                batch_eval_loss[task] += eval_metrics.losses[task].numpy() / (
                    math.ceil(EVAL_SIZE / FLAGS.batch_size)
                )
                batch_eval_acc[task] += (
                    eval_metrics.accuracies[task].numpy() / EVAL_SIZE
                )
        add_average(metrics["eval_loss"], batch_eval_loss, METRICS_AVERAGE)
        add_average(metrics["eval_acc"], batch_eval_acc, METRICS_AVERAGE)
        eval_end = time.time()
        print(f"Time to eval: {eval_end - train_end}")

        for metric in metrics:
            final_metrics[metric].append(
                compute_average(metrics[metric], METRICS_AVERAGE)
            )

        # Save past EARLY_STOP sets of parameters.
        cur_params = [
            (
                "base",
                copy.deepcopy(ResBase.trainable_weights),
                copy.deepcopy(ResBase.non_trainable_weights),
            )
        ] + [
            (
                task,
                copy.deepcopy(tower.trainable_weights),
                copy.deepcopy(tower.non_trainable_weights),
            )
            for task, tower in ResTowers.items()
        ]
        model_params.append(tuple(cur_params))

        # Evaluate on the test set.
        batch_test_acc = {task: 0.0 for task in FLAGS.tasks}
        batch_test_loss = {task: 0.0 for task in FLAGS.tasks}
        for labels, img in dataset.test:
            labels = {
                task: tf.keras.utils.to_categorical(labels[task], num_classes=2)
                for task in labels
                if task in FLAGS.tasks
            }
            test_metrics = eval_step(img, labels)
            for task in FLAGS.tasks:
                EVAL_SIZE = TEST_SIZE if FLAGS.eval == "test" else VALID_SIZE
                batch_test_loss[task] += test_metrics.losses[task].numpy() / (
                    math.ceil(EVAL_SIZE / FLAGS.batch_size)
                )
                batch_test_acc[task] += (
                    test_metrics.accuracies[task].numpy() / EVAL_SIZE
                )

        final_metrics["test_loss"].append(batch_test_loss)
        final_metrics["test_acc"].append(batch_test_acc)

        print_test_acc = "\n".join(
            [
                "{}: {:.2f}".format(task, 100.0 * metric)
                for task, metric in batch_test_acc.items()
            ]
        )
        print_test_loss = "\n".join(
            [
                "{}: {:.4f}".format(task, metric)
                for task, metric in batch_test_loss.items()
            ]
        )
        print("Test Accuracy:\n{}\n".format(print_test_acc))
        print("Test Loss:\n{}\n".format(print_test_loss))

        print_train_loss = "\n".join(
            [
                "{}: {:.4f}".format(task, metric)
                for task, metric in final_metrics["train_loss"][-1].items()
            ]
        )
        print("Train Loss:\n{}\n".format(print_train_loss))

        # print("grad metrics for fun: {}".format(gradient_metrics))

        print_eval_loss = "\n".join(
            [
                "{}: {:.4f}".format(task, metric)
                for task, metric in final_metrics["eval_loss"][-1].items()
            ]
        )
        print("Eval Loss:\n{}\n".format(print_eval_loss))
        print_eval_acc = "\n".join(
            [
                "{}: {:.2f}".format(task, 100.0 * metric)
                for task, metric in final_metrics["eval_acc"][-1].items()
            ]
        )
        print("Eval Accuracy:\n{}\n".format(print_eval_acc))
        print("\n-------------\n")
        end = time.time()

    return final_metrics, gradient_metrics, model_params


# %%
import sys

Params = namedtuple(
    "Params", ["lr", "alpha"]
)  # Params can possibly be tuned, FLAGS can't be tuned.
params = Params(lr=0.0005, alpha=0.1)
FLAGS.steps = 100  # MOO: train for 100 epochs.
FLAGS.batch_size = 256  # MOO: train with batch size = 256
FLAGS.eval = "valid"
FLAGS.method = "ours"
FLAGS.order = 2
FLAGS.tasks = [
    "5_o_Clock_Shadow",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Goatee",
    "Mustache",
    "No_Beard",
    "Rosy_Cheeks",
    "Wearing_Hat",
]  # 9 out of 40 attributes.
FLAGS(sys.argv)

import os
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH")

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        device_id = 0
        tf.config.experimental.set_visible_devices(gpus[device_id], "GPU")
        tf.config.experimental.set_memory_growth(gpus[device_id], True)
        print(f"GPU {device_id} is set for TensorFlow.")
    except RuntimeError as e:
        print(e)

# %%
# %%capture
# run the model 1 time
tf.compat.v1.reset_default_graph()
eval_metrics, gradient_metrics, model_params = train(params)
# print(gradient_metrics)

# %%
import pickle
import os

result_folder = 'results/collect/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

eval_metrics_path = os.path.join(result_folder, "eval_metrics.pkl")
gradient_metrics_path = os.path.join(result_folder, "gradient_metrics.pkl")
model_params_path = os.path.join(result_folder, "model_params.pkl")

with open(eval_metrics_path, "wb") as f:
    pickle.dump(eval_metrics, f)
with open(gradient_metrics_path, "wb") as f:
    pickle.dump(gradient_metrics, f)
with open(model_params_path, "wb") as f:
    pickle.dump(model_params, f)