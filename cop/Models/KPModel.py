import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = KP_Encoder(**model_params)
        self.decoder = KP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        self.encoded_graph = self.encoded_nodes.mean(dim=1, keepdim=True)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            # encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # self.decoder.set_q1(encoded_first_node)
        else:
            probs = self.decoder(self.encoded_graph, state.capacity, ninf_mask=state.fit_ninf_mask)
            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    prob = prob*(~state.finished)+state.finished
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob


class KP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, self.head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)


    def forward(self, input1):
        # input.shape = (batch, problem, EMBEDDING_DIM)

        q = reshape_by_heads(self.Wq(input1), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=self.head_num)
        # q shape = (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape = (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape = (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3


class KP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(1+embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.embedding_dim = embedding_dim
        self.head_num = head_num


    def set_kv(self, encoded_nodes):
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_num)
        # shape = (batch, HEAD_NUM, problem, KEY_DIM)

        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, graph, capacity, ninf_mask=None):
        batch_s = capacity.size(0)
        group_s = capacity.size(1)

        #  Multi-Head Attention
        #######################################################
        input1 = graph.expand(batch_s, group_s, self.embedding_dim)
        input2 = capacity[:, :, None]
        input_cat = torch.cat((input1, input2), dim=2)
        # shape = (batch, group, 1+EMBEDDING_DIM)

        q = reshape_by_heads(self.Wq(input_cat), head_num=self.head_num)
        # shape = (batch, HEAD_NUM, group, KEY_DIM)

        out_concat = multi_head_attention(q, self.k, self.v, ninf_mask=ninf_mask)
        # shape = (batch, group, HEAD_NUM*KEY_DIM)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch, group, EMBEDDING_DIM)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape = (batch, group, problem)

        score_scaled = score / np.sqrt(self.embedding_dim)
        # shape = (batch, group, problem)

        score_clipped = self.model_params['logit_clipping'] * torch.tanh(score_scaled)

        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch, group, problem)

        return probs


def reshape_by_heads(qkv, head_num):
    # q.shape = (batch, C, head_num*key_dim)

    batch_s = qkv.size(0)
    C = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, C, head_num, -1)
    # shape = (batch, C, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape = (batch, head_num, C, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch, head_num, n, TSP_SIZE)

    score_scaled = score / np.sqrt(key_dim)
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, problem_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']

        self.norm_by_EMB = nn.BatchNorm1d(self.embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape = (batch, problem, EMBEDDING_DIM)

        batch_s = input1.size(0)
        problem_s = input1.size(1)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, self.embedding_dim))

        return normalized.reshape(batch_s, problem_s, self.embedding_dim)


class Feed_Forward_Module(nn.Module):
    def __init__(self,**model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape = (batch, problem, EMBEDDING_DIM)

        return self.W2(F.relu(self.W1(input1)))