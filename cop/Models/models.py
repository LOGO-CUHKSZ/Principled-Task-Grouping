import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator
from itertools import chain


class COPModel(nn.Module):
    def __init__(self, problem_list, **model_params):
        super().__init__()
        self.problem_list = problem_list
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        # self.headers = nn.ModuleList([Header(problem, embedding_dim) for problem in problem_list])
        self.headers = nn.ModuleDict(
            {
                f"header_{problem}": Header(problem, embedding_dim)
                for problem in problem_list
            }
        )
        encoder_layer_num = self.model_params["encoder_layer_num"]
        self.encoder = nn.ModuleList(
            [Encoder(**model_params) for _ in range(encoder_layer_num)]
        )
        # self.decoders = nn.ModuleList([Decoder(problem, **model_params) for problem in problem_list])
        self.decoders = nn.ModuleDict(
            {
                f"decoder_{problem}": Decoder(problem, **model_params)
                for problem in problem_list
            }
        )
        self.encoded_nodes = None
        self.idxs = {problem_list[i]: i for i in range(len(self.problem_list))}

    # def pre_forward(self, reset_states):
    #     header_embedding = []
    #     for i in range(len(reset_states)):
    #         for reset_state in reset_states[i]:
    #             embed = self.headers[i](reset_state)
    #             header_embedding.append(embed)
    #     dims = [embed.shape[1] for embed in header_embedding]
    #     out = torch.cat(header_embedding, dim=1)
    #     for layer in self.encoder:
    #         out, _ = layer(out)
    #     # split the embedding for each decoder
    #     self.encoded_nodes = out.split(dims,dim=1)
    #     if 'KP' in self.problem_list:
    #         idx = self.problem_list.index('KP')
    #         self.encoded_graph = self.encoded_nodes[idx].mean(dim=1, keepdim=True)
    #     for i,embed in enumerate(self.encoded_nodes):
    #         self.decoders[i].set_kv(embed)

    def pre_forward(self, reset_states):
        header_embedding = []
        for problem, reset_states_per_problem in zip(self.headers.keys(), reset_states):
            for reset_state in reset_states_per_problem:
                embed = self.headers[problem](reset_state)
                header_embedding.append(embed)

        dims = [embed.shape[1] for embed in header_embedding]
        out = torch.cat(header_embedding, dim=1)

        for _, layer in self.encoder.items():
            out, _ = layer(out)

        # split the embedding for each decoder
        self.encoded_nodes = out.split(dims, dim=1)

        if "KP" in self.problem_list:
            idx = self.problem_list.index("KP")
            self.encoded_graph = self.encoded_nodes[idx].mean(dim=1, keepdim=True)

        for i, embed in enumerate(self.encoded_nodes):
            self.decoders[f"decoder_{self.problem_list[i]}"].set_kv(embed)

    def pre_forward_oneCOP(self, reset_state, problem):
        self.encoded_nodes = [None] * len(self.problem_list)
        if problem == "TSP":
            idx = self.problem_list.index("TSP")
            # out = self.headers[idx](reset_state)
            out = self.headers[f"header_{self.problem_list[idx]}"](reset_state)
            for layer in self.encoder:
                out, _ = layer(out)
            self.encoded_nodes[idx] = out
            # self.decoders[idx].set_kv(self.encoded_nodes[idx])
            self.decoders[f"decoder_{self.problem_list[idx]}"].set_kv(
                self.encoded_nodes[idx]
            )
        elif problem == "CVRP":
            idx = self.problem_list.index("CVRP")
            # out = self.headers[idx](reset_state)
            out = self.headers[f"header_{self.problem_list[idx]}"](reset_state)
            for layer in self.encoder:
                out, _ = layer(out)
            self.encoded_nodes[idx] = out
            # self.decoders[idx].set_kv(self.encoded_nodes[idx])
            self.decoders[f"decoder_{self.problem_list[idx]}"].set_kv(
                self.encoded_nodes[idx]
            )
        elif problem == "KP":
            idx = self.problem_list.index("KP")
            # out = self.headers[idx](reset_state)
            out = self.headers[f"header_{self.problem_list[idx]}"](reset_state)
            for layer in self.encoder:
                out, _ = layer(out)
            self.encoded_graph = out.mean(dim=1, keepdim=True)
            self.encoded_nodes[idx] = out
            # self.decoders[idx].set_kv(self.encoded_nodes[idx])
            self.decoders[f"decoder_{self.problem_list[idx]}"].set_kv(
                self.encoded_nodes[idx]
            )
        elif problem == "OP":
            idx = self.problem_list.index("OP")
            # out = self.headers[idx](reset_state)
            out = self.headers[f"header_{self.problem_list[idx]}"](reset_state)
            for layer in self.encoder:
                out, _ = layer(out)
            self.encoded_nodes[idx] = out
            # self.decoders[idx].set_kv(self.encoded_nodes[idx])
            self.decoders[f"decoder_{self.problem_list[idx]}"].set_kv(
                self.encoded_nodes[idx]
            )

    def TSP_forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(
                self.encoded_nodes[self.idxs["TSP"]], selected
            )
            # shape: (batch, pomo, embedding)
            # self.decoders[self.idxs["TSP"]].set_q1(encoded_first_node)
            self.decoders[f"decoder_TSP"].set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(
                self.encoded_nodes[self.idxs["TSP"]], state.current_node
            )
            # shape: (batch, pomo, embedding)
            # probs = self.decoders[self.idxs["TSP"]](encoded_last_node, state.ninf_mask)
            probs = self.decoders[f"decoder_TSP"](encoded_last_node, state.ninf_mask)

            # shape: (batch, pomo, problem)

            if self.training or self.model_params["eval_type"] == "softmax":
                while True:
                    selected = (
                        probs.reshape(batch_size * pomo_size, -1)
                        .multinomial(1)
                        .squeeze(dim=1)
                        .reshape(batch_size, pomo_size)
                    )
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(
                        batch_size, pomo_size
                    )
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob

    def CVRP_forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(
                batch_size, pomo_size
            )
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(
                self.encoded_nodes[self.idxs["CVRP"]], state.current_node
            )
            # shape: (batch, pomo, embedding)
            # probs = self.decoders[self.idxs["CVRP"]](
            #     encoded_last_node, state.load, state.ninf_mask
            # )
            probs = self.decoders[f"decoder_CVRP"](
                encoded_last_node, state.load, state.ninf_mask
            )
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params["eval_type"] == "softmax":
                while (
                    True
                ):  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = (
                            probs.reshape(batch_size * pomo_size, -1)
                            .multinomial(1)
                            .squeeze(dim=1)
                            .reshape(batch_size, pomo_size)
                        )
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(
                        batch_size, pomo_size
                    )
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob

    def KP_forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            # encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # self.decoder.set_q1(encoded_first_node)
        else:
            # probs = self.decoders[self.idxs["KP"]](
            #     self.encoded_graph, state.capacity, state.fit_ninf_mask
            # )
            probs = self.decoders[f"decoder_KP"](
                self.encoded_graph, state.capacity, state.fit_ninf_mask
            )
            if self.training or self.model_params["eval_type"] == "softmax":
                while True:
                    selected = (
                        probs.reshape(batch_size * pomo_size, -1)
                        .multinomial(1)
                        .squeeze(dim=1)
                        .reshape(batch_size, pomo_size)
                    )
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(
                        batch_size, pomo_size
                    )
                    # prob = prob*(~state.finished)+state.finished
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob

    def OP_forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        problem_size = state.ninf_mask.size(-1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            probs = torch.ones(size=(batch_size, pomo_size, problem_size))

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(
                batch_size, pomo_size
            )
            probs = torch.ones(size=(batch_size, pomo_size, problem_size))

        else:
            encoded_last_node = _get_encoding(
                self.encoded_nodes[self.idxs["OP"]], state.current_node
            )
            # shape: (batch, pomo, embedding)
            # probs = self.decoders[self.idxs["OP"]](
            #     encoded_last_node, state.remain_dist, state.ninf_mask
            # )

            probs = self.decoders[f"decoder_OP"](
                encoded_last_node, state.remain_dist, state.ninf_mask
            )
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params["eval_type"] == "softmax":
                while (
                    True
                ):  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = (
                            probs.reshape(batch_size * pomo_size, -1)
                            .multinomial(1)
                            .squeeze(dim=1)
                            .reshape(batch_size, pomo_size)
                        )
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(
                        batch_size, pomo_size
                    )
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                probs = None  # value not needed. Can be anything.

        return selected, probs

    def forward(self, state, problem):
        if problem == "TSP":
            selected, prob = self.TSP_forward(state)
        elif problem == "CVRP":
            selected, prob = self.CVRP_forward(state)
        elif problem == "KP":
            selected, prob = self.KP_forward(state)
        elif problem == "OP":
            selected, prob = self.OP_forward(state)
        else:
            NotImplementedError
        return selected, prob

    def get_atten_weights(self, reset_states):
        header_embedding = []
        for i in range(len(reset_states)):
            reset_state = reset_states[i]
            embed = self.headers[i](reset_state)
            header_embedding.append(embed)
        out = torch.cat(header_embedding, dim=1)
        weights = []
        for layer in self.encoder:
            out, layer_weight = layer(out)
            weights.append(layer_weight)
        return torch.stack(weights, dim=1)  # B x L x H x N x N

    def shared_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return self.encoder.parameters()

    # def task_specific_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
    #     return chain(self.headers.parameters(), self.decoders.parameters())

    def task_specific_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        headers_params = chain(*[module.parameters() for module in self.headers.values()])
        decoders_params = chain(*[module.parameters() for module in self.decoders.values()])
        return chain(headers_params, decoders_params)


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(
        batch_size, pomo_size, embedding_dim
    )
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


class Header(nn.Module):
    def __init__(self, problem, embedding_dim):
        super().__init__()
        self.problem = problem
        if problem == "TSP":
            self.header = nn.Linear(2, embedding_dim)
        elif problem == "CVRP":
            self.embedding_depot = nn.Linear(2, embedding_dim)
            self.embedding_node = nn.Linear(3, embedding_dim)
        elif problem == "KP":
            self.header = nn.Linear(2, embedding_dim)
        elif problem == "OP":
            self.embedding_depot = nn.Linear(2, embedding_dim)
            self.embedding_node = nn.Linear(3, embedding_dim)
        else:
            NotImplementedError

    def forward(self, input):
        if self.problem == "TSP":
            head_embedding = self.header(input.problems)
        elif self.problem == "CVRP":
            depot_xy = input.depot_xy
            node_xy = input.node_xy
            node_demand = input.node_demand
            node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
            embedded_depot = self.embedding_depot(depot_xy)
            embedded_node = self.embedding_node(node_xy_demand)
            head_embedding = torch.cat((embedded_depot, embedded_node), dim=1)
        elif self.problem == "KP":
            head_embedding = self.header(input.problems)
        elif self.problem == "OP":
            depot_xy, node_xy = input.depot_xy, input.node_xy
            node_prize = input.prize[:, 0, 1:]
            node_xy_prize = torch.cat((node_xy, node_prize[:, :, None]), dim=2)
            embedded_depot = self.embedding_depot(depot_xy)
            embedded_node = self.embedding_node(node_xy_prize)
            head_embedding = torch.cat((embedded_depot, embedded_node), dim=1)
        else:
            NotImplementedError
        return head_embedding


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(
            batch_s, head_num, n, input_s
        )
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
            batch_s, head_num, n, input_s
        )

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat, weights


class Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params["head_num"]

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat, weights = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3, weights


class Decoder(nn.Module):
    def __init__(self, problem, **model_params):
        super().__init__()
        self.problem = problem
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        if problem == "TSP":
            self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        elif problem == "CVRP":
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
            self.q_first = None  # saved q1, for multi-head attention
        elif problem == "KP":
            self.Wq = nn.Linear(1 + embedding_dim, head_num * qkv_dim, bias=False)
        elif problem == "OP":
            self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
            self.q_first = None
        else:
            NotImplementedError
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.embedding_dim = embedding_dim
        self.head_num = head_num

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params["head_num"]

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params["head_num"]
        if self.problem == "TSP":
            self.q_first = reshape_by_heads(
                self.Wq_first(encoded_q1), head_num=head_num
            )
        elif self.problem == "CVRP":
            self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        else:
            NotImplementedError
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, *input):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params["head_num"]

        if self.problem == "TSP":
            encoded_last_node, ninf_mask = input
            q_last = reshape_by_heads(
                self.Wq_last(encoded_last_node), head_num=head_num
            )
            q = self.q_first + q_last
        elif self.problem == "CVRP":
            encoded_last_node, load, ninf_mask = input
            input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
            # shape = (batch, group, EMBEDDING_DIM+1)

            q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
            # shape: (batch, head_num, pomo, qkv_dim)
            # q = self.q1 + self.q2 + q_last
            # # shape: (batch, head_num, pomo, qkv_dim)
            q = q_last
        elif self.problem == "KP":
            graph, capacity, ninf_mask = input
            batch_s = capacity.size(0)
            group_s = capacity.size(1)

            #  Multi-Head Attention
            #######################################################
            input1 = graph.expand(batch_s, group_s, self.embedding_dim)
            input2 = capacity[:, :, None]
            input_cat = torch.cat((input1, input2), dim=2)
            # shape = (batch, group, 1+EMBEDDING_DIM)
            q = reshape_by_heads(self.Wq(input_cat), head_num=self.head_num)
        elif self.problem == "OP":
            encoded_last_node, remain_dist, ninf_mask = input
            input_cat = torch.cat((encoded_last_node, remain_dist[:, :, None]), dim=2)
            q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
            q = q_last
        else:
            NotImplementedError

        out_concat, _ = multi_head_attention(
            q, self.k, self.v, rank3_ninf_mask=ninf_mask
        )
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params["sqrt_embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs
