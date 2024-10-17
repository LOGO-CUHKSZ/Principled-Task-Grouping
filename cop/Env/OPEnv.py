from dataclasses import dataclass
import torch
from .OPProblemDef import get_random_op_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_OP_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    prize: torch.Tensor = None
    # shape: (batch, problem)
    max_length: torch.Tensor = None


@dataclass
class Step_OP_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = 0

    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    remain_dist: torch.Tensor = None
    # shape: (batch, pomo)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    visit_inf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem)


class OPEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params["problem_size"]
        self.pomo_size = env_params["pomo_size"]
        self.prize_type = env_params["prize_type"]

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_xy = None
        self.node_xy = None
        self.prize = None
        self.max_length = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.done_idx = None
        self.length = None
        self.cur_total_prize = None

    def get_length(self, scale):
        if scale <= 20:
            leng = 2.0
        elif scale > 20 and scale <= 50:
            leng = 3.0
        elif scale > 50 and scale <= 100:
            leng = 4.0
        else:
            leng = 5.0
        return leng

    def generate_data(self, batch_size):
        depot, loc, prize, max_length = get_random_op_problems(
            batch_size, self.problem_size, self.prize_type
        )
        instance = torch.zeros(loc.shape[0], loc.shape[1] + 1, 3)
        instance[:, 0:1, :2] = depot
        instance[:, 1:, :2] = loc
        instance[:, :, 2] = prize
        return instance

    def load_problems(self, batch_size, aug_factor=1, prepare_dataset=None):
        if prepare_dataset is None:
            self.batch_size = batch_size
            depot, loc, prize, max_length = get_random_op_problems(
                batch_size, self.problem_size, self.prize_type
            )
        else:
            self.batch_size = prepare_dataset.shape[0]
            if prepare_dataset.shape[-1] == 3:
                depot, loc, prize = (
                    prepare_dataset[:, 0:1, :2],
                    prepare_dataset[:, 1:, :2],
                    prepare_dataset[:, :, -1],
                )
            else:
                depot, loc = prepare_dataset[:, 0:1, :2], prepare_dataset[:, 1:, :2]
                prize_ = ((depot - prepare_dataset[:, :, :2]) ** 2).sum(-1).sqrt()
                prize = 1 + (prize_ / prize_.max(dim=-1, keepdims=True)[0] * 99).to(
                    torch.int32
                )
                prize = prize / 100
            max_length = self.get_length(self.problem_size) * torch.ones(
                size=(self.batch_size, 1)
            )

        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                loc = augment_xy_data_by_8_fold(loc)
                prize = prize.repeat(8, 1)
                max_length = max_length.repeat(8, 1)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError
        self.depot_xy = depot
        self.node_xy = loc
        self.depot_node_xy = torch.cat([self.depot_xy, self.node_xy], dim=1)
        self.coords = self.depot_node_xy[:, None, :, :].expand(
            self.batch_size, self.pomo_size, self.problem_size + 1, -1
        )

        self.prize = prize[:, None, :].expand(
            self.batch_size, self.pomo_size, 1 + self.problem_size
        )
        self.max_length = (
            max_length[:, None, :]
            .expand(self.batch_size, self.pomo_size, 1)
            .squeeze(-1)
        )

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(
            self.batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.batch_size, self.pomo_size
        )

    def reset(self):
        self.current_node = None
        # shape: (batch, pomo)
        self.previous_node = None
        # shape: (batch, pomo)

        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long
        )
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_OP_State(
            BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX
        )
        self.step_state.finished = torch.zeros((self.batch_size, self.pomo_size)) == 1

        self.step_state.visit_inf_mask = torch.zeros(
            (self.batch_size, self.pomo_size, 1 + self.problem_size)
        )
        self.step_state.ninf_mask = torch.zeros(
            (self.batch_size, self.pomo_size, 1 + self.problem_size)
        )
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        self.length = torch.zeros((self.batch_size, self.pomo_size))
        self.cur_total_prize = torch.zeros((self.batch_size, self.pomo_size))
        return (
            Reset_OP_State(self.depot_xy, self.node_xy, self.prize, self.max_length),
            reward,
            done,
        )

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        self.step_state.selected_count += 1

        if self.previous_node is None:
            pre_coord = self.depot_xy.expand(self.batch_size, self.pomo_size, -1)
        else:
            pre_coord = self.coords[
                self.BATCH_IDX, self.POMO_IDX, self.previous_node, :
            ]
        cur_coord = self.coords[self.BATCH_IDX, self.POMO_IDX, selected, :]
        dist = (pre_coord - cur_coord).norm(p=2, dim=-1).squeeze(-1)
        self.length += dist
        self.step_state.remain_dist = self.max_length - self.length
        self.current_node = selected
        self.step_state.current_node = self.current_node
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )
        self.previous_node = self.current_node

        self.step_state.visit_inf_mask[
            self.BATCH_IDX, self.POMO_IDX, self.current_node
        ] = float("-inf")
        if self.step_state.selected_count > 1:
            # if back to depot, forbid to choose any other nodes
            idx = torch.where(self.current_node == 0)
            self.step_state.visit_inf_mask[idx[0], idx[1], :] = float("-inf")
        self.step_state.visit_inf_mask[:, :, 0] = 0
        self.step_state.ninf_mask = self.step_state.visit_inf_mask.clone()
        # judge1: mask the nodes exceeding the max length if added
        next_step_len = self.length[:, :, None] + (
            self.coords - cur_coord[:, :, None, :]
        ).norm(p=2, dim=-1)
        next_step_jud = (next_step_len - self.max_length[:, :, None]) > 0
        # judge2: mask the next nodes exceeding the max length if unable to back to the depot
        next_step_backto_depot_len = next_step_len + (
            self.coords
            - self.depot_xy[:, None, ...].expand(self.batch_size, self.pomo_size, 1, -1)
        ).norm(p=2, dim=-1)
        next_step_backto_depot_jud = (
            next_step_backto_depot_len - self.max_length[:, :, None]
        ) > 0

        self.step_state.ninf_mask[
            torch.where((next_step_jud | next_step_backto_depot_jud) == True)
        ] = float("-inf")

        selected_prize = self.prize[self.BATCH_IDX, self.POMO_IDX, selected]
        self.cur_total_prize += selected_prize
        done = (self.step_state.ninf_mask[:, :, 1:] == float("-inf")).all() and (
            selected == 0
        ).all()
        if done:
            # because pomo start to search at each node, so it may fail for op20 cause the max length is 2,
            # so we set the prize equal to 0 if it violates the constrain
            self.cur_total_prize[torch.where((self.length - self.max_length) > 0)] = 0
            if self.problem_size >= 50:
                assert (self.length <= self.max_length + 1e-5).all()
        else:
            self.step_state.ninf_mask[:, :, 0] = 0
        return self.step_state, self.cur_total_prize, done

    def _get_travel_distance(
        self,
    ):
        pi = self.selected_node_list
        if (
            pi.size(-1) == 1
        ):  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            # Return
            return (
                torch.zeros(
                    (pi.size(0), pi.size(1)), dtype=torch.float, device=pi.device
                ),
                None,
            )

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(2)[0]
        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_pi[:, :, 1:] == 0) | (sorted_pi[:, :, 1:] > sorted_pi[:, :, :-1])
        ).all(), "Duplicates"
        prize_with_depot = self.prize

        p = prize_with_depot.gather(2, pi)

        # Gather dataset in order of tour
        loc_with_depot = self.coords
        d = loc_with_depot.gather(
            2, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1))
        )

        length = (
            (d[:, :, 1:] - d[:, :, :-1])
            .norm(p=2, dim=-1)
            .sum(2)  # Prevent error if len 1 seq
            + (d[:, :, 0] - d[:, :, 0]).norm(p=2, dim=-1)  # Depot to first
            + (d[:, :, -1] - d[:, :, 0]).norm(
                p=2, dim=-1
            )  # Last to depot, will be 0 if depot is last
        )
        self.step_state.remain_dist = self.max_length - length
        # assert (length <= self.max_length + 1e-5).all(), \
        #     "Max length exceeded by {}".format((length - self.max_length).max())
        pr = p.sum(-1)

        assert torch.mean(length - self.length) < 1e-4
        assert (pr - self.cur_total_prize).mean() < 1e-4
        # We want to maximize total prize but code minimizes so return negative
        return p.sum(-1)


def rand_pick(mask):
    batch_size, pomo_size = mask.size(0), mask.size(1)
    prob = torch.ones_like(mask) + mask
    prob = torch.softmax(prob, dim=-1)
    selected = (
        prob.reshape(batch_size * pomo_size, -1)
        .multinomial(1)
        .squeeze(dim=1)
        .reshape(batch_size, pomo_size)
    )
    return selected


if __name__ == "__main__":
    from tqdm import tqdm

    env_param = {"problem_size": 20, "pomo_size": 2, "prize_type": "dist"}
    bs = 2
    pomo_s = env_param["pomo_size"]
    env = OPEnv(**env_param)
    for _ in tqdm(range(10000000)):
        env.load_problems(bs)
        env.reset()
        done = False
        i = 0
        mask = env.step_state.ninf_mask
        while not done:
            if i == 0:
                selected = torch.zeros(bs, pomo_s).long()
            elif i == 1:
                selected = torch.randint(
                    1, env_param["problem_size"] + 1, (bs, pomo_s)
                ).long()
            else:
                selected = rand_pick(mask)
            state, rew, done = env.step(selected)
            mask = state.ninf_mask
            i += 1
