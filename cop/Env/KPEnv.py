from dataclasses import dataclass
import torch
from .KPProblemDef import get_random_problems


@dataclass
class Reset_State:
    problems: torch.Tensor=None
    capacity: float =None
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor=None
    POMO_IDX: torch.Tensor=None
    # shape: (batch, pomo)
    selected_count: int = 0
    items_and_a_dummy: torch.Tensor = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    accumulated_value:torch.Tensor = None
    capacity:torch.Tensor = None
    ninf_mask_w_dummy: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    fit_ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class KPEnv:
    def __init__(self, **env_params):
        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.saved_item_data = None
        self.saved_index = None

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_item_data = loaded_dict['item_data']
        self.saved_index = 0

    def generate_data(self,batch_size):
        return get_random_problems(batch_size, self.problem_size)

    def load_problems(self, batch_size, aug_factor=1, prepare_dataset=None):
        if prepare_dataset is None:
            self.batch_size = batch_size
            self.problems = get_random_problems(batch_size, self.problem_size)
        else:
            self.batch_size = prepare_dataset.shape[0]
            self.problems = prepare_dataset

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        if self.problem_size >= 20 and self.problem_size < 50:
            capacity = 6.25
        elif self.problem_size >= 50 and self.problem_size <100:
            capacity = 12.5
        elif self.problem_size >= 100 and self.problem_size <200:
            capacity = 25
        elif self.problem_size >= 200:
            capacity = 25
        else:
            raise NotImplementedError

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.accumulated_value = torch.zeros((self.batch_size, self.pomo_size))
        # shape = (batch, group)
        self.step_state.capacity = torch.ones((self.batch_size, self.pomo_size)) * capacity
        # shape = (batch, group)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape = (batch, group, problem)
        self.step_state.fit_ninf_mask = None
        self.step_state.finished = torch.zeros((self.batch_size, self.pomo_size))==1.

        reward = None
        done = False
        return Reset_State(problems=self.problems, capacity=capacity), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done


    def step(self, selected):
        # selected.shape: (batch, pomo)
        # Dynamic-1
        ####################################
        self.step_state.selected_count += 1
        self.step_state.current_node = selected
        # shape: (batch, pomo)

        # Dynamic-2
        ####################################
        items_mat = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        gathering_index = selected[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, 2)
        selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape = (batch, pomo, 2)

        selected_item[self.step_state.finished[:,:,None].expand(self.batch_size, self.pomo_size, 2)==True] = 0.

        self.step_state.accumulated_value += selected_item[:, :, 1]
        self.step_state.capacity -= selected_item[:, :, 0]

        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        unfit_bool = (self.step_state.capacity[:, :, None] - self.problems[:, None, :, 0]) < 0
        # shape = (batch, group, problem)
        self.step_state.fit_ninf_mask = self.step_state.ninf_mask.clone()
        self.step_state.fit_ninf_mask[unfit_bool] = float('-inf')

        self.step_state.finished = (self.step_state.fit_ninf_mask == float('-inf')).all(dim=2)
        # shape = (batch, group)
        if (self.step_state.finished==True).any():
            idx= torch.where(self.step_state.finished==True)
            # idx_ = torch.zeros_like(idx[0]).to(torch.int8)
            self.step_state.fit_ninf_mask[idx[0],idx[1],0] = 0

        # returning values
        done = self.step_state.finished.all()
        if done:
            reward = self.step_state.accumulated_value
        else:
            reward = None
        return self.step_state, reward, done


    # def old_step(self, selected):
    #     # selected.shape: (batch, pomo)
    #     # Dynamic-1
    #     ####################################
    #     self.step_state.selected_count += 1
    #     self.step_state.current_node = selected
    #     # shape: (batch, pomo)
    #     self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)
    #     # shape: (batch, pomo, 0~problem)
    #
    #     # Dynamic-2
    #     ####################################
    #     items_mat = self.step_state.items_and_a_dummy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size+1, 2)
    #     gathering_index = selected[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, 2)
    #     selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)
    #     # shape = (batch, pomo, 2)
    #
    #     selected_item[self.step_state.finished[:,:,None].expand(self.batch_size, self.pomo_size, 2)==True] = 0.
    #
    #     self.step_state.accumulated_value += selected_item[:, :, 1]
    #     self.step_state.capacity -= selected_item[:, :, 0]
    #
    #     self.step_state.ninf_mask_w_dummy[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
    #     # self.step_state.ninf_mask = self.step_state.ninf_mask_w_dummy[:, :, :self.problem_size]
    #
    #     unfit_bool = (self.step_state.capacity[:, :, None] - self.problems[:, None, :, 0]) < 0
    #     # shape = (batch, group, problem)
    #     self.step_state.fit_ninf_mask = self.step_state.ninf_mask.clone()
    #     self.step_state.fit_ninf_mask[unfit_bool] = float('-inf')
    #
    #     self.step_state.finished = (self.step_state.fit_ninf_mask == float('-inf')).all(dim=2)
    #     # shape = (batch, group)
    #     self.step_state.fit_ninf_mask[self.step_state.finished[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size)] = 0
    #     # do not mask finished episode
    #
    #     # returning values
    #     done = self.step_state.finished.all()
    #     if done:
    #         reward = self.step_state.accumulated_value
    #     else:
    #         reward = None
    #     return self.step_state, reward, done



def rand_pick(mask):
    batch_size, pomo_size = mask.size(0), mask.size(1)
    prob = torch.ones_like(mask)+mask
    prob = torch.softmax(prob,dim=-1)
    selected = prob.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
    return selected



if __name__=="__main__":
    from tqdm import tqdm
    env_param = {
        'problem_size':20,
        'pomo_size':1,
    }
    bs = 2
    pomo_s = env_param['pomo_size']
    env = KPEnv(**env_param)
    for _ in tqdm(range(10000000)):
        env.load_problems(bs)
        env.reset()
        done = False
        i =0
        mask = env.step_state.ninf_mask
        while not done:
            if i == 0:
                selected = torch.zeros(bs,pomo_s).long()
            elif i==1:
                selected = torch.randint(1,env_param['problem_size'],(bs,pomo_s)).long()
            else:
                selected = rand_pick(mask)
            state, rew, done = env.step(selected)
            mask = state.ninf_mask
            i+=1

