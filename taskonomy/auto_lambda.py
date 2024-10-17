import copy
import torch

class AutoLambda(torch.nn.Module):
    def __init__(self, model, model_, criteria, num_tasks, weight_init=0.1):
        super(AutoLambda, self).__init__()
        self.model = model
        self.model_ = model_
        self.criteria = criteria
        self.num_tasks = num_tasks
        self.meta_weights = torch.tensor([weight_init] * num_tasks, requires_grad=True)
        self.meta_weights = torch.nn.Parameter(self.meta_weights)

    def forward(self,):
        pass

    def virtual_step(self, train_x, train_y, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """
        train_pred = self.model(train_x)

        train_loss = self.model_fit(train_pred, train_y)

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

    def unrolled_backward(self, train_x, train_y, val_x, val_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in range(self.num_tasks):
            pri_weights += [1.0]

        # compute validation data loss on primary tasks
        val_pred = self.model_(val_x)
        val_loss = self.model_fit(val_pred, val_y)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = - alpha * h

    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def model_fit(self, pred, targets):
        """
        define task specific losses
        """
        loss_list = []        
        for c_name,criterion_fun in self.criteria.items():
            if c_name == 'Loss':
                loss = criterion_fun(pred, targets).clone()
                continue
            loss_list.append(criterion_fun(pred, targets).clone())
        return loss_list
