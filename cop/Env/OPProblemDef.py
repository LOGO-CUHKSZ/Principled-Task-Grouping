import torch


def get_random_op_problems(batch_size, size, prize_type):
    # Details see paper
    def get_length(scale):
        if scale<=20:
            leng = 2.
        elif scale>20 and scale<=50:
            leng = 3.
        elif scale > 50 and scale <= 100:
            leng = 4.
        else:
            leng = 5.
        return leng

    loc = torch.rand(batch_size, size, 2)
    depot = torch.rand(batch_size,1,2)
    # Methods taken from Fischetti et al. 1998
    prize = torch.zeros(batch_size,1+size)
    if prize_type == 'const':
        prize[:, 1:] = 1
    elif prize_type == 'unif':
        prize[:,1:] = (1 + torch.randint(0, 100, size=(batch_size,size, ))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = (depot - loc).norm(p=2, dim=-1)
        prize[:,1:] = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.
    return depot, loc, prize, torch.ones(size=(batch_size,1))*get_length(size)


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data