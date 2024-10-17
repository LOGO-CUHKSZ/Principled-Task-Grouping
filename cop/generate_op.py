import numpy as np
import os

def write_oplib(filename, depot, loc, prize, max_length, name="problem"):
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "OP"),
                ("DIMENSION", len(loc) + 1),
                ("COST_LIMIT", int(max_length * 10000000 + 0.5)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # oplib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("NODE_SCORE_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + prize)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def get_length(scale):
    if scale <= 20:
        leng = 2.
    elif scale > 20 and scale <= 50:
        leng = 3.
    elif scale > 50 and scale <= 100:
        leng = 4.
    else:
        leng = 5.
    return leng


if __name__ == '__main__':
    scales = [110]
    inst_num = 10000
    instances = []
    for scale in scales:
        if not os.path.isdir('oplib/op_{}'.format(scale)):
            os.makedirs('oplib/op_{}'.format(scale))
        instances_scale = np.random.random((inst_num, scale+1, 2))
        if not os.path.isdir('op_data/'):
            os.makedirs('op_data/')
        np.save('op_data/op_{}.npy'.format(scale), instances_scale)
        for i in range(inst_num):
            depot = instances_scale[i,0,:]
            loc = instances_scale[i,1:,:]
            prize_ = np.linalg.norm(depot[None, :] - loc, axis=-1)
            prize = 1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)
            max_length = get_length(scale)
            write_oplib('oplib/op_{}/inst_{}.oplib'.format(scale, i), depot.tolist(), loc.tolist(), prize.tolist(), max_length,name='op_{}_{}'.format(scale, i))

