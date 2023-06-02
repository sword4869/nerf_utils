import numpy as np

def normalize_vec(x:np.ndarry):
    '''
    向量的单位化: 向量除以自己的二范数，得到和这个向量方向相同的单位向量(向量的范数为1)。
    '''
    return x / np.linalg.norm(x)


def normalize_features(x):
    '''
    归一化，所有的feature位于[0.0, 1.0]之间.
    所谓uint8[0, 255]图像到float[0.0, 1.0], img = img / 255。即 (x - 0) / (255 - 0)，这里0和255必须是指定的，而不是 x.max()和x.min()。
    '''
    # (x.max() - x/min()) 是一个scalar
    # (x - x.min()) 是 ndarry 对 scalar的减法，还是一个ndarry
    # 然后除法后还是 ndarry
    return (x - x.min()) / (x.max() - x/min())


def normalize_double(x):
    '''
    每个值均在 [-1.0, 1.0] 之间
    '''
    # 由[0,1]，再[0,2]，再[-1,1]
    n1 = normalize_features(x)
    n2 = n1 * 2 - 1
    return n2