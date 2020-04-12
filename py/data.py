"""
save and load dataset
"""

import numpy as np

DEFAULT_NT = 1_000
DEFAULT_NB = 1_000_000
DEFAULT_NQ = 1 # search one vector
DEFAULT_D = 576
DEFAULT_K = 5


class Data:
    def __init__(self,
                 nt=DEFAULT_NT,
                 nb=DEFAULT_NB,
                 nq=DEFAULT_NQ,
                 d=DEFAULT_D,
                 k=DEFAULT_K):
        """
        d : dimension
        k : nearest neighbors
        nt: train number
        nb: database number
        nq: query number
        """
        assert d % 8 == 0
        self.nt = nt
        self.nb = nb
        self.nq = nq
        self.d = d
        self.k = k
        self.xt = None
        self.xb = None
        self.xq = None

    def __make_binary_dataset(self):
        rs = np.random.RandomState(123)
        size = (self.nb + self.nq + self.nt, int(self.d / 8))
        x = rs.randint(256, size=size).astype('uint8')
        return x[:self.nt], x[self.nt:-self.nq], x[-self.nq:]

    def save_dataset(self):
        """generate dataset and write to disk.
        NOTE: This may be very slow!!
        """
        (xt, xb, xq) = self.__make_binary_dataset()
        print("start: write dataset xt: %s, xb: %s, xq: %s to disk!" %
              (xt.shape, xb.shape, xq.shape))
        np.savetxt("xt.txt", xt, fmt='%d', delimiter=',')
        np.savetxt("xb.txt", xb, fmt='%d', delimiter=',')
        np.savetxt("xq.txt", xq, fmt='%d', delimiter=',')
        print("done: write dataset to disk!")

    def load_dataset(self):
        """load dataset from disk. Assume: dataset is already generated.
        NOTE: This may be very slow!!
        """
        print("start: load dataset from disk!")
        self.xt = np.loadtxt("xt.txt", dtype='uint8', delimiter=',')
        self.xb = np.loadtxt("xb.txt", dtype='uint8', delimiter=',')
        self.xq = np.loadtxt("xq.txt", dtype='uint8', delimiter=',')
        print("done: load dataset xt: %s, xb: %s, xq: %s from disk!" %
              (self.xt.shape, self.xb.shape, self.xq.shape))


if __name__ == '__main__':
    data = Data()
    data.save_dataset()
