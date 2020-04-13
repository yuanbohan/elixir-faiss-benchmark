"""
using binary index with hamming distance
"""

from sys import argv
from data import Data
import faiss
import time

TIMES = 100


class FaissBinaryIndex:
    def __init__(self, data: Data):
        self.data = data

    def flat(self):
        print("dataset: xb: %s, xq: %s" %
              (self.data.xb.shape, self.data.xq.shape))

        print("start: benchmark faiss BinaryFlat Index performance")
        start = time.time()
        for _x in range(TIMES):
            index = faiss.IndexBinaryFlat(self.data.d)
            index.add(self.data.xb)
        end = time.time()
        print("end: BinaryFlat Index consumes: %.4f (s)" %
              ((end - start) / 100))

        index = faiss.IndexBinaryFlat(self.data.d)
        index.add(self.data.xb)

        print("start: benchmark faiss BinaryFlat Search performance")
        start = time.time()
        for _x in range(TIMES):
            index.search(self.data.xq, self.data.k)
        end = time.time()
        print("end: BinaryFlat Search consumes: %.4f (s)" %
              ((end - start) / 100))

        # index = faiss.IndexBinaryFlat(self.data.d)
        # index.add(self.data.xb)
        # D, I = index.search(self.data.xq, self.data.k)
        # print(D)
        # print(I)
        # print(I[0])
        # print(self.data.xb[I[0]])

    def ivf_flat(self):
        print("start: benchmark faiss BinaryIVF performance")
        nlist, nprobe = 8, 8
        quantizer = faiss.IndexBinaryFlat(self.data.d)
        index = faiss.IndexBinaryIVF(quantizer, self.data.d, nlist)
        index.cp.min_points_per_centroid = 5  # quiet warning
        index.nprobe = nprobe
        index.train(self.data.xt)
        index.add(self.data.xb)

        D, I = index.search(self.data.xq, self.data.k)
        print(D)
        print(I)


if __name__ == '__main__':
    data = Data()
    data.load_dataset()  # slow!!

    faiss_binary_index = FaissBinaryIndex(data)

    if len(argv) == 2 and argv[1] == "all":
        faiss_binary_index.flat()
        faiss_binary_index.ivf_flat()
    elif len(argv) == 2 and argv[1] == "ivf":
        faiss_binary_index.ivf_flat()
    else:
        faiss_binary_index.flat()
