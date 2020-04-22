"""
using binary index with hamming distance
"""

from sys import argv
from data import Data
import numpy as np
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
              ((end - start) / TIMES))

        index = faiss.IndexBinaryFlat(self.data.d)
        index.add(self.data.xb)

        print("start: benchmark faiss BinaryFlat Search performance")
        start = time.time()
        for _x in range(TIMES):
            index.search(self.data.xq, self.data.k)
        end = time.time()
        print("end: BinaryFlat Search consumes: %.4f (s)" %
              ((end - start) / TIMES))

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

    def flat_with_ids(self):
        """
        1. add 10 dataset with id [100, 110]
        2. remove 110
        3. add [11, 121] to index with id 109 # now 2 vectors with 109 id
        4. search [11,121] and [9,81] check result
        5. remove 109
        6. now left 8 vectors in index
        """
        d = 8 * 2
        arr = [[x, x * x] for x in range(1, 11)]
        ids = np.arange(101, 111)
        xb = np.array(arr, dtype='uint8')

        sub_index = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIDMap2(sub_index)
        index.add_with_ids(xb, ids)

        _D, I = index.search(np.array([[9, 81]], dtype='uint8'), 1)
        assert I[0][0] == 109

        id_to_remove = 110
        assert index.ntotal == 10
        assert index.reconstruct(id_to_remove)[0] == 10
        assert index.reconstruct(id_to_remove)[1] == 100

        ids_to_remove = np.array([id_to_remove])
        n = index.remove_ids(ids_to_remove)
        assert n == 1  # only one vector is removed

        assert index.ntotal == 9
        try:
            index.reconstruct(id_to_remove)
        except:
            # print("exception is by design")
            pass
        else:
            assert False, 'should have raised an exception'

        id_to_replace = 109
        new_xb = np.array([[11, 121]], dtype='uint8')
        ids_to_replace = np.array([id_to_replace])  # replace id=109
        # after this add, there are 2 vectors with the same 109 id
        index.add_with_ids(new_xb, ids_to_replace)

        assert index.ntotal == 10

        _D, I = index.search(np.array([[9, 81]], dtype='uint8'), 1)
        assert I[0][0] == id_to_replace  # one vector with 109 id

        _D, I = index.search(np.array([[11, 121]], dtype='uint8'), 1)
        assert I[0][0] == id_to_replace  # another vector with 109 id

        n = index.remove_ids(np.array([id_to_replace]))
        assert n == 2

        assert index.ntotal == 8


if __name__ == '__main__':
    data = Data()
    data.load_dataset()  # slow!!

    faiss_binary_index = FaissBinaryIndex(data)

    if len(argv) == 2 and argv[1] == "ivf":
        faiss_binary_index.ivf_flat()
    elif len(argv) == 2 and argv[1] == "ids":
        faiss_binary_index.flat_with_ids()
    else:
        faiss_binary_index.flat()
