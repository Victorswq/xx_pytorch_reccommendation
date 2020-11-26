import numpy as np
import torch
from torch import nn
from copy import deepcopy


def HR(ratingss, pos_items, top_k=[10, 20]):
    """
    a special hit ratio that for the negative test, not for the full test
    :param ratingss:
    :param pos_items:
    :param top_k:
    :return:
    """
    print("rating_shape is ",ratingss.shape," target shape is ",pos_items.shape)
    length = len(ratingss)
    s=[]
    for k in top_k:
        ratings = deepcopy(ratingss)
        hr = 0
        for idx in range(length):
            rating = ratings[idx]
            rating[np.argsort(rating)[:-k]] = 0
            if rating[-1]!=0:
                hr+=1
        s.append("-------------------------------------------------------HR@%d:  %.3f" % (k, hr / length))
    return s


def HitRatio(ratingss, pos_items, top_k=[10, 20]):
    print("rating_shape is ", ratingss.shape, " target shape is ", pos_items.shape)
    length = len(ratingss)
    print(length)
    for k in top_k:
        ratings = deepcopy(ratingss)
        hr = 0
        for idx in range(length):
            rating = ratings[idx]
            rank = np.argsort(rating)[-k:]
            for index in range(len(rank)):
                if rank[index] == pos_items[idx]:
                    hr += 1
                    break
        print("-------------------------------------------------------HR@%d:  %.3f" % (k, hr / length))


def MRR(ratingss,pos_items,top_k=[10,20]):
    print("rating_shape is ", ratingss.shape, " target shape is ", pos_items.shape)
    length=len(ratingss)
    for k in top_k:
        ratings=deepcopy(ratingss)
        mrr=0
        for idx in range(length):
            rating=ratings[idx]
            rank=np.argsort(rating)[-k:]
            for index in range(len(rank)):
                if rank[index]==pos_items[idx]:
                    mrr+=1/(k-index)
                    break
        print("------------------------------------------------------MRR@%d:   %.3f"%(k,mrr/length))


def Precision(ratingss,pos_items,top_k=[10,20]):
    length=len(ratingss)
    for k in top_k:
        ratings=deepcopy(ratingss)
        p=0
        for idx in range(length):
            rating=ratings[idx]
            rank=np.argsort(rating)[-k:]
            for index in range(len(rank)):
                if rank[index]==pos_items[idx]:
                    p+=1/k
                    break
        print("-------------------------------------------------------Precision@%d:    %.3f"%(k,p/length))


# def NDCG()