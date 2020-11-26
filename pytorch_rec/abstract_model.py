import torch
from dataset.utils import *
from logger.logger import Logger
import os
import time
import torch.nn as nn
from torch.nn.init import xavier_normal_,constant_
from time import time as timee
from Dataset import DataSet


class abstract_model(torch.nn.Module):
    def __init__(self,data_name="ml-1m",model_name="SASRec",min_user_number=5,min_item_number=5):
        super(abstract_model, self).__init__()
        self.data_name=data_name
        self.model_name=model_name
        self.min_user_number=min_user_number
        self.min_item_number=min_item_number

        road = os.path.abspath(os.path.join(os.getcwd()))
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logger = Logger("%s\log\%s\%s" % (road, self.model_name, str(localtime)))


    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module,nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data,0)


    def logging(self):
        self.logger.info("------------------"+str(self.model_name)+"--------------------")
        self.logger.info("learning_rate:"+str(self.learning_rate))
        self.logger.info("batch_size:"+str(self.batch_size))
        self.logger.info("embedding_size:"+str(self.embedding_size))
        self.logger.info("number_of_epochs:"+str(self.episodes))
        self.logger.info("verbose:"+str(self.verbose))
        self.logger.info("data_name: " + str(self.data_name))
        self.logger.info("num_user:"+str(self.user_number))
        self.logger.info("num_items:"+str(self.item_number))
        # self.logger.info("max_len:" + str(self.max_seq_length))