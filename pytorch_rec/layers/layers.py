from utils import build_map
import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_


class MLPLayers(nn.Module):
    def __init__(self,layers,dropout=0.,activation="relu",bn=False,init_method=None):
        super(MLPLayers,self).__init__()
        self.layers=layers
        self.dropout=dropout
        self.activation=activation
        self.use_bn=bn
        self.init_method=init_method

        mlp_module=[]

        for idx,(input_size,output_size) in enumerate(self.layers):
            mlp_module.append(nn.Dropout(p=self.dropout))
            mlp_module.append(nn.Linear(input_size,output_size))
            if self.use_bn is True:
                mlp_module.append(nn.BatchNorm1d(num_features=output_size))
            if self.activation.lower() is "relu":
                mlp_module.append(nn.ReLU())
            elif self.activation.lower() is "sigmoid":
                mlp_module.append(nn.Sigmoid())
            elif self.activation.lower() is "tanh":
                mlp_module.append(nn.Tanh())
            elif self.activation.lower() is "leakyrelu":
                mlp_module.append(nn.LeakyReLU())
            elif self.activation.lower() is "none":
                pass
            else:
                print("you have to input a right activation like relu, tanh, leakyrelu, sigmoid, or none")
        self.mlp_layers=nn.Sequential(*mlp_module)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self,module):
        if isinstance(module,nn.Linear):
            if self.init_method is "norm":
                normal_(module.weight.data,0.,0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.)


class MLPLayerss(nn.Module):
    def __init__(self,layers,dropout=0,activation="relu",bn=False,init_method=None):
        super(MLPLayerss, self).__init__()
        self.layers=layers
        self.dropout=dropout
        self.activation=activation
        self.use_bn=bn
        self.init_method=init_method

        mlp_modules=[]

        for idx,(input_size,output_size) in enumerate(zip(self.layers[:-1],self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size,output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            if self.activation.lower() is "sigmoid":
                mlp_modules.append(nn.Sigmoid())
            elif self.activation.lower() is "tanh":
                mlp_modules.append(nn.Tanh())
            elif self.activation.lower() is "relu":
                mlp_modules.append(nn.ReLU())
            elif self.activation.lower() is "leakyrelu":
                mlp_modules.append(nn.LeakyReLU())
            elif self.activation.lower() is None:
                pass
            else:
                print("you got something for activation function like relu, tanh, sigmoid")
        self.mlp_layers=nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_)

    def init_weights(self,module):
        if isinstance(module,nn.Linear):
            if self.init_method is "norm":
                normal_(module.weight.data,0.,0.01)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self,input_feature):
        return self.mlp_layers(input_feature)


class FMEmbedding(nn.Module):
    def __init__(self,item_num,embedding_size,offsets,is_padding=True):
        super(FMEmbedding, self).__init__()
        self.item_num=item_num
        self.embedding_size=embedding_size
        self.offsets=offsets
        self.is_padding=is_padding
        if self.is_padding is True:
            self.item_embedding_matrix=nn.Embedding(self.item_num,self.embedding_size,padding_idx=0)
        else:
            self.item_embedding_matrix=nn.Embedding(self.item_num,self.embedding_size)

    def forward(self,input_x):
        input_x=input_x+input_x.new_tensor(self.offsets).unsqueeze(0)
        output=self.item_embedding_matrix(input_x)
        return output


class BaseFactorizationMachine(nn.Module):
    # batch_size * seq_len * embedding_size ==>>
    def __init__(self,reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.is_reduce_sum=reduce_sum

    def forward(self,input_batch_seq_embedding):
        add=torch.sum(input_batch_seq_embedding,dim=1)
        add_square=add**2
        # batch_size * embedding_size
        square=input_batch_seq_embedding**2
        square_add=torch.sum(square,dim=1)
        # batch_size * embedding_size
        result=0.5*(add_square-square_add)
        if self.is_reduce_sum is True:
            result=torch.sum(result,dim=1,keepdim=True)
        return result


class BiGNNLayer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.linear=nn.Linear(in_dim,out_dim)
        self.interActTransform=nn.Linear(in_dim,out_dim)

    def forward(self,lap_matrix,eye_matrix,features):
        x=torch.sparse.mm(lap_matrix,features)


class Repeat_explore_mechanism(nn.Module):

    def __init__(self,hidden_size):
        super(Repeat_explore_mechanism, self).__init__()
        self.hidden_size=hidden_size
        self.Wre=nn.Linear(hidden_size,hidden_size,bias=False)
        self.Ur=nn.Linear(hidden_size,hidden_size,bias=False)
        self.tanh=nn.Tanh()
        self.Vre=nn.Linear(hidden_size,1,bias=False)
        self.Wre_last=nn.Linear(hidden_size,2)
        self.softmax=nn.Softmax()


    def forward(self,last_memory,all_memory):
        """
        :param last_memory: batch_size * hidden_size
        == Wre ==>> batch_size * hidden_size
        == unsqueeze ==> batch_size * 1 * hidden_size
        == repeat ==> batch_size * seq_len * hidden_size
        :param all_memory: batch_size * seq_len * hidden_size
        == Ur ==> batch_size * seq_eln * hidden_size
        ***
        last_memory + all_memory == batch_size * seq_len * hidden_size
        ***
        Are: Vre ==> batch_size * seq_len * 1
        == multiply all_memory ==>> batch_size * seq_len * hidden_size
        == sum ==> batch_size * hidden_size
        ***
        == Wre_last ==> batch_size * 2
        == soft_max ==> result
        :return:
        """
        batch_size,seq_len,hidden_size=all_memory.size()
        all_memory_values=all_memory
        last_memory=self.Wre(last_memory)
        last_memory=last_memory.unsqueeze(1).repeat(1,seq_len,1)
        all_memory=self.Ur(all_memory)

        memory=self.tanh(all_memory+last_memory)
        memory=self.Vre(memory)
        memory=memory*all_memory_values
        memory=memory.sum(1)

        memory=self.Wre_last(memory)
        memory=nn.Softmax(dim=-1)(memory)
        return memory


class Repeat_recommendation_decoder(nn.Module):

    def __init__(self,hidden_size):
        super(Repeat_recommendation_decoder, self).__init__()
        self.hidden_size=hidden_size
        self.Wr=nn.Linear(hidden_size,hidden_size,bias=False)
        self.Ur=nn.Linear(hidden_size,hidden_size,bias=False)
        self.tanh=nn.Tanh()
        self.Vr=nn.Linear(hidden_size,1,bias=False)
        self.softmax=nn.Softmax()


    def forward(self,seq_item,last_memory,all_memory,mask,item_matrix):
        """
        :param last_memory: batch_size * hidden_size
        == Wr ==> batch_size * hidden_size
        == unsqueeze ==> batch_size * 1 * hidden_size
        :param all_memory: batch_size * seq_len * hidden_size
        == Ur ==> batch_size * seq_len * hidden_size
        ***
        last_memory + all_memory ==> batch_size * seq_len * embedding_size
        ***
        == Vr ==> batch_size * seq_len * 1
        == squeeze ==> batch_size * seq_len
        :return:
        """
        batch_size,seq_len,hidden_size=all_memory.size()
        embedding_size,num_item=item_matrix.size()
        last_memory=self.Wr(last_memory).unsqueeze(1).repeat(1,seq_len,1)

        all_memory=self.Ur(all_memory)

        memory=last_memory+all_memory
        memory=self.tanh(memory)

        memory=self.Vr(memory).squeeze(-1)
        memory.masked_fill_(mask,-1e9)
        memory=nn.Softmax(dim=-1)(memory)
        memory=memory.unsqueeze(1)

        map=build_map(seq_item,max=num_item)
        """
        ******************************* 
        something should be fixed because you have no ideal about it
        !!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        memory=torch.matmul(memory,map).squeeze(1)
        # batch_size * seq_len * num_item
        # batch_size * num_item
        return memory


class Explore_recommendation_decoder(nn.Module):

    def __init__(self,hidden_size):
        super(Explore_recommendation_decoder, self).__init__()
        self.hidden_size=hidden_size
        self.We=nn.Linear(hidden_size,hidden_size,bias=False)
        self.Ue=nn.Linear(hidden_size,hidden_size,bias=False)
        self.tanh=nn.Tanh()
        self.Ve=nn.Linear(hidden_size,1,bias=False)
        self.softmax=nn.Softmax()


    def forward(self,last_memory,all_memory,item_matrix_transpose,mask):
        """
        :param last_memory: batch_size * hidden_size
        == We ==> batch_size * hidden_size
        == unsqueeze ==> batch_size * 1 * hidden_size
        == repeat ==> batch_size * seq_len * hidden_size
        :param all_memory:
        == Ue ==> batch_size * seq_len * hidden_size
        ***
        last_memory + all_memory == batch_size * seq_len * hidden_size
        == Ve ==> batch_size * seq_len * 1
        soft_max dim==1 ==> batch_size * seq_len * 1
        ***
        mul all_memory:
        == >> batch_size * seq_len * hidden_size
        == sum ==> batch_size * hidden_size
        ***
        concat the last_memory
        batch_size * 2_hidden_size
        ***
        matmul weight_matrix: batch_size * item_num
        batch_size * item_number
        :param item_matrix_transpose:
        :return:
        """
        batch_size,seq_len,hidden_size=all_memory.size()
        last_memory_values,all_memory_values=last_memory,all_memory
        last_memory=self.We(last_memory).unsqueeze(1).repeat(1,seq_len,1)

        all_memory=self.Ue(all_memory)

        memory=last_memory+all_memory

        memory=self.tanh(memory)
        memory=self.Ve(memory)
        memory=nn.Softmax(1)(memory)

        memory=memory*all_memory_values
        memory=memory.sum(1)
        memory=torch.cat([memory,last_memory_values],dim=1)
        memory=torch.matmul(memory,item_matrix_transpose)
        for x,y in zip(memory,mask):
            x[y]=-1e9
        memory=nn.Softmax(1)(memory)
        return memory


class feed_forward_layer(torch.nn.Module):
    def __init__(self,embedding_size,semi_size,dropout_rate=0.2):
        super(feed_forward_layer, self).__init__()
        self.embedding_size=embedding_size
        self.dropout_rate=dropout_rate
        self.semi_size=semi_size

        self.conv1_d=nn.Conv1d(in_channels=self.embedding_size,out_channels=self.semi_size,kernel_size=1)
        self.dropout_1=nn.Dropout(p=self.dropout_rate)
        self.conv2_d=nn.Conv1d(in_channels=self.semi_size,out_channels=self.embedding_size,kernel_size=1)
        self.dropout_2=nn.Dropout(p=self.dropout_rate)
        self.relu=nn.ReLU()


    def forward(self,inputs):
        residual,inputs=inputs,inputs.transpose(-1,-2)
        # batch_size * embedding_size * seq_len
        inputs=self.conv1_d(inputs)
        # inputs=self.relu(inputs)
        inputs=self.dropout_1(inputs)
        inputs=self.conv2_d(inputs)
        inputs=self.dropout_2(inputs)
        inputs=inputs.transpose(-1,-2)
        # batch_size * seq_len * embedding_size
        return inputs+residual