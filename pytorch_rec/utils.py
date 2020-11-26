import torch
from torch import nn
import numpy as np


def multi_head_attentive(n_head,seq_q,seq_k,seq_v,mask):
    """

    :param n_head: the number of split head
    :param seq_q: batch_size * seq_len * embedding_size
    :param seq_k: batch_size * seq_len * embedding_size
    :param seq_v: batch_size * seq_len * embedding_size
    :param mask : batch_size * seq_len * seq_len
    :return: the value from self_attentive
    """
    batch_size,seq_len,embedding_size=seq_q.size()
    layer_norm = nn.LayerNorm(embedding_size)
    residual,seq_q,seq_k,seq_v=seq_v,layer_norm(seq_q),layer_norm(seq_k),layer_norm(seq_v)

    matrix_q=nn.Linear(embedding_size,embedding_size,bias=False)
    matrix_k=nn.Linear(embedding_size,embedding_size,bias=False)
    matrix_v=nn.Linear(embedding_size,embedding_size,bias=False)
    fc=nn.Linear(embedding_size,embedding_size,bias=False)

    seq_q=matrix_q(seq_q)
    seq_q=seq_q.view(batch_size,seq_len,n_head,-1).transpose(1,2)

    seq_k=matrix_k(seq_k)
    seq_k=seq_k.view(batch_size,seq_len,n_head,-1).transpose(1,2)

    seq_v=matrix_v(seq_v)
    seq_v=seq_v.view(batch_size,seq_len,n_head,-1).transpose(1,2)
    """
    the change of seq_q, seq_k, seq_v: 
    [batch_size * seq_len * embedding_size] == matrix_q ==>> [batch_size * seq_len * embedding_size] == multi_head ==>> [batch_size * n_head * seq_len * embedding_size]
    """
    q_k=torch.matmul(seq_q,seq_k.transpose(2,3))/torch.sqrt(embedding_size/n_head)
    # batch_size * n_head * seq_len * seq_len

    # change the shape of the mask for the q_k
    mask.unsqueeze(1).repeat(1,n_head,1,1)
    # batch_size * n_head * seq_len * seq_len
    q_k.masked_fill(mask,-1e9)

    attention=nn.Softmax(q_k,dim=-1)
    attention_v=torch.matmul(attention,seq_v)
    # batch_size * n_head * seq_len * embedding_size
    attention_v.transpose(1,2).view(batch_size,seq_len,-1)
    # batch_size * seq_len * embedding_size

    attention_v=fc(attention_v)
    # batch_size * seq_len * embedding_size
    attention_v+=residual

    return layer_norm(attention_v)


def feed_forward(attention_v,dropout_rate,semi_transpose):
    """

    :param attention_v: batch_size * seq_len * embedding_size
    :param dropout_rate: the rate to loss
    :param semi_transpose: the semi number in the feed_forward networks
    :return:
    """
    residual=attention_v
    batch_size,seq_len,embedding_size=attention_v.size()
    layer_norm=nn.LayerNorm(embedding_size)
    conv1d_1=nn.Conv1d(embedding_size,semi_transpose)
    conv1d_2=nn.Conv1d(semi_transpose,embedding_size)
    dropout_1=nn.Dropout(p=dropout_rate)
    dropout_2=nn.Dropout(p=dropout_rate)

    attention_v=attention_v.transpose(1,2)
    # batch_size * embedding_size * seq_len
    attention_v=conv1d_1(attention_v)
    attention_v=dropout_1(attention_v)
    attention_v=nn.ReLU(attention_v)
    attention_v=conv1d_2(attention_v)
    attention_v=dropout_2(attention_v)
    attention_v=attention_v.transpose(1,2)
    # batch_size * seq_len * embedding_size
    attention_v+=residual

    return layer_norm(attention_v)


def get_padding_mask(seq_data):
    batch_size,seq_len=seq_data.shape
    mask=(seq_data==0)
    mask=np.expand_dims(mask,1)
    print("final try ",mask.shape)
    mask=np.tile(mask,[1,seq_len,1])
    return mask


def get_seq_mask(seq_data):
    batch_size,seq_len=seq_data.shape
    mask=torch.ones([batch_size,seq_len,seq_len],dtype=torch.uint8)
    mask=mask.triu(diagonal=1)
    return mask


def build_map(b_map, max=None):
    batch_size, b_len = b_map.size()
    if max is None:
        max=b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # batch_size *
    b_map_.requires_grad=False
    return b_map_