from dataset.utils import *
from mertic import *
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from Dataset import DataSet
from abstract_model import abstract_model


class SSAP(abstract_model):
    """
    Sequential self-attentive with user preference

    ???
    it has been proved bad in the paper
    """

    def __init__(self,
                 model_name="SSAP",
                 data_name="ml-1m",
                 embedding_size=32,
                 n_head=1,
                 learning_rate=0.001,
                 episodes=10,
                 batch_size=512,
                 verbose=1,
                 seq_len=10,
                 ):
        super(SSAP, self).__init__(model_name=model_name,data_name=data_name,min_user_number=5)
        self.embedding_size=embedding_size
        self.n_head=n_head
        self.learning_rate=learning_rate
        self.episodes=episodes
        self.batch_size=batch_size
        self.verbose=verbose
        self.seq_len=seq_len

        self.dataset=Data_for_SSAP(data_name=self.data_name,seq_len=self.seq_len)

        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        self.build_variable()
        self.criterion = torch.nn.BCEWithLogitsLoss()


    def build_variable(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.multi_head_attention=MultiHeadAttention_plus(d_model=self.embedding_size)


    def get_item_matrix_transpose(self):

        return self.item_matrix.weight.t()


    def forward(self,data):
        """

        :param data: seq_item user
        seq_item: batch_size * seq_len
        user: batch_size
        :return:
        """
        seq_item,user=data
        seq_item_embedding=self.item_matrix(seq_item)
        user_embedding=self.user_matrix(user)
        batch_size,seq_len,embedding_size=seq_item_embedding.size()

        # get the mask
        padding_mask=seq_item.data.eq(0).unsqueeze(1)
        padding_mask=padding_mask.repeat(1,self.seq_len,1)
        # batch_size * seq_len * seq_len

        ones=torch.ones((batch_size,seq_len,seq_len),dtype=torch.uint8)
        subsequence_mask=ones.triu(diagonal=1)
        # batch_size * seq_len * seq_len

        mask=padding_mask+subsequence_mask
        mask=torch.gt((padding_mask+subsequence_mask),0)
        # batch_size * seq_len * seq_len

        seq_item_embedding=self.multi_head_attention.forward(seq_item_embedding,seq_item_embedding,seq_item_embedding,mask=mask,)
        # batch_size * seq_len * embedding_size

        user_embedding=user_embedding.unsqueeze(1).repeat(1,seq_len,1)
        # batch_size * seq_len * embedding_size

        user_item_embedding=seq_item_embedding+user_embedding
        # batch_size * seq_len * embedding_size

        return user_item_embedding


    def calculate_loss(self,data):
        seq_item,user,pos_item,neg_item=data

        user_item_embedding=self.forward([seq_item,user])
        # batch_size * seq_len * embedding_size

        pos_item_embedding=self.item_matrix(pos_item)
        # batch_size * seq_len * embedding_size

        neg_item_embedding=self.item_matrix(neg_item)
        # batch_size * seq_len * embedding_size

        pos_logits=(user_item_embedding*pos_item_embedding).sum(dim=-1)
        # batch_size * seq_len
        neg_logits=(user_item_embedding*neg_item_embedding).sum(dim=-1)
        # batch_size * seq_len

        pos_label=torch.ones_like(pos_logits)
        neg_label=torch.zeros_like(neg_logits)
        # batch_size * seq_len

        index=torch.where(seq_item!=0)

        loss=self.criterion(pos_logits[index],pos_label[index])
        loss+=self.criterion(neg_logits[index],neg_label[index])

        return loss


    def prediction(self,data):
        seq_item,user=data
        user_item_embedding=self.forward([seq_item,user])
        # batch_size * seq_len * embedding_size
        user_item_embedding=user_item_embedding[:,-1,:]
        # batch_size * embedding_size
        prediction=torch.matmul(user_item_embedding,self.get_item_matrix_transpose())
        # batch_size * item_number

        return prediction


class MultiHeadAttention(nn.Module):

    def __init__(self,embedding_size=32,n_head=1,):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size=embedding_size
        self.n_head=n_head
        self.W1=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.W2=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.W3=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.layer_norm=nn.LayerNorm(self.embedding_size)


    def forward(self,seq_k,seq_q,seq_v,mask=None,is_layer_norm=False,is_residual=False):
        """

        :param seq_k: batch_size * seq_len * embedding_size
        == view ==>> batch_size * seq_len * n_head * length
        == transpose ==>> batch_size * n_head * seq_len * length
        :param seq_q: batch_size * seq_len * embedding_size
        == view ==>> batch_size * seq_len * n_head * length
        == transpose ==>> batch_size * n_head * length * seq_len
        ***
        seq_k matmul seq_q: batch_size * n_head * seq_len * seq_len
        :param seq_v: batch_size * seq_len * embedding_size
        == view & transpose ==>> batch_size * n_head * seq_len * length
        :param mask: batch_size * seq_len * seq_len
        == view ==>> batch_size * 1 * seq_len * seq_len
        == repeat ==>> batch_size * n_head * seq_leb * seq_len
        ***
        get attn: batch_size * n_head * seq_len * seq_len
        attn matmul seq_v: batch_size * n_head * seq_len * length
        == transpose ==>> batch_size * seq_len * n_head * length
        == view ==>> batch_size * seq_len * embedding_size
        :return: batch_size * seq_len * embedding_size
        """
        if is_layer_norm is True:
            seq_v=self.layer_norm(seq_v)

        seq_v_value=seq_v

        batch_size,seq_len,embedding_size=seq_k.size()
        length=embedding_size/self.n_head

        seq_q=self.W1(seq_q)
        seq_k=self.W2(seq_k)
        seq_v=self.W3(seq_v)
        # after matrix == >> batch_size * embedding_size * embedding_size

        seq_k=seq_k.view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        seq_q=seq_q.view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        seq_v=seq_v.view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        seq_q=seq_q.transpose(2,3)

        q_k=torch.matmul(seq_k,seq_q)

        if mask is not None:
            mask=mask.view(batch_size,1,seq_len,seq_len).repeat(1,self.n_head,1,1)
            q_k.masked_fill_(mask,-1e9)

        attn = nn.Softmax(dim=-1)(q_k)
        value=torch.matmul(attn,seq_v)
        value=value.transpose(1,2)
        value=value.view(batch_size,seq_len,-1)

        if is_residual is True:
            value=value+seq_v_value

        if is_layer_norm is True:
            value=self.layer_norm(value)

        return value


class Data_for_SSAP(DataSet):

    def __init__(self,data_name="ml-1m",seq_len=10):
        super(Data_for_SSAP, self).__init__(data_name=data_name,min_user_number=5)
        self.data_name=data_name
        self.seq_len=seq_len


    def get_data_for_model(self):
        data=self.get_data()

        return data


    def generate_negative(self,data):
        negative=[]
        for x in data:
            neg=np.random.choice(self.item_number)
            while neg==x:
                neg=np.random.choice(self.item_number)
            negative+=[neg]


        return negative


    def get_data(self):
        data_values=self.data.values

        user_item={}
        for value in data_values:
            user_id=value[0]
            item_id=value[1]

            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        train_data,validation_data,test_data=[],[],[]

        for user,value in user_item.items():
            test=value[-self.seq_len-1:]+[user]
            test_data.append(test)

            valid=value[-self.seq_len-2:-1]+[user]
            validation_data.append(valid)

            neg=value[-self.seq_len-3:-3]
            train=value[-self.seq_len-3:-2]+[user]

            neg=self.generate_negative(neg)
            train+=neg
            train_data.append(train)

        train_data=self.pad_sequences(train_data,max_length=self.seq_len+2+self.seq_len)
        validation_data=self.pad_sequences(validation_data,max_length=self.seq_len+2)
        test_data=self.pad_sequences(test_data,max_length=self.seq_len+2)

        train_data,validation_data,test_data=np.array(train_data),np.array(validation_data),np.array(test_data)

        return train_data,validation_data,test_data


class MultiHeadAttention_plus(nn.Module):

    def __init__(self, n_head=1, d_model=32, d_k=32, d_v=32):
        super(MultiHeadAttention_plus, self).__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_head, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_head, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_head, bias=False)
        self.fc = nn.Linear(self.n_head * self.d_v, self.d_model, bias=False)
        self.layernorm = nn.LayerNorm(self.d_model)

    def scale_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

    def forward(self, input_Q, input_K, input_V, mask=None):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        context, attn = self.scale_dot_product_attention(Q, K, V, mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)
        output = self.fc(context)
        return self.layernorm(output + residual)


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,data):
        train_data_value = data
        total_loss=0
        count=1
        for train_data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (seq_item,user_id,pos_item,neg_item)
            """
            self.optimizer.zero_grad()
            seq_item = train_data[:, :self.model.seq_len]
            pos_item=train_data[:,1:self.model.seq_len+1]
            user_id = train_data[:, self.model.seq_len+1]
            neg_item=train_data[:,-self.model.seq_len:]
            seq_item,pos_item,user_id,neg_item=torch.LongTensor(seq_item),torch.LongTensor(pos_item),torch.LongTensor(user_id),torch.LongTensor(neg_item)
            loss = self.model.calculate_loss(data=[seq_item,user_id,pos_item,neg_item])
            if count%2==0:
                print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.model.logging()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,validation_data,test_data=self.model.dataset.get_data_for_model()
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                validation=validation_data
                label = validation[:, self.model.seq_len]
                validation=torch.LongTensor(validation)
                seq_item = validation[:, :self.model.seq_len]
                user_id=validation[:,self.model.seq_len+1]
                scores=self.model.prediction([seq_item,user_id])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20,50])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=SSAP()
trainer=trainer(model)
trainer.train()