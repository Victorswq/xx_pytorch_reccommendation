from mertic import *
from dataset.utils import *
from torch.nn import functional as F
from utils import *
import torch
import torch.nn as nn
from abstract_model import abstract_model
from Dataset import DataSet
import torch.optim as optim


class RepeatNet(abstract_model):
    """

    """
    def __init__(self,
                 data_name="diginetica",
                 model_name="RepeatNet",
                 embedding_size=32,
                 hidden_size=32,
                 learning_rate=0.001,
                 seq_len=10,
                 min_user_number=5,
                 min_item_number=5,
                 episodes=50,
                 verbose=1,
                 batch_size=512,
                 ):
        super(RepeatNet, self).__init__(data_name=data_name,model_name=model_name,min_user_number=min_user_number,min_item_number=min_item_number)
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.learning_rate=learning_rate
        self.seq_len=seq_len
        self.episodes=episodes
        self.verbose=verbose
        self.batch_size=batch_size

        self.dataset=Data_for_RepeatNet(data_name=self.data_name,seq_len=self.seq_len,min_user_number=self.min_user_number,min_item_number=self.min_item_number)

        self.user_number = self.dataset.user_number + 1
        self.item_number = self.dataset.item_number + 1

        self.build_variable()


    def build_variable(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.gru=nn.GRU(self.embedding_size,self.hidden_size,batch_first=True)
        self.repeat_explore_mechanism=Repeat_Explore_Mechanism(hidden_size=self.hidden_size,seq_len=self.seq_len)
        self.repeat_recommendation_decoder=Repeat_Recommendation_Decoder(hidden_size=self.hidden_size,seq_len=self.seq_len,num_item=self.item_number)
        self.explore_recommendation_decoder=Explore_Recommendation_Decoder(hidden_size=self.hidden_size,seq_len=self.seq_len,num_item=self.item_number)


    def forward(self,seq_item):
        batch_seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len == embedding ==>> batch_size * seq_len * embedding_size

        all_memory,last_memory=self.gru(batch_seq_item_embedding)
        last_memory=last_memory.squeeze(0)
        # all_memory: batch_size * seq_item * hidden_size
        # last_memory: batch_size * hidden_size

        repeat_recommendation_mechanism=self.repeat_explore_mechanism.forward(all_memory=all_memory,last_memory=last_memory)
        # batch_size * 2

        timeline_mask = torch.BoolTensor(seq_item == 0)
        repeat_recommendation_decoder=self.repeat_recommendation_decoder.forward(all_memory=all_memory,last_memory=last_memory,seq_item=seq_item,mask=timeline_mask)
        # batch_size * num_item

        explore_recommendation_decoder=self.explore_recommendation_decoder.forward(all_memory=all_memory,last_memory=last_memory,seq_item=seq_item,mask=timeline_mask)
        # batch_size * num_item

        prediction=repeat_recommendation_decoder*repeat_recommendation_mechanism[:,0].unsqueeze(1)+explore_recommendation_decoder*repeat_recommendation_mechanism[:,1].unsqueeze(1)
        # batch_size * num_item

        return prediction


    def calculate_loss(self,data):
        seq_item,target=data
        prediction = self.forward(seq_item)
        loss = F.nll_loss((prediction + 1e-8).log(), target, ignore_index=0)

        return loss


    def prediction(self,seq_item):
        prediction=self.forward(seq_item)

        return prediction


class Repeat_Explore_Mechanism(nn.Module):

    def __init__(self,hidden_size=32,seq_len=10):
        super(Repeat_Explore_Mechanism, self).__init__()
        self.hidden_size=hidden_size
        self.seq_len=seq_len
        self.Wr=nn.Linear(hidden_size,hidden_size,bias=False)
        self.Ur=nn.Linear(hidden_size,hidden_size,bias=False)
        self.tanh=nn.Tanh()
        self.Vre=nn.Linear(hidden_size,1,bias=False)
        self.Wre=nn.Linear(hidden_size,2,bias=False)


    def forward(self,all_memory,last_memory):
        """

        :param all_memory: batch_size * seq_len * hidden_size
        == Ur ==>> batch_size * seq_len * hidden_size

        :param last_memory: batch_size * hidden_size
        == Wr ==>> batch_size * hidden_size
        == unsqueeze ==>> batch_size * 1 * hidden_size
        == repeat ==>> batch_size * seq_len * hidden_size

        ***
        last_memory + all_memory == tanh == output
        output: batch_size * seq_len * hidden_size

        ***
        output == Vre ==>>
        output: batch_size * seq_len * 1
        == repeat (1,1,hidden_size) ==>> batch_size * seq_len * hidden_size
        == multiply all_memory_values ==>> batch_size * seq_len * hidden_size
        == sum dim=1 ==>> batch_size * hidden_size

        ***
        output == Wre ==>>
        outpur: batch_size * 2
        == softmax ==>> batch_size * 2
        :return:
        """
        all_memory_values=all_memory

        all_memory=self.Ur(all_memory)

        last_memory=self.Wr(last_memory)
        last_memory=last_memory.unsqueeze(1)
        last_memory=last_memory.repeat(1,self.seq_len,1)

        output=self.tanh(all_memory+last_memory)

        output=self.Vre(output)
        output=output.repeat(1,1,self.hidden_size)
        output=output*all_memory_values
        output=output.sum(dim=1)

        output=self.Wre(output)

        output=nn.Softmax(dim=-1)(output)

        return output


class Repeat_Recommendation_Decoder(nn.Module):

    def __init__(self,hidden_size=32,seq_len=10,num_item=40000):
        super(Repeat_Recommendation_Decoder, self).__init__()
        self.hidden_size=hidden_size
        self.seq_len=seq_len
        self.num_item=num_item
        self.Wr=nn.Linear(hidden_size,hidden_size,bias=False)
        self.Ur=nn.Linear(hidden_size,hidden_size,bias=False)
        self.tanh=nn.Tanh()
        self.Vr=nn.Linear(hidden_size,1)


    def forward(self,all_memory,last_memory,seq_item,mask=None):
        """

        :param all_memory: batch_size * seq_len * hidden_size
        == Ur ==>> batch_size * seq_len * hidden_size
        :param last_memory: batch_size * hidden_size
        == Wr ==>> batch_size * hidden_size
        == unsqueeze ==> batch_size * 1 * hidden_size
        == repeat ==> batch_size * seq_len * hidden_size
        ***
        output = nn.tanh( last_memory + all_memory )
        batch_size * seq_len * hidden_size
        ***
        output == Vr ==>> batch_size * seq_len * 1 == squeeze ==> batch_size * seq_len
        == mask the padding item ==>> batch_size * seq_len
        == softmax the last dim ==>> batch_size * seq_len
        ***
        reflect to the shape of: batch_size * num_item
        == unsqueeze the dim 1: batch_size * 1 * seq_len
        map: batch_size * seq_len * num_item
        ***
        output matmul map: batch_size * 1 * num_item
        == squeeze dim 1 == batch_size * num_item

        :return:
        """
        all_memory=self.Ur(all_memory)

        last_memory=self.Wr(last_memory)
        last_memory=last_memory.unsqueeze(1)
        last_memory=last_memory.repeat(1,self.seq_len,1)

        output=self.tanh(last_memory+all_memory)

        output=self.Vr(output).squeeze(2)

        if mask is not None:
            output.masked_fill_(mask,-1e9)

        output=nn.Softmax(dim=-1)(output)
        output=output.unsqueeze(1)

        map = build_map(seq_item, max=self.num_item)
        output = torch.matmul(output, map).squeeze(1)
        output=output.squeeze(1)

        return output


class Explore_Recommendation_Decoder(nn.Module):

    def __init__(self,hidden_size,seq_len,num_item):
        super(Explore_Recommendation_Decoder, self).__init__()
        self.hidden_size=hidden_size
        self.seq_len=seq_len
        self.num_item=num_item
        self.We=nn.Linear(hidden_size,hidden_size)
        self.Ue=nn.Linear(hidden_size,hidden_size)
        self.tanh=nn.Tanh()
        self.Ve=nn.Linear(hidden_size,1)
        self.matrix_for_explore=nn.Linear(2*self.hidden_size,self.num_item,bias=False)


    def forward(self,all_memory,last_memory,seq_item,mask=None):
        """

        :param all_memory: batch_size * seq_len * hidden_size
        == Ue ==>> batch_size * seq_len * hidden_size
        :param last_memory: batch_size * hidden_size
        == We ==>> batch_size * hidden_size
        == unsqueeze ==>> batch_size * 1 * hidden_size
        == repeat  ==>> batch_size * seq_len * hidden_size
        ***
        output = nn.tanh( last_memory + all_memory )
        == Ve ==>> batch_size * seq_len * 1
        == repeat ==>> batch_size * seq_len * hidden_size
        ***
        output = all_memory_values * output : batch_size * seq_len * hidden_size
        == sum dim=1 ==>> batch_size * hidden_size
        ***
        concat output and last_memory:== >> batch_size * 2_mul_hidden_size
        ***
        == matrix for explore ==>> batch_size * num_item
        :return:
        """
        all_memory_values,last_memory_values=all_memory,last_memory

        all_memory=self.Ue(all_memory)

        last_memory=self.We(last_memory)
        last_memory=last_memory.unsqueeze(1)
        last_memory=last_memory.repeat(1,self.seq_len,1)

        output=self.tanh(all_memory+last_memory)
        output=self.Ve(output).squeeze(-1)

        if mask is not None:
            output.masked_fill_(mask,-1e9)

        output=output.unsqueeze(-1)

        output=nn.Softmax(dim=1)(output)
        output=output.repeat(1,1,self.hidden_size)
        output=(output*all_memory_values).sum(dim=1)
        output=torch.cat([output,last_memory_values],dim=1)
        output=self.matrix_for_explore(output)

        map = build_map(seq_item, max=self.num_item)
        explore_mask = torch.bmm((seq_item > 0).float().unsqueeze(1), map).squeeze(1)
        output = output.masked_fill(explore_mask.bool(), float('-inf'))
        output=nn.Softmax(1)(output)

        return output


class Data_for_RepeatNet(DataSet):

    def __init__(self,data_name="diginetica",seq_len=10,min_user_number=5,min_item_number=5):
        super(Data_for_RepeatNet, self).__init__(data_name=data_name,min_user_number=min_user_number,min_item_number=min_item_number)
        self.data_name=data_name
        self.seq_len=seq_len


    def get_data_for_model(self):
        data=self.leave_out_out(data=self.data,max_seq_length=self.seq_len+1)
        # the reason plus 1 is because we put the target at the last place, like this (s1,s2,s3 ... sn target)
        return data


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,data):
        train_datas = data
        total_loss=0
        count=1
        for train_data in get_batch(train_datas,batch_size=self.model.batch_size):
            self.optimizer.zero_grad()
            seq_item = train_data[:, :-1]
            target = train_data[:, -1]
            seq_item,target=torch.LongTensor(seq_item),torch.LongTensor(target)
            loss = self.model.calculate_loss(data=[seq_item,target])
            if count%1==0:
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
                label = validation[:500, -1]
                validation=torch.LongTensor(validation)
                validation=validation[:500,:-1]
                scores=self.model.prediction(validation)
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=RepeatNet(min_user_number=3)
trainer=trainer(model=model)
trainer.train()