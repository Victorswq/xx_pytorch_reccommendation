from mertic import *
from dataset.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from abstract_model import abstract_model
from Dataset import DataSet


class STAMP(abstract_model):

    def __init__(self,
                 data_name="ml-1m",
                 model_name="STAMP",
                 embedding_size=32,
                 episodes=10,
                 batch_size=512,
                 learning_rate=0.001,
                 seq_len=7,
                 verbose=1,
                 ):
        super(STAMP, self).__init__(data_name=data_name,model_name=model_name)

        self.embedding_size=embedding_size
        self.episodes=episodes
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.seq_len=seq_len
        self.verbose=verbose

        self.criterion=nn.CrossEntropyLoss()

        self.dataset=Data_for_STAMP(data_name=self.data_name,seq_len=self.seq_len)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        self.build_variable()


    def build_variable(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.attention=Attention(embedding_size=self.embedding_size)
        self.W1=nn.Linear(self.embedding_size,self.embedding_size)
        self.W2=nn.Linear(self.embedding_size,self.embedding_size)


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def forward(self,data):
        seq_item=data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        last_memory=seq_item_embedding[:,-1,:]
        # batch_size * embedding_size
        average_memory=torch.div(seq_item_embedding.sum(dim=1),self.seq_len)
        # batch_size * embedding_size

        mask=seq_item.data.eq(0)
        # batch_size * seq_len

        attention=self.attention.forward(seq_item_embedding,last_memory,average_memory,mask)
        hs=self.W1(attention)

        ht=self.W2(last_memory)

        return torch.mul(hs,ht)


    def calculate_loss(self,data):
        seq_item,target=data
        prediction=self.forward(seq_item)
        prediction=torch.matmul(prediction,self.get_item_matrix_weight_transpose())

        loss=self.criterion(prediction,target)

        return loss


    def prediction(self,data):
        seq_item=data
        prediction = self.forward(seq_item)
        prediction = torch.matmul(prediction, self.get_item_matrix_weight_transpose())

        return prediction


class Attention(nn.Module):

    def __init__(self,embedding_size=32):
        super(Attention, self).__init__()
        self.embedding_size=embedding_size
        self.W1=nn.Linear(self.embedding_size,self.embedding_size)
        self.W2=nn.Linear(self.embedding_size,self.embedding_size)
        self.W3=nn.Linear(self.embedding_size,self.embedding_size)
        self.sigmoid=nn.Sigmoid()
        self.W4=nn.Linear(self.embedding_size,1,bias=False)


    def forward(self,all_memory,last_memory,average_memory,mask=None):
        """

        :param all_memory: batch_size * seq_len * embedding_size
            == W1 ==>> batch_size * seq_len * embedding_size
        :param last_memory: batch_size * embedding_size
            == W2 ==>> batch_size * embedding_size
            == unsqueeze & repeat ==>> batch_size * seq_len * embedding_size
        :param average_memory: batch_size * embedding_size
            == W3 ==>> batch_size * embedding_size
            == unsqueeze & repeat ==>> batch_size * seq_len * embedding_size
        mask: batch_size * seq_len
            == unsqueeze & repeat ==>> batch_size * seq_len * embedding_size
        :return:
        """
        batch_size,seq_len,embedding_size=all_memory.size()
        mask=mask.unsqueeze(2).repeat(1,1,embedding_size)
        all_memory_value=all_memory

        all_memory=self.W1(all_memory)

        last_memory=self.W2(last_memory)
        last_memory=last_memory.unsqueeze(1).repeat(1,seq_len,1)

        average_memory=self.W3(average_memory).unsqueeze(1).repeat(1,seq_len,1)

        output=all_memory+last_memory+average_memory
        if mask is None:
            output.masked_fill_(mask,-1e9)

        output=self.sigmoid(output)
        output=self.W4(output).repeat(1,1,embedding_size)
        # batch_size * seq_len * embedding_size

        output=torch.mul(output,all_memory_value).sum(dim=1)
        # batch_size * embedding_size

        return output


class Data_for_STAMP(DataSet):

    def __init__(self,data_name="ml-1m",seq_len=10):
        super(Data_for_STAMP, self).__init__(data_name=data_name,min_user_number=3)
        self.data_name=data_name
        self.seq_len=seq_len


    def get_data_for_model(self):
        data=self.get_data()

        return data


    def get_data(self):
        data_value=self.data.values
        user_item={}

        for value in data_value:
            user_id=value[0]
            item_id=value[1]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        train_data,valid_data,test_data=[],[],[]
        for user_id,item_list in user_item.items():
            length=len(item_list)
            if length<3:
                continue

            test=item_list[-self.seq_len-1:]
            test_data.append(test)

            valid=item_list[-self.seq_len-2:-1]
            valid_data.append(valid)

            for i in range(length-3):
                train=item_list[:i+2]
                train_data.append(train)

        train_data=self.pad_sequences(train_data,max_length=self.seq_len+1)
        valid_data=self.pad_sequences(valid_data,max_length=self.seq_len+1)
        test_data=self.pad_sequences(test_data,max_length=self.seq_len+1)

        train_data,valid_data,test_data=np.array(train_data),np.array(valid_data),np.array(test_data)

        return [train_data,valid_data,test_data]


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
            target=train_data[:,self.model.seq_len]
            seq_item,target=torch.LongTensor(seq_item),torch.LongTensor(target)
            loss = self.model.calculate_loss(data=[seq_item,target])
            if count%500==0:
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
                validation=validation_data[:500,:]
                label = validation[:, self.model.seq_len]
                validation=torch.LongTensor(validation)
                seq_item = validation[:, :self.model.seq_len]
                scores=self.model.prediction(seq_item)
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20,50])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=STAMP(data_name="diginetica")
trainer=trainer(model=model)
trainer.train()