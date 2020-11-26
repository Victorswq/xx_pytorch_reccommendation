from mertic import *
from dataset.utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from Dataset import DataSet
from abstract_model import abstract_model


class NARM(abstract_model):

    def __init__(self,
                 data_name="diginetica",
                 model_name="NARM",
                 embedding_size=32,
                 learning_rate=0.001,
                 hidden_size=32,
                 episodes=6,
                 verbose=1,
                 seq_len=6,
                 batch_size=512,
                 ):
        super(NARM, self).__init__(data_name=data_name,model_name=model_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.hidden_size=hidden_size
        self.episodes=episodes
        self.verbose=verbose
        self.seq_len=seq_len
        self.batch_size=batch_size

        self.dataset=Data_for_NARM(seq_len=self.seq_len,data_name=self.data_name)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        self.criterion=nn.CrossEntropyLoss()

        self.build_variables()


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.gru=nn.GRU(input_size=self.embedding_size,hidden_size=self.hidden_size,batch_first=True)
        self.attention=Attention(embedding_size=self.embedding_size)
        self.matrix_b=nn.Linear(2*self.embedding_size,self.embedding_size,bias=False)


    def get_item_matrix_transpose(self):

        return self.item_matrix.weight.t()


    def forward(self,data):
        """

        :param data: seq_item
            seq_item: batch_size * seq_len
        :return:
        """
        seq_item=data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size

        # get the pad mask
        mask=seq_item.data.eq(0)

        all_memory,last_memory=self.gru(seq_item_embedding)
        # all_memory: batch_size * seq_len * embedding_size
        # last_memory: 1 * batch_size * embedding_size
        cg=last_memory.squeeze(0)

        cl=self.attention.forward(all_memory=all_memory,last_memory=last_memory,mask=mask)

        clg=torch.cat([cg,cl],dim=1)

        return clg

    def calculate_loss(self,data):
        seq_item,label=data
        clg=self.forward(seq_item)
        prediction=self.matrix_b(clg)
        # batch_size * embedding_size
        prediction=torch.matmul(prediction,self.get_item_matrix_transpose())
        # batch_size * num_item
        loss=self.criterion(prediction,label)

        return loss


    def prediction(self,data):
        seq_item=data
        clg=self.forward(seq_item)
        prediction=self.matrix_b(clg)
        prediction=torch.matmul(prediction,self.get_item_matrix_transpose())

        return prediction


class Attention(nn.Module):

    def __init__(self,embedding_size=32):
        super(Attention, self).__init__()
        self.embedding_size=embedding_size
        self.W1=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.W2=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.sigmoid=nn.Sigmoid()
        self.W3=nn.Linear(self.embedding_size,1)


    def forward(self,all_memory,last_memory,mask=None):
        """

        :param all_memory: batch_size * seq_len * embedding_size
            == W1 ==>> batch_size * seq_len * embedding_size
        :param last_memory: 1 * batch_size * embedding_size
            == transpose(0,1) ==>> batch_size * 1 * embedding_size
            == W2 ==>> batch_size * 1 * embedding_size
            == repeat ==>> batch_size * seq_len * embedding_size
            == W3 & squeeze dim 2 ==>> batch_size * seq_len
            == softmax ==>> batch_size * seq_len
            == unsqueeze dim 2 and repeat ==>> batch_size * seq_len * embedding_size
        ***
        attn_output=(output * all_memory_value).sum(dim=1)
            ==>> batch_size * embedding_size
        :param mask: batch_size * seq_len
        :return:
        """
        batch_size,seq_len,embedding_size=all_memory.size()
        all_memory_value=all_memory
        last_memory=last_memory.transpose(0,1)

        all_memory=self.W1(all_memory)
        last_memory=self.W2(last_memory).repeat(1,seq_len,1)

        output=self.sigmoid(all_memory+last_memory)
        output=self.W3(output).squeeze(2)

        if mask is not None:
            output.masked_fill_(mask,-1e9)

        output=nn.Softmax(dim=1)(output)
        output=output.unsqueeze(2).repeat(1,1,embedding_size)

        attn_output=torch.mul(output,all_memory_value).sum(dim=1)

        return attn_output


class Data_for_NARM(DataSet):

    def __init__(self,seq_len=7,data_name="diginetica",min_user_number=3):
        super(Data_for_NARM, self).__init__(data_name=data_name,min_user_number=min_user_number)
        self.seq_len=seq_len


    def get_data_for_model(self):
        data=self.get_data()

        return data


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
        for user,item_list in user_item.items():
            length=len(item_list)
            if length<3:
                continue
            test=item_list[-self.seq_len-1:]
            test_data.append(test)

            valid=item_list[-self.seq_len-2:-1]
            validation_data.append(valid)

            for i in range(length-3):
                train=item_list[:i+2]
                train_data.append(train)

        train_data=self.pad_sequences(train_data,max_length=self.seq_len+1)
        validation_data=self.pad_sequences(validation_data,max_length=self.seq_len+1)
        test_data=self.pad_sequences(test_data,max_length=self.seq_len+1)

        train_data,validation_data,test_data=np.array(train_data),np.array(validation_data),np.array(test_data)

        return train_data,validation_data,test_data


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


model=NARM()
trainer=trainer(model=model)
trainer.train()