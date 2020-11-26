from dataset.utils import *
import pandas as pd
from mertic import *
import torch
import torch.nn as nn
from torch.nn import functional as F
from abstract_model import abstract_model
import torch.optim as optim


class Caser(abstract_model):
    def __init__(self,
                 data_name="ml-1m",
                 model_name="Caser",
                 embedding_size=32,
                 learning_rate=0.001,
                 L=4,
                 T=2,
                 batch_size=5120,
                 seq_length=10,
                 episodes=50,
                 verbose=1,
                 ):
        super(Caser, self).__init__(data_name,model_name)
        self.data_name=data_name
        self.model_name=model_name
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.L=L
        self.T=T
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.episodes=episodes
        self.verbose=verbose

        self.relu=nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()
        self.build_variables()


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number+1,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number+1,self.embedding_size)
        self.horizontal_convolutional_layer=Horizontal_Convolutional_Layer(seq_length=self.seq_length,embedding_size=self.embedding_size,L=self.L)
        self.vertical_convolutional_layer=Vertical_Convolutional_Layer(seq_length=self.L,embedding_size=self.embedding_size)
        self.full_connect_layer=Full_Connect_layer(L=self.L,embedding_size=self.embedding_size)
        self.fc_prediction=nn.Linear(self.embedding_size*2,self.item_number+1)


    def forward(self,seq_item,user):
        seq_item_embedding=self.item_matrix(seq_item).unsqueeze(1)
        user_embedding=self.user_matrix(user)
        # batch_size * 1 * seq_len * embedding_size
        horizontal_output=self.horizontal_convolutional_layer.forward(batch_seq_item=seq_item_embedding)
        vertical_output=self.vertical_convolutional_layer.forward(batch_seq_item=seq_item_embedding)
        h_v_embedding=torch.cat([horizontal_output,vertical_output],dim=1)
        connect_user_item=self.full_connect_layer.forward(user_embedding=user_embedding,h_v_embedding=h_v_embedding)
        prediction=self.relu(self.fc_prediction(connect_user_item))

        return prediction


    def calculate_loss(self,data):
        """

        :param data:
        data: user seq_item target
        user: batch_size
        seq_item: batch_size * seq_len
        target: batch_size * T
        :return:
        """
        user,seq_item,target=data
        prediction=self.forward(seq_item,user)
        batch_size,num_item=prediction.size()
        prediction=prediction.unsqueeze(1).repeat(1,2,1).view(2*batch_size,-1)
        target=target.view(-1)
        loss=self.criterion(prediction,target)

        return loss


    def prediction(self,data):
        seq_item,user=data
        inference=self.forward(seq_item,user)

        return inference


class Horizontal_Convolutional_Layer(nn.Module):

    def __init__(self,seq_length=10,embedding_size=32,L=4):
        super(Horizontal_Convolutional_Layer, self).__init__()
        self.seq_length=seq_length
        self.embedding_size=embedding_size
        self.L=L
        self.relu=nn.ReLU()
        self.conv_h=nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(i,self.embedding_size)) for i in range(1,self.L+1)])

    def forward(self,batch_seq_item):
        """

        :param batch_seq_item: batch_size * 1 * seq_len * embedding_size
        == conv2d ==>> batch_size * 1 * (seq_len-i+1)
        == max_pool ==>> batch_size * 1
        == concat ==>> batch_size * L
        :return:
        """
        out_hs = list()
        for conv in self.conv_h:
            conv_out=self.relu(conv(batch_seq_item).squeeze(3))
            pool_out=F.max_pool1d(conv_out,conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out)
        out_h=torch.cat(out_hs,dim=1)
        return out_h


class Vertical_Convolutional_Layer(nn.Module):

    def __init__(self,seq_length=10,embedding_size=32):
        super(Vertical_Convolutional_Layer, self).__init__()
        self.seq_length=seq_length
        self.embedding_size=embedding_size
        self.conv_v=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(self.seq_length,1))


    def forward(self,batch_seq_item):
        """

        :param batch_seq_item: batch_size * 1 * seq_len * embedding_size
        == conv2d ==> batch_size * 1 * 1 * embedding_size
        :return:
        """
        output=self.conv_v(batch_seq_item)
        output=output.view(-1,self.embedding_size)
        return output


class Full_Connect_layer(nn.Module):

    def __init__(self,L=4,embedding_size=32):
        super(Full_Connect_layer, self).__init__()
        self.L=L
        self.embedding_size=embedding_size
        self.relu=nn.ReLU()
        self.fc_1=nn.Linear(self.L+self.embedding_size,self.embedding_size)


    def forward(self,user_embedding,h_v_embedding):
        """

        :param user_embedding: batch_size * embedding_size
        :param h_v_embedding: batch_size * (L+embedding_size)
        == fc ==>> batch_size * embedding_size
        == concat the user_embedding ==>> batch_size * 2_mul_embedding_size
        :return:
        """
        result=self.relu(self.fc_1(h_v_embedding))
        output=torch.cat([result,user_embedding],dim=1)
        return output


class Data_For_Caser():
    """
    product the data for the model Caser

    """
    def __init__(self,data_name="ml-1m",seq_length=10,L=4,T=2):
        self.data_name=data_name
        self.L=L
        self.T=T


    def get_data_for_model(self):
        """
        clear the data:
        min_user_inter=6
        min_item_inter=6
        :return:
        """
        data=pd.read_csv(self.data_name+".inter",sep="\t")
        min_user_inters = 6
        min_item_inters = 6
        for i in range(1):
            data=min_item_inter(data,min_item_inters)
            data=min_user_inter(data,min_user_inters)
        data=unique_id(data)
        train_data, valid_data, test_data=self.leave_T_out(data)

        return train_data,valid_data,test_data


    def pad_sequences(self,data, max_length=50):
        for idx, data_value in enumerate(data):
            new_data = np.zeros(shape=(max_length))
            if len(new_data) > len(data_value):
                new_data[-len(data_value):] = data_value
            else:
                new_data = data_value[-max_length - 1:]
            data[idx] = new_data
        return data


    def leave_T_out(self,data):
        data_values=data.values
        columns = data.columns.values
        sort(data, by=[columns[0], columns[1]], ascending=True)
        user_item={}
        for value in data_values:
            user_id=value[0]
            item_id=value[1]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]
        train_data,valid_data,test_data=[],[],[]
        for key,value in user_item.items():
            test=value[-1-self.L:]+[key]
            test_data.append(test)
            valid=value[-2-self.L:-1]+[key]
            valid_data.append(valid)
            length_of_value=len(value)-2
            for i in range(1,length_of_value-self.T+1):
                if i-self.L>=0:
                    train=value[i-self.L:i+self.T]+[key]
                else:
                    train=value[:i+self.T]+[key]
                train_data.append(train)
        train_data=self.pad_sequences(train_data,self.T+self.L+1)
        test_data=self.pad_sequences(test_data,self.L+1)
        valid_data=self.pad_sequences(valid_data,self.L+1)
        train_data,valid_data,test_data=np.array(train_data),np.array(valid_data),np.array(test_data)

        return train_data,valid_data,test_data


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,data):
        train_datas = data
        total_loss=0
        count=1
        for train_data in get_batch(train_datas,batch_size=self.model.batch_size):
            self.optimizer.zero_grad()
            seq_item = train_data[:, :-self.model.T-1]
            target = train_data[:, -self.model.T-1:-1]
            user = train_data[:, -1]
            seq_item,target,user=torch.LongTensor(seq_item),torch.LongTensor(target),torch.LongTensor(user)
            loss = self.model.calculate_loss(data=[user,seq_item,target])
            if count%20==0:
                print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.model.logging()
        data_for_caser=Data_For_Caser(data_name=self.model.data_name,seq_length=self.model.seq_length,L=self.model.L,T=self.model.T)
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,validation_data,test_data=data_for_caser.get_data_for_model()
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                validation=validation_data
                label = validation[:, -2]
                user=validation[:,-1]
                validation,user=torch.LongTensor(validation),torch.LongTensor(user)
                validation=validation[:,:-2]
                scores=self.model.prediction([validation,user])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                Precision(ratingss=scores.detach().numpy(),pos_items=label,top_k=[1,5,10])


model=Caser()
trainer=trainer(model)
trainer.train()