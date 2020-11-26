from dataset.utils import *
from mertic import *
import torch
import torch.nn as nn
import torch.optim as optim
from abstract_model import abstract_model
from Dataset import DataSet


class NARMU(abstract_model):

    def __init__(self,
                 embbedding_size=32,
                 data_name="ml-1m",
                 model_name="NARMU",
                 episodes=6,
                 hidden_size=32,
                 learning_rate=0.001,
                 verbose=1,
                 batch_size=512,
                 seq_len=10,):
        super(NARMU, self).__init__(data_name=data_name,model_name=model_name)
        self.embedding_size=embbedding_size
        self.episodes=episodes
        self.hidden_size=hidden_size
        self.verbose=verbose
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.seq_len=seq_len

        self.dataset=Data_for_NARMU(data_name=self.data_name)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        self.criterion=nn.CrossEntropyLoss()
        self.build_variables()


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.gru=nn.GRU(input_size=self.embedding_size,hidden_size=self.hidden_size,batch_first=True)
        self.attention_user=Attention(embedding_size=self.embedding_size)
        self.attention=Attention(embedding_size=self.hidden_size)
        self.w1=nn.Linear(self.embedding_size*3,self.embedding_size)


    def forward(self,data):
        seq_item,user=data
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        user_embedding=self.user_matrix(user)
        # batch_size * embedding_size

        all_memory,last_memory=self.gru(seq_item_embedding)
        """
        all_memory: batch_size * seq_len * embedding_size
        last_memory: 1 * batch_size * embedding_size
        """
        last_memory=last_memory.squeeze(0)
        # batch_size * embedding_size

        mask=seq_item.data.eq(0)

        general_thinking=self.attention.forward(seq_item_embedding=all_memory,user_embedding=last_memory,mask=mask)
        # batch_size * embedding_size

        user_preference=self.attention_user.forward(seq_item_embedding=seq_item_embedding,user_embedding=user_embedding,mask=mask)

        whole_embedding=torch.cat([general_thinking,user_preference,last_memory],dim=1)
        # batch_size * 3_mul_embedding_size
        whole_embedding=self.w1(whole_embedding)
        # batch_size * embedding_size

        return whole_embedding


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def calculate_loss(self,data):
        user,seq_item,label=data
        prediction=self.forward([seq_item,user])
        # batch_size * embedding_size
        prediction=torch.matmul(prediction,self.get_item_matrix_weight_transpose())
        # batch_size * item_number
        loss=self.criterion(prediction,label)

        return loss


    def prediction(self,data):
        user,seq_item=data
        prediction=self.forward([seq_item,user])
        prediction=torch.matmul(prediction,self.get_item_matrix_weight_transpose())

        return prediction


class Attention(nn.Module):

    def __init__(self,embedding_size=32):
        super(Attention, self).__init__()
        self.embedding_size=embedding_size
        self.w1=nn.Linear(self.embedding_size,self.embedding_size)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)
        self.tanh=nn.ReLU()
        self.w3=nn.Linear(self.embedding_size,1)

    def forward(self,seq_item_embedding,user_embedding,mask=None):
        """

        :param seq_item_embedding: batch_size * seq_len * embedding_size
            == w1 == >> batch_size * embedding_size
        :param user_embedding: batch_size * embedding_size
            == w2 == >> batch_size * embedding_size
            == unsqueeze & repeat == >> batch_size * seq_len * embedding_size
        ***
        output=seq_item_embedding+user_embedding: batch_size * seq_len * embedding_size
            == w3 == >> batch_size * seq_len * 1 == squeeze dim 2 ==>> batch_size * seq_len
            == unsqueeze dim=2 & repeat(1,1,embedding_size) == >> batch_size * seq_len * embedding_size
        :param mask: batch_size * seq_len
            == unsqueeze & repeat == >> batch_size * seq_len * embedding_size
        :return:
        """
        batch_size,seq_len,embedding_size=seq_item_embedding.size()
        seq_item_embedding_value=seq_item_embedding

        seq_item_embedding=self.w1(seq_item_embedding)
        user_embedding=self.w2(user_embedding).unsqueeze(1).repeat(1,seq_len,1)

        output=self.tanh(seq_item_embedding+user_embedding)
        output=self.w3(output).squeeze(2)

        if mask is not None:
            output.masked_fill_(mask,-1e9)
            # batch_size * seq_len * embedding_size

        output=output.unsqueeze(2).repeat(1,1,embedding_size)

        output=torch.mul(output,seq_item_embedding_value).sum(dim=1)
        # batch_size * embedding_size

        return output


class Data_for_NARMU(DataSet):

    def __init__(self,seq_len=10,data_name="ml-1m"):
        super(Data_for_NARMU, self).__init__(data_name=data_name,min_user_number=3)
        self.seq_len=seq_len
        self.data_name=data_name


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
        for user_id,item_list in user_item.items():
            length=len(item_list)
            if length<3:
                continue
            test=item_list[-self.seq_len-1:-1]+[user_id]
            test_data.append(test)
            valid=item_list[-self.seq_len-2:-2]+[user_id]
            validation_data.append(valid)
            for i in range(2,length-2):
                train=item_list[:i]+[user_id]
                train_data.append(train)
        train_data=self.pad_sequences(train_data,max_length=self.seq_len+2)
        validation_data=self.pad_sequences(validation_data,max_length=self.seq_len+2)
        test_data=self.pad_sequences(test_data,max_length=self.seq_len+2)
        train_data=np.array(train_data)
        validation_data=np.array(validation_data)
        test_data=np.array(test_data)

        return [train_data,validation_data,test_data]


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
            user=train_data[:,self.model.seq_len+1]
            seq_item,target,user=torch.LongTensor(seq_item),torch.LongTensor(target),torch.LongTensor(user)
            loss = self.model.calculate_loss(data=[user,seq_item,target])
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
                user = validation[:, self.model.seq_len + 1]
                seq_item = validation[:, :self.model.seq_len]
                scores=self.model.prediction([user,seq_item])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20,50])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=NARMU(data_name="diginetica")
trainer=trainer(model)
trainer.train()