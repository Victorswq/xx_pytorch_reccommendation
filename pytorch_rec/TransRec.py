from dataset.utils import *
from mertic import *
import numpy as np
import torch
import torch.nn as nn
from Dataset import DataSet
from abstract_model import abstract_model
import torch.optim as optim

class TBRec(abstract_model):

    def __init__(self,
                 data_name='ml-1m',
                 model_name="TBRec",
                 embedding_size=32,
                 learning_rate=0.001,
                 batch_size=512,
                 episodes=20,
                 verbose=1,
                 ):
        super(TBRec, self).__init__(data_name=data_name,model_name=model_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.episodes=episodes
        self.verbose=verbose

        self.loss=BPRLoss()
        self.dataset=Dataset_for_TBRec(data_name=self.data_name)

        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        self.build_variable()


    def build_variable(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def forward(self,last_item,user):
        item_embedding=self.item_matrix(last_item)
        # batch_size * embedding_size
        user_embedding=self.user_matrix(user)
        # batch_size * embedding_size
        item_user=item_embedding+user_embedding
        # batch_size * embedding_size

        return item_user


    def calculate_loss(self,data):
        last_item,user,pos_item,neg_item=data
        pos_item_embedding=self.item_matrix(pos_item)
        # batch_size * embedding_size
        neg_item_embedding=self.item_matrix(neg_item)
        # batch_size * embedding_size

        item_user=self.forward(last_item,user)

        pos_score=torch.mul(item_user,pos_item_embedding).sum(dim=1)
        # batch_size
        neg_score=torch.mul(item_user,neg_item_embedding).sum(dim=1)
        # batch_size

        loss=self.loss.forward(pos_score,neg_score)

        return loss


    def prediction(self,data):
        last_item,user=data
        prediction=self.forward(last_item,user)
        # batch_size * embedding_size
        prediction=torch.matmul(prediction,self.get_item_matrix_weight_transpose())

        return prediction


class Dataset_for_TBRec(DataSet):

    def __init__(self,data_name="ml-1m"):
        super(Dataset_for_TBRec, self).__init__(data_name=data_name,item_id_index=2,user_id_index=1,time_id_index=4,sep="\t",min_user_number=10,min_item_number=10)
        self.data_name=data_name

    def get_data_for_model(self):
        data=self.data_for_TBRec()

        return data


    def data_for_TBRec(self):
        """
        (last item, pos_item, user_id, neg_item)
        :return:
        """
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
        for key,value in user_item.items():
            lengths=len(value)
            if lengths<3:
                continue
            test=value[-2:]
            test+=[key]
            test_data.append(test)

            valid=value[-3:-1]
            valid+=[key]
            validation_data.append(valid)

            length=len(value)
            for i in range(length-3):
                train=value[i:i+2]
                target=train[-1]
                train+=[key]
                neg=np.random.choice(self.item_number)
                while neg==target or neg==0:
                    neg=np.random.choice(self.item_number)
                train+=[neg]
                train_data.append(train)
        train_data,validation_data,test_data=np.array(train_data),np.array(validation_data),np.array(test_data)

        return [train_data,validation_data,test_data]


class BPRLoss(nn.Module):

    def __init__(self,gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma=gamma


    def forward(self,pos_score,neg_score):
        loss=-torch.log(self.gamma+torch.sigmoid(pos_score-neg_score)).mean()

        return loss


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,data):
        train_data_value = data
        total_loss=0
        count=1
        for train_data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (last_item, pos_item, user_id, neg_item)
            """
            self.optimizer.zero_grad()
            last_item = train_data[:, 0]
            pos_item=train_data[:,1]
            user_id = train_data[:, 2]
            neg_item=train_data[:,3]
            last_item,pos_item,user_id,neg_item=torch.LongTensor(last_item),torch.LongTensor(pos_item),torch.LongTensor(user_id),torch.LongTensor(neg_item)
            loss = self.model.calculate_loss(data=[last_item,user_id,pos_item,neg_item])
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
                validation=validation_data
                label = validation[:, 1]
                validation=torch.LongTensor(validation)
                last_item = validation[:, 0]
                user_id=validation[:,2]
                scores=self.model.prediction([last_item,user_id])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20,50])
                # MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=TBRec(data_name="ml-100k")
trainer=trainer(model)
trainer.train()