from dataset.utils import *
from mertic import *
import torch
import torch.nn as nn
import torch.optim as optim
from abstract_model import abstract_model
from Dataset import DataSet


class FPMC(abstract_model):

    def __init__(self,
                 embedding_size=32,
                 data_name="ml-1m",
                 model_name="FPMC",
                 learning_rate=0.001,
                 dropout_rate=0.3,
                 batch_size=512,
                 verbose=1,
                 episodes=50,
                 ):
        super(FPMC, self).__init__(data_name=data_name,model_name=model_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.dropout_rate=dropout_rate
        self.batch_size=batch_size
        self.verbose=verbose
        self.episodes=episodes

        self.criterion=BPR_loss()
        self.dataset=Data_for_FPMC(data_name=self.data_name)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        self.build_variables()


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.w1=nn.Linear(2*self.embedding_size,self.embedding_size)
        self.relu=nn.ReLU()
        self.dropout1=nn.Dropout(self.dropout_rate)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size)


    def forward(self,data):
        user,item,last_item=data
        user_embedding=self.user_matrix(user)
        item_embedding=self.item_matrix(item)
        last_item_embedding=self.item_matrix(last_item)

        user_embedding=torch.cat([user_embedding,last_item_embedding],dim=1)
        # batch_size * 2_mul_embedding_size
        user_embedding=self.relu(self.w1(user_embedding))
        user_embedding=self.dropout1(user_embedding)
        user_embedding=self.w2(user_embedding)

        return item_embedding,user_embedding


    def calculate_loss(self,data):
        user,pos_item,last_item,neg_item=data

        user_embedding,pos_item_embedding=self.forward([user,pos_item,last_item])
        pos_scores = torch.mul(user_embedding, pos_item_embedding).sum(dim=1)
        # batch_size
        user_embedding,neg_item_embedding=self.forward([user,neg_item,last_item])
        neg_scores = torch.mul(user_embedding, neg_item_embedding).sum(dim=1)

        loss=self.criterion.forward(pos_scores,neg_scores)

        return loss


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def prediction(self,data):
        user,last_item=data
        user_embedding,item_embedding=self.forward([user,last_item,last_item])
        prediction=torch.matmul(user_embedding,self.get_item_matrix_weight_transpose())

        return prediction


class Data_for_FPMC(DataSet):

    def __init__(self,data_name="ml-1m",ratio=[0.8,0.1,0.1],shuffle=False):
        super(Data_for_FPMC, self).__init__(data_name=data_name)
        self.data_name=data_name
        self.ratio=ratio
        self.shuffle=shuffle


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

        data=[]
        for user_id,item_list in user_item.items():
            length=len(item_list)
            for i in range(1,length):
                datas = []
                target=item_list[i]
                datas+=item_list[i-1:i+1]
                datas+=[user_id]
                neg_id=np.random.choice(self.item_number)
                while neg_id==0 or neg_id==target:
                    neg_id=np.random.choice(self.item_number)
                datas+=[neg_id]
                data.append(datas)

        data=np.array(data,dtype=np.int)
        data=self.split_by_ratio(data,ratio=self.ratio,shuffle=self.shuffle)

        return data


class BPR_loss(nn.Module):

    def __init__(self,gamma=1e-9):
        super(BPR_loss, self).__init__()
        self.gamma=gamma


    def forward(self,pos_scores,neg_scores):
        loss=-torch.log(self.gamma+torch.sigmoid(pos_scores-neg_scores)).mean()

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
            train_data: (user,pos_item,last_item,neg_item)
            """
            self.optimizer.zero_grad()
            last_item = train_data[:, 0]
            pos_item=train_data[:,1]
            user_id = train_data[:, 2]
            neg_item=train_data[:,3]
            last_item,pos_item,user_id,neg_item=torch.LongTensor(last_item),torch.LongTensor(pos_item),torch.LongTensor(user_id),torch.LongTensor(neg_item)
            loss = self.model.calculate_loss(data=[user_id,pos_item,last_item,neg_item])
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
                x=np.random.choice(len(validation_data),size=5120)
                validation=validation_data[x,:]
                label = validation[:, 1]
                validation=torch.LongTensor(validation)
                last_item = validation[:, 0]
                user_id=validation[:,2]
                scores=self.model.prediction([user_id,last_item])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[20,50])
                # MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=FPMC()
trainer=trainer(model=model)
trainer.train()