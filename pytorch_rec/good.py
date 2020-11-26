import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.optim as optim

from dataset.utils import *


class Model(nn.Module):

    def __init__(self,features_number=32,
                 batch_size=512,
                 learning_rate=0.001,
                 episodes=50,
                 verbose=10,
                 dropout_prob=0.2,
                 layers=[64,64,32]):
        super(Model, self).__init__()
        self.features_number=features_number
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.episodes=episodes
        self.verbose=verbose
        self.dropout_prob=dropout_prob
        self.layers=[self.features_number]+layers

        # build the loss function
        self.loss=MSE()

        # build the variables
        self.build_variables()

        # init the weight of the parameter in the module
        self.apply(self.init_weights)


    def build_variables(self):

        self.model=nn.ModuleList()
        for i in range(len(self.layers)-1):
            input_size=self.layers[i]
            output_size=self.layers[i+1]
            x=self.build_layer(input_size,output_size)
            self.model.append(x)

        self.predict_layer=nn.Linear(in_features=self.layers[-1],out_features=1,bias=False)
        self.sigmoid=nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)


    def init_weights(self,module):

        if isinstance(module,nn.Linear):
            xavier_normal_(module.weight.data)


    def build_layer(self,input_size,output_size):

        w1=nn.Linear(input_size,output_size)

        return w1


    def forward(self,data):
        """

        :param data: batch_size * features_number
        :return:
        """
        for layer in self.model:
            data=self.dropout(self.relu(layer(data)))
        preiction=self.sigmoid(self.predict_layer(data))

        return preiction.squeeze(-1)


    def calculate_loss(self,data):

        data,label=data
        prediction=self.forward(data)
        loss=self.loss.forward(prediction,label)

        return loss


    def prediction(self,data,is_precision=False):
        #做预测并计算正确率

        data,label=data
        prediction=self.forward(data)

        if is_precision is True:
        # just for validation
            self.precision(prediction,label)

        return prediction


    def precision(self,prediction,label):

        real=0
        for x,y in zip(prediction,label):
            if x>0.5 and y==1 or x<0.5 and y==0:
                real+=1

        print("Precision is : ",real/len(label))


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self,prediction,label):

        loss=torch.square(prediction-label).mean()

        return loss


class data_for_model(nn.Module):

    def init(self,data_name="x.csv"):
        super(data_for_model, self).init()
        self.data_name=data_name
        self.data=None


    def read_the_data(self):
        pass


    def split_by_ratio(self,ratio=[0.8,0.1,0.1]):

        index=np.arange(len(self.data))
        index=np.random.shuffle(index)
        self.data=self.data[index]

        train_split=len(self.data)*ratio[0]
        valid_split=len(self.data)*(ratio[0]+ratio[1])
        train_data=self.data[:train_split,:]
        valid_data=self.data[train_split:valid_split,:]
        test_data=self.data[-valid_split:]

        return train_data,valid_data,test_data


class trainer():
    def __init__(self,model,train_data,valid_data):
        self.model=model
        self.train_data=train_data
        self.validation_data=valid_data

    def train_epoch(self,data):
        train_data_value = data
        total_loss=0
        count=1
        for train_data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (seq_item,user_id,pos_item,neg_item)
            """
            self.optimizer.zero_grad()
            train_data=torch.FloatTensor(train_data)
            seq_item = train_data[:, :self.model.features_number]
            target=train_data[:,-1]
            loss = self.model.calculate_loss(data=[seq_item,target])
            print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,validation_data=self.train_data,self.validation_data
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                validation=validation_data
                label = validation[:, -1]
                validation=torch.FloatTensor(validation)
                seq_item = validation[:, :self.model.features_number]
                self.model.prediction([seq_item,label],True)


data=np.random.normal(size=(5120,32))
label=np.random.randint(0,2,size=(5120,1))
data=np.hstack([data,label])
train_data=data[:5120-512,:]
valid_data=data[5120-512:,:]
model=Model()
trainer=trainer(model=model,train_data=train_data,valid_data=valid_data)
trainer.train()