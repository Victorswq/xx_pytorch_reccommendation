import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.optim as optim

from dataset.utils import *


class Model(nn.Module):

    def __init__(self,features_number=32,
                 batch_size=512,
                 learning_rate=0.001,
                 episodes=5,
                 verbose=1,
                 dropout_prob=0.5,
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
        for x in preiction:
            if x<0:
                print(x)

        return preiction


    def calculate_loss(self,data):

        data,label=data
        prediction=self.forward(data).squeeze(-1)
        loss=self.loss.forward(prediction,label)

        return loss


    def prediction_for_test(self,data):
        #做预测并计算正确率

        prediction=self.forward(data)
        prediction=prediction>0.5
        prediction=prediction+1-1

        return prediction


    def prediction(self,data,is_precision=False):
        #做预测并计算正确率

        data,label=data
        prediction=self.forward(data).squeeze(-1)
        pre=None

        if is_precision is True:
        # just for validation
            pre=self.precision(prediction,label)

        return pre


    def precision(self,prediction,label):

        real=0
        for x,y in zip(prediction,label):
            if x>0.5 and y==1 or x<0.5 and y==0:
                real+=1

        print("Precision is : ",real/len(label))
        return real/len(label)


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self,prediction,label):

        loss=torch.square(prediction-label).mean()

        return loss


class data_for_model(nn.Module):

    def __init__(self,data_name="x.csv",ratio=[0.8,0.1,0.1]):
        super(data_for_model, self).__init__()
        self.data_name=data_name
        self.data=None
        self.ratio=ratio


    def read_the_data(self):

        self.data=pd.read_csv("base_train1.csv")
        self.data=self.data.fillna(0)
        self.data=self.data.values
        print(len(self.data))
        self.feature_number=self.data.shape[1]-3
        data=self.split_by_ratio(ratio=self.ratio)

        return data


    def split_by_ratio(self,ratio=[0.8,0.1,0.1]):

        # index=np.arange(len(self.data))
        # index=np.random.shuffle(index)
        # self.data=self.data[index]
        # print(self.data.shape)

        test=pd.read_csv("base_test1.csv").fillna(0).values

        train_split=int(len(self.data)*ratio[0])
        valid_split=int(len(self.data)*(ratio[0]+ratio[1]))
        train_data=self.data[:train_split,:]
        valid_data=self.data[train_split:,:]
        test_data=self.data[-valid_split:]

        valid_data=np.array(valid_data,dtype=np.int)
        new=[]
        new_test=[]
        for x in valid_data:
            if x[-1]==1:
                new.append(x)
            else:
                new_test.append(x)
        new=np.array(new)
        new_test=np.array(new_test)
        print(len(new))
        print(len(valid_data)/len(new))
        return train_data,valid_data,test


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
            seq_item = train_data[:, 2:-1]
            target=train_data[:,-1]
            loss = self.model.calculate_loss(data=[seq_item,target])
            if count%300==0:
                print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,validation_data=self.train_data,self.validation_data
        pre = 0
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)
            if (episode+1)%self.model.verbose==0:
                validation=validation_data
                label = validation[:, -1]
                validation=torch.FloatTensor(validation)
                seq_item = validation[:, 2:-1]
                new_pre=self.model.prediction([seq_item,label],True)
            if new_pre>pre:
                pre=new_pre
                test=test_data
                label = test[:, -3]
                id=test_data[:,1:2]
                test=torch.FloatTensor(test)
                seq_item = test[:, 2:-1]
                preiction=self.model.prediction_for_test(seq_item).detach().numpy()
                preiction=np.hstack([id,preiction])
                preiction=pd.DataFrame(preiction)
                preiction.to_csv("yeah.csv",index=False,header=["id","prediction"])
        print("highest precision is ",pre)


data=data_for_model()
train_data,valid_data,test_data=data.read_the_data()
model=Model(features_number=data.feature_number,batch_size=512,episodes=500,dropout_prob=0.3,learning_rate=0.0005)
trainer=trainer(model=model,train_data=train_data,valid_data=valid_data)
trainer.train()