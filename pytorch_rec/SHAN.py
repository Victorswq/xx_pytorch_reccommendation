from dataset.utils import *
from mertic import *
import torch.optim as optim
from abstract_model import abstract_model
from Dataset import DataSet
import torch
import torch.nn as nn


class SHAN(abstract_model):

    def __init__(self,
                 data_name="Tmall",
                 model_name="SHAN",
                 learning_rate=0.001,
                 embedding_size=32,
                 batch_size=512,
                 episodes=50,
                 verbose=1,
                 max_seq_len=10,
                 min_user_number=3,
                 min_item_number=5):
        super(SHAN, self).__init__(data_name=data_name,model_name=model_name,min_user_number=min_user_number,min_item_number=min_item_number)
        self.data_name=data_name
        self.model_name=model_name
        self.learning_rate=learning_rate
        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.episodes=episodes
        self.verbose=verbose
        self.max_seq_len=max_seq_len
        self.sigmoid=nn.Sigmoid()

        self.dataset=Data_for_SHAN(max_seq_len=self.max_seq_len,data_name=self.data_name,min_user_number=self.min_user_number,min_item_number=self.min_item_number)

        self.user_number=self.dataset.user_number
        self.item_number=self.dataset.item_number

        self.build_variable()

        # build the loss function
        self.criterion=BPRLoss()

        # init the weight of the module parameter
        self.apply(self.init_weights)


    def build_variable(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size)
        self.long_tern_attention_based_pooling_layer=Long_term_Attention_based_Pooling_Layer(embedding_size=self.embedding_size)
        self.long_and_short_term_attention_based_pooling_layer=Long_and_Short_term_Attention_based_Pooling_Layer(embedding_size=self.embedding_size)



    def forward(self,seq_item,user):
        # get the mask first
        mask=torch.BoolTensor(seq_item==0)
        # mask: batch_size * seq_len
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        short_item_embedding=seq_item_embedding[:,-5:,:]
        # batch_size * 1 * embedding_size
        user_embedding=self.user_matrix(user)
        # batch_size * embedding_size

        long_term_attention_based_pooling_layer=self.long_tern_attention_based_pooling_layer.forward(user_embedding,seq_item_embedding,mask)
        # batch_size * embedding_size
        long_term_attention_based_pooling_layer=long_term_attention_based_pooling_layer.unsqueeze(1)
        # batch_size * 1 * embedding_size
        seq_item_embedding=torch.cat([long_term_attention_based_pooling_layer,short_item_embedding],dim=1)
        # batch_size * 2 * embedding_size
        batch_size,short_seq_len,embedding_size=seq_item_embedding.size()
        # get the new mask
        new=torch.ones((batch_size,1))
        mask=torch.cat([new,mask],dim=1)
        # batch_size * (1+seq_len)
        long_and_short_term_attention_based_pooling_layer=self.long_and_short_term_attention_based_pooling_layer.forward(seq_item_embedding,user_embedding)
        # batch_size * embedding_size

        return long_and_short_term_attention_based_pooling_layer


    def get_item_matrix_transpose(self):
        return self.item_matrix.weight.t()


    def calculate_loss(self,data):
        seq_item,user,pos_item,neg_item=data
        # seq_item: batch_size * seq_len
        # user: batch_size
        # pos_item: batch_size
        # neg_item: batch_size
        prediction_matrix=self.forward(seq_item=seq_item,user=user)
        # batch_size * embedding_size

        pos_item_embedding=self.item_matrix(pos_item)
        # batch_size * embedding_size
        neg_item_embedding=self.item_matrix(neg_item)
        # batch_size * embedding_size

        pos_scores=(prediction_matrix*pos_item_embedding).sum(1)
        # batch_size

        neg_scores=(prediction_matrix*neg_item_embedding).sum(1)
        # batch_size

        loss=self.criterion.forward(pos_scores,neg_scores)

        return loss


    def prediction(self,data):
        seq_item,user=data
        prediction=self.forward(seq_item,user)
        prediction=torch.matmul(prediction,self.get_item_matrix_transpose())

        return prediction


class BPRLoss(nn.Module):

    def __init__(self,gamma=1e-9):
        super(BPRLoss, self).__init__()
        self.gamma=gamma


    def forward(self,pos_scores,neg_scores):

        loss=-torch.log(self.gamma+torch.sigmoid(pos_scores-neg_scores)).mean()

        return loss


class Long_term_Attention_based_Pooling_Layer(nn.Module):

    def __init__(self,embedding_size=32):
        super(Long_term_Attention_based_Pooling_Layer, self).__init__()
        self.embedding_size=embedding_size
        self.W1=nn.Linear(self.embedding_size,self.embedding_size)
        self.relu=nn.ReLU()


    def forward(self,user_embedding,seq_item_embedding,mask=None):
        """

        :param user_embedding: batch_size * embedding_size
        == unsqueeze dim 1 ==>> batch_size * embedding_size * 1
        :param seq_item_embedding: batch_size * seq_len * embedding_size
        == W1 ==>> batch_size * seq_len * embedding_size
        == relu ==>> batch_size * seq_len * embedding_size
        ***
        item_seq_embedding: batch_size * seq_len * embedding_size
        user_embedding: batch_size * embedding_size * 1
        matmul: batch_size * seq_len * 1
        ***
        mask: batch_size * seq_len == batch_size * seq_len * 1
        ***
        == softmax the second dim ==>> batch_size * seq_len * 1
        == repeat ==>> batch_size * seq_len * embedding_size
        ***
        seq_item_embedding_value * item_matmul_user
        == >> batch_size * seq_len * embedding_size
        == sum at dim 1 ==>> batch_size * embedding_size
        :return:
        """
        seq_item_embedding_values=seq_item_embedding

        seq_item_embedding=self.W1(seq_item_embedding)
        seq_item_embedding=self.relu(seq_item_embedding)

        user_embedding=user_embedding.unsqueeze(2)
        
        item_matmul_user=torch.matmul(seq_item_embedding,user_embedding).squeeze(2)

        if mask is not None:
            item_matmul_user.masked_fill_(mask,-1e9)

        item_matmul_user=item_matmul_user.unsqueeze(2)
        
        item_matmul_user=nn.Softmax(dim=1)(item_matmul_user)
        
        item_matmul_user=item_matmul_user.repeat(1,1,self.embedding_size)
        
        output=item_matmul_user*seq_item_embedding_values  
        output=output.sum(1)

        return output


class Long_and_Short_term_Attention_based_Pooling_Layer(nn.Module):

    def __init__(self,embedding_size=32):
        super(Long_and_Short_term_Attention_based_Pooling_Layer, self).__init__()
        self.embedding_size=embedding_size
        self.W2=nn.Linear(self.embedding_size,self.embedding_size)
        self.relu=nn.ReLU()


    def forward(self,long_short_seq_embedding,user_embedding,mask=None):
        """
                ::: seq_len = seq_len + 1

        :param long_short_seq_embedding: batch_size * seq_len * embedding_size
        == W2 ==>> batch_size * seq_len * embedding_size
        == relu ==>> batch_size * seq_len * embedding_size
        :param user_embedding: batch_size * embedding_size
        == unsqueeze 2 ==>> batch_size * embedding_size * 1
        *** long_short_seq_embedding matmul user_embedding ==>>
                batch_size * seq_len * 1
        ***
        mask: batch_size * seq_len == >> batch_size * seq_len * 1
            == repeat ==>> batch_size * seq_len * embedding_size
        ***
        item_matmul_user * long_short_seq_embedding_value
        == >> batch_size * seq_len * embedding_size
        == sum dim 1 ==>> batch_size * embedding_size
        :return:
        """
        long_short_seq_embedding_values=long_short_seq_embedding
        long_short_seq_embedding=self.W2(long_short_seq_embedding)
        long_short_seq_embedding=self.relu(long_short_seq_embedding)

        user_embedding=user_embedding.unsqueeze(2)

        item_matmul_user=torch.matmul(long_short_seq_embedding,user_embedding).squeeze(2)

        if mask is not None:
            item_matmul_user.masked_fill_(mask, -1e9)

        item_matmul_user = item_matmul_user.unsqueeze(2)
        item_matmul_user=nn.Softmax(dim=1)(item_matmul_user)

        item_matmul_user=item_matmul_user.repeat(1,1,self.embedding_size)


        output=item_matmul_user*long_short_seq_embedding_values
        output=output.sum(1)

        return output


class Data_for_SHAN(DataSet):

    def __init__(self,max_seq_len,data_name="Tmall",min_user_number=50,min_item_number=50):
        super(Data_for_SHAN, self).__init__(data_name=data_name,min_user_number=min_user_number,min_item_number=min_item_number,time_id_index=4,sep="\t")
        self.max_seq_len=max_seq_len
        self.data_name=data_name


    def get_data_for_model(self):
        data = self.the_last_seq_data(data=self.data, max_seq_data=self.max_seq_len + 1)
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
            seq_item = train_data[:, 1:-2]
            user_id=train_data[:,0]
            target = train_data[:, -2]
            neg=train_data[:,-1]
            seq_item,user_id,target,neg=torch.LongTensor(seq_item),torch.LongTensor(user_id),torch.LongTensor(target),torch.LongTensor(neg)
            loss = self.model.calculate_loss(data=[seq_item,user_id,target,neg])
            if count%15==0:
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
                label = validation[:, -2]
                validation=torch.LongTensor(validation)
                user = validation[:, 0]
                validation=validation[:,1:-2]
                scores=self.model.prediction([validation,user])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=SHAN(data_name="ml-1m",learning_rate=0.001,episodes=100,max_seq_len=10)
trainer=trainer(model=model)
trainer.train()