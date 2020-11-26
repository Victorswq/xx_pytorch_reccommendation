from dataset.utils import *
from mertic import *
import torch
import torch.nn as nn
import torch.optim as optim
from abstract_model import abstract_model
from Dataset import DataSet


class GRU_DIB_PEB(abstract_model):

    def __init__(self,
                 data_name="ml-1m",
                 model_name="GRU-DIB-PEB",
                 embedding_size=32,
                 learning_rate=0.001,
                 verbose=1,
                 episodes=100,
                 batch_size=512,
                 hidden_size=32,
                 similar_user_len=10,
                 neg_numbers=3,
                 blocks=3,
                 seq_len=10,
                 gamma=1e-9,):
        super(GRU_DIB_PEB, self).__init__(data_name=data_name,model_name=model_name)
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.verbose=verbose
        self.episodes=episodes
        self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.similar_user_len=similar_user_len
        self.neg_numbers=neg_numbers
        self.blocks=blocks
        self.seq_len=seq_len
        self.gamma=gamma

        self.dataset=Data_for_GRU_DIB_PEB(data_name=self.data_name,seq_len=self.seq_len)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        self.criterion=nn.CrossEntropyLoss()

        self.build_variables()

        self.apply(self.init_weights)


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.gru=nn.GRU(input_size=self.embedding_size,hidden_size=self.hidden_size,batch_first=True)
        self.user_matrix=nn.Embedding(self.user_number,self.embedding_size,padding_idx=0)
        self.dib=DIB(embedding_size=self.embedding_size)
        self.peb=PEB(embedding_size=self.embedding_size,hidden_size=self.hidden_size,blocks=self.blocks,neg_numbers=self.neg_numbers)


    def forward(self,seq_item,user,similar_user):
        """

        :param seq_item: batch_size * seq_len
        :param user: batch_size
        :param similar_user: batch_size * seq_len * similar_user_len
        :return:
        """
        # get the mask first
        mask=similar_user.data.eq(0)

        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        user_embedding=self.user_matrix(user)
        # batch_size * embedding_size
        similar_user_embedding=self.user_matrix(similar_user)
        # batch_size * seq_len * similar_user_len * embedding_size
        seq_item_embedding=self.dib.forward(current_user_embedding=user_embedding,similar_user_embedding=similar_user_embedding,current_item_embedding=seq_item_embedding,mask=mask)
        # batch_size * seq_len * embedding_size
        all_memory,last_memory=self.gru(seq_item_embedding)
        last_memory=last_memory.squeeze(0)
        """
        all_memory: batch_size * seq_len * hidden_size
        last_memory: batch_size * hidden_size
        """
        pu=torch.cat([last_memory,user_embedding],dim=1)
        # batch_size * embedding_size_plus_hidden_size

        return pu


    def calculate_loss(self,seq_item,user,similar_user,neg_item,pos_item):
        """

        :param seq_item: batch_size * seq_len
        :param user: batch_size
        :param similar_user: batch_size * seq_len * similar_user_len
        :param neg_item: batch_size * neg_numbers
        :param pos_item: batch_size
        :return:
        """

        pu=self.forward(seq_item,user,similar_user)
        # batch_size * embedding_size
        neg_item_embedding=self.item_matrix(neg_item)
        # batch_size * neg_numbers * embedding_size
        pos_item_embedding=self.item_matrix(pos_item)
        # batch_size * embedding_size
        x,multi_user_item_atten,a=self.peb.forward(pu=pu,pos_item_embedding=pos_item_embedding,neg_item_embedding=neg_item_embedding)
        # a: batch_size * blocks

        prediction=torch.matmul(x,self.get_item_matrix_weight_transpose())
        # batch_size * blocks * num_items
        prediction_attn=nn.Softmax(dim=-1)(prediction)
        # batch_size * blocks * num_items
        # a=a.unsqueeze(2).repeat(1,1,prediction.size(2))
        # predictions=torch.mul(a,prediction)
        loss=0
        # for i in range(prediction.size(1)):
        #     loss+=self.criterion(prediction[:,i,:],pos_item)

        pos_scores = multi_user_item_atten[:, :, 0]
        pos_labels = torch.ones_like(pos_scores)
        pos_a = a[:, :, 0]
        pos_scores=pos_scores
        a_loss=torch.abs(pos_a).mean()
        pos_loss=-torch.log(pos_scores+self.gamma).mean()

        neg_scores=multi_user_item_atten[:,:,1:]
        neg_label=torch.ones_like(neg_scores)
        neg_a=a[:,:,1:]
        neg_scores=neg_label-neg_scores
        neg_loss=-torch.log((neg_scores+self.gamma+neg_a)/2).mean()*self.neg_numbers

        loss+=pos_loss+neg_loss+a_loss

        return loss


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def prediction(self,seq_item,user,similar_user):
        pu = self.forward(seq_item, user, similar_user)
        # batch_size * embedding_size
        w, multi_user=self.peb.forward_for_prediction(pu=pu)
        """
        w: batch_size * blocks
        multi_user: batch_size * blocks * embedding_size
        """
        prediction=torch.matmul(multi_user,self.get_item_matrix_weight_transpose())
        # batch_size * blocks * number_items
        number_items=prediction.size(2)
        w=w.unsqueeze(2).repeat(1,1,number_items)
        prediction=torch.mul(w,prediction).sum(dim=1)
        # batch_size * number_items

        return prediction


class DIB(nn.Module):

    def __init__(self,embedding_size=32):
        super(DIB, self).__init__()
        self.embedding_size=embedding_size


    def forward(self,current_user_embedding,similar_user_embedding,current_item_embedding,mask=None):
        """
        mask: batch_size * seq_len * similar_user_len
        :param current_user_embedding: batch_size * embedding_size
                == unsqueeze dim 1 and unsqueeze dim 1 and repeat(1,similar_user_len,1)
                == >> batch_size * seq_len * similar_user_len * embedding_size
        :param similar_user_embedding: batch_size * seq_len * similar_user_len * embedding_size
        ***
        output=current_user_embedding * similar_user_embedding and sum(dim=2)
            == >> batch_size * seq_len * similar_user_len
            == >> unsqueeze dim=3 and unsqueeze dim 3 == >> batch_size * similar_user_len * 1
            == >> repeat (1,1,1,embedding_size) == >> batch_size * seq_len * similar_user_len * embedding_size
        ***
        output=output*similar_user_embedding_value and sum(dim=1)
        == >> batch_size * seq_len * embedding_size

        current_item_embedding: batch_size * seq_len * embedding_size
        output=current_item_embedding + output
        :return:
        """
        similar_user_embedding_value=similar_user_embedding

        batch_size,seq_len,similar_user_len,embedding_size=similar_user_embedding.size()

        current_user_embedding=current_user_embedding.unsqueeze(1).unsqueeze(1).repeat(1,seq_len,similar_user_len,1)
        output=torch.mul(current_user_embedding,similar_user_embedding).sum(dim=3)
        # batch_size * seq_len * similar_user_len

        if mask is not None:
            output.masked_fill_(mask,-1e9)

        output=nn.Softmax(dim=2)(output)

        output = output.unsqueeze(3).repeat(1, 1, 1, embedding_size)

        output=torch.mul(output,similar_user_embedding_value).sum(dim=2)
        output+=current_item_embedding

        return output


class PEB(nn.Module):

    def __init__(self,embedding_size=32,hidden_size=32,blocks=3,neg_numbers=3):
        super(PEB, self).__init__()
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.blocks=blocks
        self.neg_numbers=neg_numbers
        self.B=nn.Linear(in_features=self.embedding_size+self.hidden_size,out_features=self.blocks)
        self.multi_user=nn.Linear(in_features=self.embedding_size+self.hidden_size,out_features=self.blocks*self.embedding_size)
        self.sigmoid=nn.Sigmoid()


    def forward(self,pu,pos_item_embedding,neg_item_embedding):
        """

        :param pu: batch_size * embedding_size
        :param pos_item_embedding: batch_size * embedding_size
        :param neg_item_embedding: batch_size * neg_numbers * embedding_size
        :return:
        """
        pos_item_embedding=pos_item_embedding.unsqueeze(1)
        # batch_size * 1 * embedding_size
        item_embedding=torch.cat([pos_item_embedding,neg_item_embedding],dim=1).transpose(1,2)
        # batch_size * (neg_numbers+1) * embedding_size
        # transpose ==>> batch_size * embedding_size * (neg_numbers+1)
        batch_size=pu.size(0)

        # b=self.sigmoid(self.B(pu))
        b = self.B(pu)
        # pu = B==>> batch_size * blocks
        w=nn.Softmax(dim=1)(b)
        # softmax the b to get the w

        multi_user=self.multi_user(pu).view(batch_size,self.blocks,-1)
        # pu = multi_user matrix ==>> batch_size * blocks * embedding_size
        multi_user_item=torch.matmul(multi_user,item_embedding)
        # batch_size * blocks * (1+neg_numbers)
        multi_user_item_atten=nn.Softmax(dim=-1)(multi_user_item)


        length=multi_user_item_atten.size(2)
        b=b.unsqueeze(2).repeat(1,1,length)
        # batch_size * blocks * length
        a=torch.abs(torch.sub(b,multi_user_item_atten))

        return multi_user,multi_user_item_atten,a


    def forward_for_prediction(self,pu):
        batch_size = pu.size(0)

        b = self.B(pu)
        # pu = B==>> batch_size * blocks
        w = nn.Softmax(dim=1)(b)
        # batch_size * blocks
        # softmax the b to get the w

        multi_user = self.multi_user(pu).view(batch_size, self.blocks, -1)
        # pu = multi_user matrix ==>> batch_size * blocks * embedding_size

        return w,multi_user


class Data_for_GRU_DIB_PEB(DataSet):

    def __init__(self,seq_len=10,data_name="ml-1m",neg_numbers=3,similar_user_len=3):
        super(Data_for_GRU_DIB_PEB, self).__init__(data_name=data_name,min_item_number=30,min_user_number=30,user_id_index=1,time_id_index=4,item_id_index=2,sep="\t")
        self.seq_len=seq_len
        self.data_name=data_name
        self.neg_numbers=neg_numbers
        self.similar_user_len=similar_user_len


    def get_data_for_model(self):
        data_values=self.data.values
        user_item={}
        item_user={}
        for value in data_values:
            user_id=value[0]
            item_id=value[1]
            if item_id in item_user.keys():
                item_user[item_id]+=[user_id]
            else:
                item_user[item_id]=[user_id]
        item_user[0] = [0]
        for i in range(self.similar_user_len):
            item_user[0]+=[0]

        for value in data_values:
            user_id=value[0]
            item_id=value[1]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
            else:
                user_item[user_id]=[item_id]

        train_data,validation_data,test_data=[],[],[]
        train_similar,validation_similar,test_similar=[],[],[]
        for user,item_list in user_item.items():
            length=len(item_list)
            if length<5:
                continue

            test=item_list[-self.seq_len-1:]
            test=self.padding(test,seq_len=self.seq_len+1)
            similar = self.pad_similar_user(test[:-1], item_user=item_user, similar_user_len=self.similar_user_len)
            test+=[user]
            test_data.append(test)
            test_similar.append(similar)

            valid=item_list[-self.seq_len-2:-1]
            valid=self.padding(valid,seq_len=self.seq_len+1)
            similar = self.pad_similar_user(valid[:-1], item_user=item_user, similar_user_len=self.similar_user_len)
            valid+=[user]
            validation_data.append(valid)
            validation_similar.append(similar)

            for i in range(1,length-2):
                train=item_list[:i+1]
                train=self.padding(train,self.seq_len+1)
                similar = self.pad_similar_user(train[:-1], item_user, self.similar_user_len)
                train+=[user]
                target=train[-1]
                neg=self.generate_negatives(target,self.neg_numbers)
                train+=neg
                train_data.append(train)
                train_similar.append(similar)
        train_data=np.array(self.pad_sequences(train_data,max_length=self.seq_len+2+self.neg_numbers))
        validation_data=np.array(self.pad_sequences(validation_data,max_length=self.seq_len+2))
        test_data=np.array(self.pad_sequences(test_data,max_length=self.seq_len+2))

        return [train_data,train_similar,validation_data,validation_similar,test_data,test_similar]


    def pad_similar_user(self,list_test,item_user,similar_user_len=3):
        similar_user=[]
        for i in list_test:
            user_list=item_user[i]
            user_list=user_list[-similar_user_len:]
            length=len(user_list)
            while length<similar_user_len:
                user_list=[0]+user_list
                length+=1
            similar_user.append(user_list)

        return similar_user


    def generate_negatives(self,target,neg_numbers):
        neg=[]
        for i in range(neg_numbers):
            x=np.random.choice(self.item_number)
            if x==0 or x==target:
                x=np.random.choice(self.item_number)
            neg.append(x)

        return neg


    def padding(self,item_list,seq_len=10):
        length=len(item_list)
        item_list=item_list[-self.seq_len-1:]
        while length<seq_len:
            length+=1
            item_list=[0]+item_list

        return item_list


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,datas):
        train_data_value,train_similar_user = datas
        total_loss=0
        count=1
        for data,similar_users in zip(get_batch(train_data_value,batch_size=self.model.batch_size),get_batch(train_similar_user,batch_size=self.model.batch_size)):
            """
            train_data: (seq_item,user,similar_user,neg_item,pos_item)
            """
            self.optimizer.zero_grad()
            seq_item = data[:, :self.model.seq_len]
            pos_item=data[:,self.model.seq_len]
            user=data[:,self.model.seq_len+1]
            similar_user=similar_users
            neg_item=data[:,-self.model.neg_numbers:]
            seq_item,user,similar_user,neg_item,pos_item=torch.LongTensor(seq_item),torch.LongTensor(user),torch.LongTensor(similar_user),torch.LongTensor(neg_item),torch.LongTensor(pos_item)
            loss = self.model.calculate_loss(seq_item,user,similar_user,neg_item,pos_item)
            if count%50==0:
                print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.model.logging()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,train_similar,validation_data,validation_similar,test_data,test_similar=self.model.dataset.get_data_for_model()
        for episode in range(self.model.episodes):
            loss=self.train_epoch(datas=[train_data,train_similar])
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                """
                data: (seq_item,user,similar_user)
                """
                validation=validation_data
                label = validation[:, self.model.seq_len]
                validation=torch.LongTensor(validation)
                seq_item = validation[:, :self.model.seq_len]
                user=validation[:,self.model.seq_len+1]
                similar_user=torch.LongTensor(validation_similar)
                scores=self.model.prediction(seq_item,user,similar_user)
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[5,10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=GRU_DIB_PEB(data_name="ml-100k",learning_rate=0.01)
trainer=trainer(model=model)
trainer.train()

# data=Data_for_GRU_DIB_PEB()
# train_data,train_similar,validation_data,validation_similar,test_data,test_similar=data.get_data_for_model()