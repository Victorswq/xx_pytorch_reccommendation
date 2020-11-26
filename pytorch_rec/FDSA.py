from dataset.utils import *
from mertic import *
import torch
import torch.nn as nn
import torch.optim as optim
from abstract_model import abstract_model
from Dataset import DataSet


class FDSA(abstract_model):

    def __init__(self,
                 data_name="ml-1m",
                 model_name="FDSA",
                 embedding_size=64,
                 n_head=1,
                 episodes=10,
                 verbose=10,
                 learning_rate=0.001,
                 seq_len=30,
                 min_user_inter=15,
                 min_item_inter=20,
                 cate_number=10000,
                 brand_number=10000,
                 description_number=6000,
                 num_blocks=2,
                 dropout_rate=0.001,
                 batch_size=512,
                 ):
        super(FDSA, self).__init__(data_name=data_name,model_name=model_name)
        self.embedding_size=embedding_size
        self.n_head=n_head
        self.episodes=episodes
        self.verbose=verbose
        self.learning_rate=learning_rate
        self.seq_len=seq_len
        self.min_user_number=min_user_inter
        self.min_item_number=min_item_inter
        self.cate_number=cate_number
        self.brand_number=brand_number
        self.description_number=description_number
        self.num_blocks=num_blocks
        self.dropout_rate=dropout_rate
        self.batch_size=batch_size

        self.criterion=nn.BCEWithLogitsLoss()

        self.dataset=Data_for_FDSA(data_name=self.data_name,seq_len=self.seq_len,min_user_inter=self.min_user_number,min_item_inter=self.min_item_number)
        self.item_number=self.dataset.item_number
        self.user_number=self.dataset.user_number

        self.build_variables()

        self.apply(self.init_weights)


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number,self.embedding_size,padding_idx=0)
        self.cate_matrix=nn.Embedding(self.cate_number,self.embedding_size,padding_idx=0)
        self.brand_matrix=nn.Embedding(self.brand_number,self.embedding_size,padding_idx=0)
        self.description_matrix=nn.Embedding(self.description_number,self.embedding_size,padding_idx=0)
        self.vanilla_attention_layer=Vanilla_Attention_Layer(embedding_size=self.embedding_size)

        self.item_position=nn.Embedding(self.batch_size,self.embedding_size,padding_idx=0)
        self.feature_position=nn.Embedding(self.batch_size,self.embedding_size)

        self.item_QKV_list=nn.ModuleList()
        self.item_FFN_list=nn.ModuleList()
        self.feature_QKV_list=nn.ModuleList()
        self.feature_FFN_list=nn.ModuleList()

        for i in range(self.num_blocks):
            item_qkv=QKV(embedding_size=self.embedding_size,n_head=self.n_head)
            self.item_QKV_list.append(item_qkv)
            feature_qkv=QKV(embedding_size=self.embedding_size,n_head=self.n_head)
            self.feature_QKV_list.append(feature_qkv)
            item_ffn=FFN(embedding_size=self.embedding_size,dropout_rate=self.dropout_rate)
            self.item_FFN_list.append(item_ffn)
            feature_ffn=FFN(embedding_size=self.embedding_size,dropout_rate=self.dropout_rate)
            self.feature_FFN_list.append(feature_ffn)

        self.w1=nn.Linear(2*self.embedding_size,self.embedding_size)
        self.dropout_1=nn.Dropout(self.dropout_rate)


    def get_item_matrix_weight_transpose(self):

        return self.item_matrix.weight.t()


    def forward(self,data):
        seq_item,cate,brand=data

        # get the item information
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        batch_size,seq_len,embedding_size=seq_item_embedding.size()

        padding_mask = seq_item.data.eq(0).unsqueeze(1).expand(batch_size,seq_len,seq_len)
        # batch_size * seq_len * seq_len

        sequence_mask=torch.ones(size=(batch_size,seq_len,seq_len))
        sequence_mask=sequence_mask.triu(diagonal=1)

        mask=torch.gt((padding_mask+sequence_mask),0)

        # get the feature information
        cate_embedding=self.cate_matrix(cate).unsqueeze(2)
        brand_embedding=self.brand_matrix(brand).unsqueeze(2)
        # description_embedding=self.description_matrix(description).unsqueeze(2)
        # batch_size * seq_len * 1 * embedding_size

        feature_embedding=torch.cat([cate_embedding,brand_embedding],dim=2)
        # batch_size * seq_len * 3 * embedding_size

        feature_embedding=self.vanilla_attention_layer.forward(all_memory=feature_embedding)
        # batch_size * seq_len * embedding_size

        for qkv,ffn in zip(self.item_QKV_list,self.item_FFN_list):
            seq_item_embedding=qkv.forward(seq_item_embedding,seq_item_embedding,seq_item_embedding,mask)
            seq_item_embedding=ffn.forward(seq_item_embedding)
            # batch_size * seq_len * embedding_size

        for qkv,ffn in zip(self.feature_QKV_list,self.feature_FFN_list):
            feature_embedding=qkv.forward(feature_embedding,feature_embedding,feature_embedding,mask)
            feature_embedding=ffn.forward(feature_embedding)
            # batch_size * seq_len * embedding_size

        item_feature_embedding=torch.cat([seq_item_embedding,feature_embedding],dim=2)
        # batch_size * seq_len * 2_mul_embedding_size
        item_feature_embedding=self.dropout_1(self.w1(item_feature_embedding))
        # batch_size * seq_len * embedding_size

        return item_feature_embedding


    def calculate_loss(self,data):
        seq_item, cate, brand,pos_item,neg_item=data
        item_feature_embedding=self.forward([seq_item,cate,brand])
        # batch_size * seq_len * embedding_size

        pos_item_embedding=self.item_matrix(pos_item)
        neg_item_embedding=self.item_matrix(neg_item)
        # batch_size * seq_len * embedding_size

        pos_scores=torch.mul(item_feature_embedding,pos_item_embedding).sum(dim=2)
        neg_scores=torch.mul(item_feature_embedding,neg_item_embedding).sum(dim=2)
        # batch_size * seq_len

        pos_label=torch.ones_like(pos_scores)
        neg_label=torch.zeros_like(neg_scores)
        indices = np.where(seq_item != 0)

        loss=self.criterion(pos_scores[indices],pos_label[indices])
        loss+=self.criterion(neg_scores[indices],neg_label[indices])

        return loss


    def prediction(self,data):
        seq_item, cate, brand= data
        item_feature_embedding = self.forward([seq_item, cate, brand])
        item_feature_embedding=item_feature_embedding[:,-1,:]
        # batch_size * embedding_size

        prediction=torch.matmul(item_feature_embedding,self.get_item_matrix_weight_transpose())

        return prediction


class QKV(nn.Module):

    def __init__(self,embedding_size=32,n_head=1):
        super(QKV, self).__init__()
        self.embedding_size=embedding_size
        self.n_head=n_head
        self.w1=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w2=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w3=nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.w4=nn.Linear(self.embedding_size,self.embedding_size)
        self.layernorm=nn.LayerNorm(self.embedding_size)


    def forward(self,seq_q,seq_k,seq_v,mask=None):
        """

        :param seq_q: batch_size * seq_len * embedding_size
        :param seq_k: batch_size * seq_len * embedding_size
        :param seq_v: batch_size * seq_len * embedding_size
        :param mask: batch_size * seq_len * seq_len
        :return:
        """
        batch_size,seq_len,embedding_size=seq_q.size()

        seq_v=self.layernorm(seq_v)
        residual=seq_v

        seq_q=self.w1(seq_q).unsqueeze(2).view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        seq_k=self.w2(seq_k).unsqueeze(2).view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        # batch_size * seq_len * 1 * embedding_size
        # == >> batch_size * seq_len * n_head * embedding_size
        # == >> batch_size * n_head * seq_len * embedding_size
        seq_v=self.w3(seq_v).unsqueeze(2).view(batch_size,seq_len,self.n_head,-1).transpose(1,2)
        # batch_size * n_head * seq_len * embedding_size
        q_k=torch.matmul(seq_q,seq_k.transpose(2,3))
        # batch_size * n_head * seq_len * seq_len

        # get_the mask
        if mask is not None:
            mask=mask.unsqueeze(1).repeat(1,self.n_head,1,1)
            # batch_size * n_head * seq_len * seq_len
            q_k.masked_fill_(mask,-1e9)

        q_k=nn.Softmax(dim=-1)(q_k)
        # batch_size * n_head * seq_len * seq_len

        output=torch.matmul(q_k,seq_v)
        # batch_size * n_head * seq_len * embedding_size

        output=output.transpose(1,2).view(batch_size,seq_len,-1)
        # batch_size * seq_len * embedding_size
        output=self.w4(output)
        # batch_size * seq_len * embedding_size

        output+=residual

        return self.layernorm(output)


class FFN(nn.Module):

    def __init__(self,embedding_size=32,dropout_rate=0.2):
        super(FFN, self).__init__()
        self.embedding_size=embedding_size
        self.dropout_rate=dropout_rate
        self.layernorm=nn.LayerNorm(self.embedding_size)
        self.conv1d_1=nn.Conv1d(in_channels=self.embedding_size,out_channels=self.embedding_size,kernel_size=1)
        self.dropout_1=nn.Dropout(self.dropout_rate)
        self.conv1d_2=nn.Conv1d(in_channels=self.embedding_size,out_channels=self.embedding_size,kernel_size=1)
        self.dropout_2=nn.Dropout(self.dropout_rate)
        self.relu=nn.ReLU()


    def forward(self,seq_embedding):
        """

        :param seq_embedding: batch_size * seq_len * embedding_size
            == transpose(1,2) == >> batch_size * embedding_size * seq_len
            == conv1d == >> batch_size * embedding_size * seq_len
            == conv1d == >> batch_size * embedding_size * seq_len
            == transpose == >> batch_size * seq_len * embedding_size
        :return:
        """
        seq_embedding=self.layernorm(seq_embedding)

        seq_embedding_value=seq_embedding
        seq_embedding=seq_embedding.transpose(1,2)
        seq_embedding=self.dropout_1(self.relu(self.conv1d_1(seq_embedding)))
        seq_embedding=self.dropout_2(self.relu(self.conv1d_2(seq_embedding)))
        seq_embedding=seq_embedding.transpose(1,2)
        # batch_size * seq_len * embedding_size
        seq_embedding+=seq_embedding_value

        return self.layernorm(seq_embedding)


class Vanilla_Attention_Layer(nn.Module):

    def __init__(self,embedding_size=32):
        super(Vanilla_Attention_Layer, self).__init__()
        self.embedding_size=embedding_size
        self.wf=nn.Linear(self.embedding_size,self.embedding_size)


    def forward(self,all_memory):
        """

        :param all_memory: batch_size * seq_len * 3 * embedding_size
            == wf ==>> batch_size * seq_len * 3 * embedding_size
        ***
        output: all_memory * all_memory_value == sum(dim=-2) == >> batch_size * seq_len * embedding_size
        :return:
        """
        all_memory_value=all_memory

        all_memory=self.wf(all_memory)
        all_memory=nn.Softmax(dim=-2)(all_memory)

        output=torch.mul(all_memory,all_memory_value).sum(dim=-2)

        return output


class Data_for_FDSA(DataSet):

    def __init__(self,seq_len=10,data_name="tmall_test",min_user_inter=15,min_item_inter=30):
        super(Data_for_FDSA, self).__init__(data_name=data_name,min_user_number=min_user_inter,min_item_number=min_item_inter,sep=",")
        self.seq_len=seq_len
        self.data_name=data_name
        self.min_item_number=min_item_inter
        self.min_user_number=min_user_inter


    def get_data_for_model(self):
        self.unique(index=[3,4,5])
        data=self.get_data()

        return data


    def unique(self,index=[3,4,5]):
        columns=self.data.columns.values
        for i in index:
            column=columns[i-1]
            values=self.data[column].values
            new_list=np.zeros_like(values)
            id_dict={}
            id_start=1
            for idx,value in enumerate(values):
                if value in id_dict.keys():
                    new_list[idx]=id_dict[value]
                else:
                    id_dict[value]=id_start
                    new_list[idx]=id_dict[value]
                    id_start+=1
            self.data[column]=new_list


    def get_data(self):
        data_value=self.data.values
        user_item={}
        user_cate={}
        user_brand={}

        for value in data_value:
            user_id=value[0]
            item_id=value[1]
            cate_id=value[2]
            brand_id=value[4]
            if user_id in user_item.keys():
                user_item[user_id]+=[item_id]
                user_cate[user_id]+=[cate_id]
                user_brand[user_id]+=[brand_id]
            else:
                user_item[user_id]=[item_id]
                user_cate[user_id]=[cate_id]
                user_brand[user_id]=[brand_id]

        train_data,validation_data,test_data=[],[],[]
        for user_id,item_list in user_item.items():
            if len(item_list)<5:
                continue
            cate_list=user_cate[user_id]
            brand_list=user_brand[user_id]
            test=item_list[-self.seq_len-1:]+cate_list[-self.seq_len-1:]+brand_list[-self.seq_len-1:]
            test_data.append(test)
            valid=item_list[-self.seq_len-2:-1]+cate_list[-self.seq_len-2:-1]+brand_list[-self.seq_len-2:-1]
            validation_data.append(valid)
            target=item_list[-self.seq_len-2:-2]
            neg=self.get_negative(target)

            train=item_list[-self.seq_len-3:-2]+cate_list[-self.seq_len-3:-2]+brand_list[-self.seq_len-3-2:-2]+neg
            train_data.append(train)

        train_data=self.pad_sequences(train_data,max_length=4*(self.seq_len+1)-1)
        validation_data=self.pad_sequences(validation_data,max_length=3*(self.seq_len+1))
        test_data=self.pad_sequences(test_data,max_length=3*(self.seq_len+1))

        train_data,validation_data,test_data=np.array(train_data),np.array(validation_data),np.array(test_data)

        return [train_data,validation_data,test_data]


    def get_negative(self,target):
        neg=[]
        length=len(target)
        while length<self.seq_len:
            neg+=[0]
            length+=1
        for t in target:
            n=np.random.choice(self.item_number)
            if n==t or n==0:
                n=np.random.choice(self.item_number)
            neg.append(n)

        return neg


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,datas):
        train_data_value = datas
        total_loss=0
        count=1
        for data in get_batch(train_data_value,batch_size=self.model.batch_size):
            """
            train_data: (seq_item, cate, brand, description,pos_item,neg_item)
            """
            self.optimizer.zero_grad()
            seq_items = data[:, :self.model.seq_len + 1]
            cates = data[:, self.model.seq_len + 1:2 * (self.model.seq_len + 1)]
            brands = data[:, 2*(self.model.seq_len +1):3*(self.model.seq_len+1)]
            neg=data[:,-self.model.seq_len:]
            seq_item = seq_items[:, :-1]
            cate_seq_item = cates[:, :-1]
            brand_seq_item = brands[:, :-1]
            seq_item_target = seq_items[:, :self.model.seq_len]
            seq_item,cate,brand,pos_item,neg_item=torch.LongTensor(seq_item),torch.LongTensor(cate_seq_item),torch.LongTensor(brand_seq_item),torch.LongTensor(seq_item_target),torch.LongTensor(neg)
            loss = self.model.calculate_loss(data=[seq_item,cate,brand,pos_item,neg_item])
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
            loss=self.train_epoch(datas=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                """
                data: (seq_item, cate, brand)
                """
                validation=validation_data[:500,:]
                label = validation[:, self.model.seq_len]
                validation=torch.LongTensor(validation)
                cate = validation[:, self.model.seq_len+1:2*(self.model.seq_len + 1)-1]
                brand=validation[:,2*(self.model.seq_len+1):3*(self.model.seq_len+1)-1]
                seq_item = validation[:, :self.model.seq_len]
                scores=self.model.prediction([seq_item,cate,brand])
                HitRatio(ratingss=scores.detach().numpy(),pos_items=label,top_k=[5,10,20])
                MRR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])


model=FDSA(data_name="Tmall",episodes=10000,learning_rate=0.01,verbose=1)
trainer=trainer(model)
trainer.train()