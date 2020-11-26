from utils import *
from abstract_model import abstract_model
from layers.layers import feed_forward_layer
import torch.optim as optim
from dataset.utils import *
from mertic import *
import torch
from torch import nn


class SASRec(abstract_model):
    def __init__(self,
                 model_name="SASRec",
                 data_name="ml-1m",
                 learning_rate=0.001,
                 batch_size=128,
                 epochs=20,
                 embedding_size=100,
                 max_seq_length=50,
                 num_blocks=1,
                 n_head=1,
                 nearest_length=200,
                 dropout_rate=0.2,
                 semi_size=50,
                 episodes=50,
                 verbose=1
                 ):
        self.data_name=data_name
        super(SASRec, self).__init__(self.data_name,model_name)
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.epochs=epochs
        self.embedding_size=embedding_size
        self.max_seq_length=max_seq_length
        self.num_blocks=num_blocks
        self.n_head=n_head
        self.nearest_length=nearest_length
        self.dropout_rate=dropout_rate
        self.semi_size=semi_size
        self.episodes=episodes
        self.verbose=verbose

        # set up the variable
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.build_variables()


    def build_variables(self):
        self.item_matrix=nn.Embedding(self.item_number+1,self.embedding_size,padding_idx=0)
        self.position_matrix=nn.Embedding(self.max_seq_length+1,self.embedding_size,padding_idx=0)
        self.emb_layer_norm=nn.LayerNorm(self.embedding_size)
        self.attentive_layer=nn.ModuleList()
        self.feedward_layer=nn.ModuleList()
        self.attentive_norm_layer=nn.ModuleList()
        self.feedward_norm_layer=nn.ModuleList()
        for i in range(self.num_blocks):
            new_layer_norm=nn.LayerNorm(self.embedding_size)
            self.attentive_norm_layer.append(new_layer_norm)

            new_attentive_layer=nn.MultiheadAttention(self.embedding_size,self.n_head,dropout=self.dropout_rate)
            self.attentive_layer.append(new_attentive_layer)

            new_feed_forward_layer=feed_forward_layer(embedding_size=self.embedding_size,semi_size=self.semi_size,dropout_rate=self.dropout_rate)
            self.feedward_layer.append(new_feed_forward_layer)

            new_layer_norm_=nn.LayerNorm(self.embedding_size)
            self.feedward_norm_layer.append(new_layer_norm_)


    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


    def get_attn_subsequence_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        ones = torch.ones(attn_shape, dtype=torch.uint8)
        subsequence_mask = ones.triu(diagonal=1)
        return subsequence_mask


    def item_matrix_transpose(self):
        return self.item_matrix().weight.t()


    def forward(self,seq_data,position):
        item_seq_embedding=self.item_matrix(seq_data)
        position_embedding=self.position_matrix(position)
        # print("the shape of item embedding is ",item_seq_embedding.size())
        # print("the shape of position is ",position_embedding.size())
        item_seq_embedding+=position_embedding

        # get the mask
        timeline_mask = torch.BoolTensor(seq_data == 0)
        item_seq_embedding *= ~timeline_mask.unsqueeze(-1)

        tl = seq_data.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))

        for i in range(self.num_blocks):
            Q=self.attentive_norm_layer[i](item_seq_embedding)
            Q=torch.transpose(Q,0,1)
            item_seq_embedding = torch.transpose(item_seq_embedding, 0, 1)
            item_seq_embedding,_=self.attentive_layer[i](Q,item_seq_embedding,item_seq_embedding,attn_mask=attention_mask)
            item_seq_embedding+=Q
            item_seq_embedding = torch.transpose(item_seq_embedding, 0, 1)
            item_seq_embedding=self.feedward_norm_layer[i](item_seq_embedding)
            item_seq_embedding=self.feedward_layer[i].forward(item_seq_embedding)
            item_seq_embedding *= ~timeline_mask.unsqueeze(-1)

        return item_seq_embedding


    def calculate_loss(self,data):
        seq_data,position,pos_item,neg_item=data
        item_pos_for_use=pos_item
        seq_data,position,pos_item,neg_item=torch.LongTensor(seq_data),torch.LongTensor(position),torch.LongTensor(pos_item),torch.LongTensor(neg_item)
        item_seq_embedding=self.forward(seq_data,position)


        pos_item = self.item_matrix(pos_item)
        # batch_size * seq_len * embedding_size
        pos_logits = (item_seq_embedding * pos_item).sum(dim=-1)
        # batch_size * seq_len

        neg_item = self.item_matrix(neg_item)
        neg_logits = (item_seq_embedding * neg_item).sum(dim=-1)
        # batch_size * seq_len
        # batch_size * seq_len * embedding_size ==>> batch_size * seq_len
        pos_labels, neg_labels = torch.ones_like(pos_logits), torch.zeros_like(neg_logits)
        indices = np.where(item_pos_for_use != 0)
        loss = self.criterion(pos_logits[indices], pos_labels[indices])
        loss += self.criterion(neg_logits[indices], neg_labels[indices])
        return loss


    def prediction(self,data):
        seq_data, position,neg= data
        prediction = self.forward(seq_data, position)
        prediction=prediction[:,-1,:]
        item_embedding=self.item_matrix(neg)
        print(prediction.size())
        print(item_embedding.size())
        prediction=torch.matmul(prediction.unsqueeze(1),item_embedding.transpose(1,2))
        # batch_size * 1 * seq_len
        prediction=prediction.squeeze(1)
        print(prediction.size())
        # batch_size * item_num
        return prediction


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,data):
        train_data, train_position, neg_item = data
        pos_item=train_data[:,1:]
        train_data=train_data[:,:-1]
        total_loss=0
        count=1
        for train_da,position,pos,neg in zip(get_batch(train_data),get_batch(train_position),get_batch(pos_item),get_batch(neg_item)):
            self.optimizer.zero_grad()
            # neg=generate_negative_item(pos,max_item=self.model.item_number)
            # neg=np.array(neg)
            loss = self.model.calculate_loss(data=[train_da,position,pos,neg])
            if count%5==0:
                print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.model.logging()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,validation_data,test_data=data_for_sasrec(data_name=self.model.data_name,max_length=self.model.max_seq_length)
        a,b,c=train_data
        print(a.shape)
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                validation,position=validation_data
                label = validation[:, -1]
                validation,position=torch.LongTensor(validation),torch.LongTensor(position)
                validation=validation[:,:-1]
                neg=generate_negative_item(np.ones((validation.size(0),101)),max_item=self.model.item_number)
                neg=np.array(neg)
                neg[:,-1]=label
                neg=torch.LongTensor(neg)
                scores=self.model.prediction([validation,position,neg])
                s=HR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                self.model.logger.info("Episode %d     HR: %s"%(episode,s))


model=SASRec(max_seq_length=50)
trainer=trainer(model=model)
trainer.train()