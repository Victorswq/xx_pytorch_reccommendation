from dataset.utils import *
from utils import *
from mertic import *
from dataset.data_for_repeatnet import *
import torch
from torch import nn
from abstract_model import abstract_model
from layers.layers import Repeat_explore_mechanism,Repeat_recommendation_decoder,Explore_recommendation_decoder
import numpy as np
import torch.optim as optim


class RepeatNet(abstract_model):

    def __init__(self,
                 data_name="ml-1m",
                 model_name="RepeatNet",
                 hidden_size=32,
                 embedding_size=32,
                 learning_rate=0.001,
                 max_seq_length=10,
                 batch_size=512,
                 verbose=1,
                 episodes=50,):
        super(RepeatNet, self).__init__(data_name=data_name,model_name=model_name)
        self.data_name=data_name
        self.model_name=model_name
        self.hidden_size=hidden_size
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.max_seq_length=max_seq_length
        self.episodes=episodes
        self.verbose=verbose
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.build_variable()


    def build_variable(self):
        self.item_matrix=nn.Embedding(self.item_number+1,self.embedding_size,padding_idx=0)
        self.item_matrix_for_explore=nn.Embedding(self.item_number+1,self.hidden_size*2)
        self.repeat_explore_mechanism=Repeat_explore_mechanism(hidden_size=self.hidden_size)
        self.repeat_recommendation_decoder=Repeat_recommendation_decoder(hidden_size=self.hidden_size)
        self.explore_recommendation_decoder=Explore_recommendation_decoder(hidden_size=self.hidden_size)
        self.gru=nn.GRU(input_size=self.embedding_size,hidden_size=self.hidden_size,batch_first=True)
        self.get_matrix_for_repeat()


    def get_item_matrix_T(self):
        return self.item_matrix_for_explore.weight.t()


    def get_matrix_for_repeat(self):
        _, num_item = self.get_item_matrix_T().size()
        self.item_matrix_for_repeat = torch.zeros((self.batch_size, num_item), requires_grad=True)

    def forward(self,seq_item,target):
        seq_item_embedding=self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        all_memory,_=self.gru(seq_item_embedding)
        last_memory=all_memory[:,-1,:]
        repeat_explore_possible=self.repeat_explore_mechanism(last_memory,all_memory)
        # batch_size * 2
        timeline_mask = torch.BoolTensor(seq_item == 0)
        repeat=self.repeat_recommendation_decoder.forward(seq_item,last_memory,all_memory,timeline_mask,self.item_matrix_for_repeat)
        # repeat=self.item_matrix_for_repeat
        # batch_size * num_item
        explore=self.explore_recommendation_decoder.forward(last_memory,all_memory,self.get_item_matrix_T(),seq_item)

        # batch_size * num_item
        x=repeat_explore_possible[:,0]
        x=x.unsqueeze(1).repeat(1,repeat.size(1))
        repeat=repeat*x
        # batch_size * num_item
        y=repeat_explore_possible[:, 1]
        y=y.unsqueeze(1).repeat(1,explore.size(1))
        explore=explore*y
        # batch_size * num_item
        result=repeat+explore
        # batch_size * num_item
        result_=self.gather_indexes(result,target)
        repeat_=self.gather_indexes(repeat,target)
        explore_=self.gather_indexes(explore,target)
        return result_,repeat_,explore_


    def forward_for_prediction(self,seq_item):
        seq_item_embedding = self.item_matrix(seq_item)
        # batch_size * seq_len * embedding_size
        all_memory, _ = self.gru(seq_item_embedding)
        last_memory = all_memory[:, -1, :]
        repeat_explore_possible = self.repeat_explore_mechanism(last_memory, all_memory)
        # batch_size * 2
        timeline_mask = torch.BoolTensor(seq_item == 0)
        repeat = self.repeat_recommendation_decoder.forward(seq_item, last_memory, all_memory, timeline_mask,
                                                            self.item_matrix_for_repeat)
        # repeat=self.item_matrix_for_repeat
        # batch_size * num_item
        explore = self.explore_recommendation_decoder.forward(last_memory, all_memory, self.get_item_matrix_T(),
                                                              seq_item)

        # batch_size * num_item
        x = repeat_explore_possible[:, 0]
        x = x.unsqueeze(1).repeat(1, repeat.size(1))
        repeat = repeat * x
        # batch_size * num_item
        y = repeat_explore_possible[:, 1]
        y = y.unsqueeze(1).repeat(1, explore.size(1))
        explore = explore * y
        # batch_size * num_item
        result = repeat + explore
        # batch_size * num_item
        return result


    def calculate_loss(self,data):
        seq_item,target,in_session=data
        result,repeat,explore=self.forward(seq_item,target)

        pos_label=torch.ones_like(result)
        loss_rec=self.criterion(result,pos_label)

        indices_in_session = np.where(in_session != 0)
        indices_out_session=np.where(in_session == 0)
        loss_mode=self.criterion(repeat[indices_in_session],pos_label[indices_in_session])
        loss_mode+=self.criterion(explore[indices_out_session],pos_label[indices_out_session])
        loss=loss_rec+loss_mode

        return loss


    def predict(self,data):
        seq_item= data
        result= self.forward_for_prediction(seq_item)

        return result


    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1)
        # gather_index: batch_size * 1
        # output: batch_size * number_item
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
        # output: batch_size


class trainer():
    def __init__(self,model):
        self.model=model
    def train_epoch(self,data):
        train_datas = data
        total_loss=0
        count=1
        for train_data in get_batch(train_datas,batch_size=self.model.batch_size):
            self.optimizer.zero_grad()
            # neg=generate_negative_item(pos,max_item=self.model.item_number)
            # neg=np.array(neg)
            seq_item = train_data[:, :-2]
            target = train_data[:, -2]
            is_in_session = train_data[:, -1]
            seq_item,target,is_in_session=torch.LongTensor(seq_item),torch.LongTensor(target),torch.LongTensor(is_in_session)
            loss = self.model.calculate_loss(data=[seq_item,target,is_in_session])
            if count%1==0:
                print("the %d step  the current loss is %f"%(count,loss))
            count+=1
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss


    def train(self):
        self.model.logging()
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.model.learning_rate)
        train_data,validation_data,test_data=read_data_for_repeat_net(data_name=self.model.data_name,max_seq_length=self.model.max_seq_length)
        for episode in range(self.model.episodes):
            loss=self.train_epoch(data=train_data)
            print("loss is ",loss)

            if (episode+1)%self.model.verbose==0:
                validation=validation_data
                label = validation[:, -1]
                validation=torch.LongTensor(validation)
                validation=validation[:,:-1]
                scores=self.model.prediction(validation)
                s=HR(ratingss=scores.detach().numpy(),pos_items=label,top_k=[10,20])
                self.model.logger.info("Episode %d     HR: %s"%(episode,s))


model=RepeatNet(data_name="diginetica")
trainer=trainer(model)
trainer.train()