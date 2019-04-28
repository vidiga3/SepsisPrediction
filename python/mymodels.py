import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset

def loadDataSet(df,model_type,classificationLabel='angus'):
    data = df.loc[:,df.columns != 'angus'].values
    target = torch.from_numpy(df.angus.values).type(torch.LongTensor)
    if(model_type=='CNN'):
    	data = torch.from_numpy(data.astype('float32')).unsqueeze(1)
    elif(model_type == 'RNN'):
    	data = torch.from_numpy(data.astype('float32')).unsqueeze(2)
    
    dataset = TensorDataset(data, target)
    
    return dataset


class MyCNN(nn.Module):
	def __init__(self,size):
		super(MyCNN, self).__init__()
		self.size = size
		self.conv1 = nn.Conv1d(in_channels=1,out_channels=6,kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(in_channels=6,out_channels=16,kernel_size=5)
		self.linear1 = nn.Linear(self.size,128)
		self.linear2 = nn.Linear(128,64)
		self.linear_out = nn.Linear(128,2)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		# print(x.shape)
		x = x.view(-1, self.size)
		x = F.relu(self.linear1(x))
		x = self.linear_out(x)
		return x

class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
		# print('rnn',self.rnn)
		self.fc  = nn.Linear(16,5)
		# print('fc',self.fc.weight.shape)

	def forward(self, x):
		# print(0,x.shape)
		x, _ = self.rnn(x)
		# print(1,x.shape)
		x = torch.tanh(x[:, -1, :])
		# print(2,x.shape)
		x = self.fc(x)
		# print(3,x.shape)

		return x