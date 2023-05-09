import numpy as np
import torch
import torch.nn
import torch.nn.functional as F


class point_cloud(torch.nn.Module):
    def __init__(self,input,hidden_num1,hidden_num2,output):
        super(point_cloud,self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input,128,dtype=torch.float64),torch.nn.ReLU())

        self.layer2 = torch.nn.Sequential(torch.nn.Linear(128,256,dtype=torch.float64),torch.nn.ReLU())

        self.layer3 = torch.nn.Sequential(torch.nn.Linear(256,output,dtype=torch.float64),torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(self.layer1,self.layer2,self.layer3)
    def forward(self,X):

        return self.mlp(X)

class point_empty(torch.nn.Module):
    def __init__(self):
        super(point_empty, self).__init__()
    def forward(self,X):
        return X

class point_cloud_classical(torch.nn.Module):
    def __init__(self,input,output):
        super(point_cloud_classical, self).__init__()

        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input,64,dtype=torch.float64),torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64,128,dtype=torch.float64),torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Linear(128,output,dtype=torch.float64))
        self.mlp = torch.nn.Sequential(self.layer1,self.layer2,self.layer3)
    def forward(self,X):
        return self.mlp(X)

class point_cloud_classical_num():
    def __init__(self,number,input,output):
        self.number = number
        self.point_cloud_map = {}
        for i in range(number):
            self.point_cloud_map[f'model{i}'] = point_cloud_classical(input[i],output[i])

    def get_model(self):
        return self.point_cloud_map

    def __getitem__(self, item):
        return self.get_model()[item]

    def __len__(self):
        return self.number
