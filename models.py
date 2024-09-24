import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self,size=3) -> None:
        super(CNN_Model,self).__init__()
        self.size = size
        self.conv1 = nn.Conv2d(in_channels=self.size, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding="same")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, self.size*self.size)

    # this function is put here to ensures reverse compatibility with the old model
    @staticmethod
    def create_input_tensor(board):
        # the board is a 1D array of 9 elements
        shape = board.shape
        if len(shape) == 1:
            shape = (1,shape[0])
            board = board.unsqueeze(0)
        x = torch.zeros(shape[0],3,3,3)
        dim_board = board.reshape(-1,3,3)
        x[:,0] = (dim_board == 0) # empty spots 
        x[:,1] = (dim_board == 1) # X spots
        x[:,2] = (dim_board == -1) # O spots
        return x,board

    def forward(self, inp):
        x,inp = CNN_Model.create_input_tensor(inp)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x[(inp != 0.0).nonzero(as_tuple=True)] = float('-inf')  
        return x.squeeze()

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

class DNN(nn.Module):
    def __init__(self,size):
        super(DNN, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(size**2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, size**2)

    def forward(self, inp):
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x[(inp != 0.0).nonzero(as_tuple=True)] = float('-inf')
        x = x
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
