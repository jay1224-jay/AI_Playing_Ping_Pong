import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

RECT_WIDTH = 10
RECT_HEIGHT = 70

class PingPongAIMath:
    def __init__(self):
        pass

    def getResponse(self, ball_pos, ball_dir, player_pos):
        
        
        
        t = (player_pos[0] - ball_pos[0]) / ball_dir[0]
        y = ball_pos[1] + t * ball_dir[1]
        
        self.response = 1 if (player_pos[1]+RECT_HEIGHT/2 > y)  else 2

        # 1->up, 2-> down
        return self.response
            

"""
class PingPongAIModel(nn.Module):
    def __init__(self):
        super(PingPongAIModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # First hidden layer with 128 neurons
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)         # Second hidden layer with 64 neurons

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def getResponse(self, ball_pos, ball_dir, player_pos):
        
        x_test = torch.tensor([ball_pos[0], ball_pos[1], ball_dir[0], ball_dir[1]], dtype=torch.float32)
        with torch.inference_mode():
            y_pred = round((x_test)).item()
        print(y_pred)
        self.response = y_pred
        if self.response == 1:
            return "up"
        else:
            return "down"
"""

class Model(nn.Module):
    hidden_size = 128
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size/2), int(self.hidden_size/4)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size/4), 1)    
        )

    def forward(self, x):
        return self.layers(x)
        
    def getResponse(self, ball_pos, ball_dir, player_pos):
        
        x_test = torch.tensor([ball_pos[0], ball_pos[1], ball_dir[0], ball_dir[1]], dtype=torch.float32)
        with torch.inference_mode():
            y_pred = round((x_test)).item()
        print(y_pred)
        self.response = y_pred
        if self.response == 1:
            return "up"
        else:
            return "down"

"""
print(" ==== writing records to human_records.txt ==== ")
            
with open("human_records.txt", "w") as f:
    for i in records:
        f.write("{}, {}, {}\n".format(i[0], i[1], i[2]))

if ( ball_dir[0] < 0 ):
        if n % 50 == 0:
            print("Player (x, y): (%d, %d)" % (player1_rect.x, player1_rect.y))
            print("Ball (x, y): %d, %d" % (ball_x, ball_y))
            print("ball dir: (%d, %d)" % (ball_dir[0], ball_dir[1]))
            records.append([player1_rect.y, (int(ball_x), int(ball_y)), (round(ball_dir[0], 3), round(ball_dir[1], 3))])
        n += 1  
"""
