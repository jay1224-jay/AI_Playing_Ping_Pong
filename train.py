import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

import sys

data_size = 0
with open("./human_records.txt", "r") as f:
    lines = f.read().split()
    data_size = len(lines)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HEIGHT = 400
RECT_HEIGHT = 70

data_n = 0
x = [  ] 
y = [  ]
with open("./human_records.txt", "r") as f:
    lines = f.read().split('\n')
    data_size = len(lines)
    
    for i in range(len(lines)):
        try:
            line = lines[i].split(',')
            # print(lines)
            player_y    = float(line[0])
            ball_x      = float(line[1])
            ball_y      = float(line[2])
            ball_dir_x  = float(line[3])
            ball_dir_y  = float(line[4])
            
            ratio    = float(line[5])
            dummy = [ float(x) for x in line ][1:-1]
            x.append(dummy)
            y.append(ratio)
            data_n += 1
        except:
            pass

data_size = data_n
print(data_size)
# print(np.array(x))
# print(np.array(y))
x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test = x[train_split:]
y_test = y[train_split:]

input_size = 4
hidden_size = 128
output_size = 1
# Step 2: Define Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), int(hidden_size/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/4), 1)    
        )

    def forward(self, x):
        return self.layers(x)

# Hyperparameters

model = SimpleNN()

# Step 3: Training Setup
# criterion = nn.CrossEntropyLoss()  # Loss function
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Optimizer
batch_size = 32  # Batch size
num_epochs = 1000  # Number of epochs

model.to(device)

accs = []
y_pred = model(x_train).squeeze()
print(y_pred)
print(y_train)
# Step 4: Train the Model
for epoch in range(num_epochs+1):
    model.train()
    
    y_pred = model(x_train).squeeze()
    
    # for i in range(data_size-train_split):
        # print(round(y_pred[i].item()), y_train[i].item())
    
    
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation step
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        test_pred = model(x_test).squeeze()
        val_loss += loss_fn(test_pred, y_test)
        total = data_size - train_split
        
        for i in range(total):
            test_ball_y = y_test[i].item() * HEIGHT
            pred_player_y = test_pred[i].item() * HEIGHT
            
            # hit accuarcy
            if ( test_ball_y >= pred_player_y and test_ball_y <= pred_player_y + RECT_HEIGHT ):
                # hit the ball successfully
                correct += 1
       
    
    val_accuracy = 100 * correct / total
    accs.append(val_accuracy)
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Step 5: Test the Model
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
model.eval()
correct = 0
total = 0
test_pred = []
with torch.no_grad():
    test_pred = model(x_test)
    val_loss += loss_fn(test_pred, y_test)
    total = data_size - train_split
    
    for i in range(total):
        test_ball_y = y_test[i].item() * HEIGHT
        pred_player_y = test_pred[i].item() * HEIGHT
        
        # hit accuarcy
        if ( test_ball_y >= pred_player_y and test_ball_y <= pred_player_y + RECT_HEIGHT ):
            # hit the ball successfully
            correct += 1

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


torch.save(model.state_dict(), f'pp_model_{hidden_size}_hiddensize_4_layer_lr.pth')
print(f"Model saved as 'pp_model_{hidden_size}_hiddensize_4_layer.pth'")


plt.figure(figsize=(10, 7))

plt.scatter([x for x in range(len(accs))], accs, c="b", s=4, label="Accuracy")

plt.legend(prop={"size": 14})
plt.savefig(f"pp_model_{hidden_size}_hiddensize_4_layer_lr.png", dpi=300)
plt.show()


