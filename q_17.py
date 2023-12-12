import torch
import pandas as pd
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

LR = 0.0001
EPOCH = 10
MOMENTUM = 0.9
BATCH_SIZE = 4
transform = transforms.Compose(
    [transforms.Resize((28, 28)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

PREFIX='/var/scratch/dl23037/'
#PREFIX='./'
train_dataset = torchvision.datasets.ImageFolder(root=PREFIX+'mnist-varres/train')
test_dataset = torchvision.datasets.ImageFolder(root=PREFIX+'mnist-varres/test')


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           shuffle=True)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 72, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(72, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

results = {
    'Epoch': [],
    'Loss': [],
    'Accuracy': [],
    'Method': [],
}

net = Net()
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(net.parameters(), lr=LR)
for epoch in range(EPOCH):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0): # get the inputs; data is a list of [inputs, labels] 
        inputs, labels = data # zero the parameter gradients
        optimizer.zero_grad() # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    correct = 0.0
    total = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    results['Epoch'].append(epoch)
    results['Loss'].append(running_loss)
    results['Accuracy'].append(correct / total)
    results['Method'].append('FS')
    #print(f'Epoch: {epoch}, acc: {correct / total}')
df = pd.DataFrame.from_dict(results)
df.to_csv('fs.csv')