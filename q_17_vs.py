from itertools import chain
import torch
import pandas as pd
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

batch_tensors = {
    'train': {},
    'test': {}
}
batch_loaders = {
    'train': {},
    'test': {}
}

for t in train_dataset:
    img_size_str = f'{t[0].size[0]}x{t[0].size[1]}'
    tensor_image = transforms.ToTensor()(t[0])
    tensor_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor_image)
    if not img_size_str in batch_tensors['train']:
        batch_tensors['train'][img_size_str] = []
        batch_loaders['train'][img_size_str] = None
    batch_tensors['train'][img_size_str].append((tensor_image, t[1]))

for t in test_dataset:
    img_size_str = f'{t[0].size[0]}x{t[0].size[1]}'
    tensor_image = transforms.ToTensor()(t[0])
    tensor_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor_image)
    if not img_size_str in batch_tensors['test']:
        batch_tensors['test'][img_size_str] = []
        batch_loaders['test'][img_size_str] = None
    batch_tensors['test'][img_size_str].append((tensor_image, t[1]))



# TODO Make unique set out of test and train
for img_size_str in list(batch_loaders['train'].keys()):
    batch_loaders['train'][img_size_str] = torch.utils.data.DataLoader(batch_tensors['train'][img_size_str],
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
    batch_loaders['test'][img_size_str] = torch.utils.data.DataLoader(batch_tensors['test'][img_size_str],
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                            shuffle=True,
#                                            transform=transforms)
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                                            shuffle=True,
#                                            transform=transforms)
N = 133
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 72, 3, 1, 1)
        self.fc1 = nn.Linear(72, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = F.adaptive_max_pool2d(x, output_size=1).squeeze(3).squeeze(2)
        x = self.fc1(x)
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
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
    total_all_batch = 0.0
    running_loss = 0.0
    for img_size in list(batch_tensors['train'].keys()):
        for i, data in enumerate(batch_loaders['train'][img_size], 0): # get the inputs; data is a list of [inputs, labels] 
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
            for data in batch_loaders['test'][img_size]:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        total_all_batch += (correct / total)
    results['Epoch'].append(epoch)
    results['Loss'].append(running_loss)
    results['Accuracy'].append(total_all_batch / 3)
    results['Method'].append('VS')
    #print(f'Epoch: {epoch}, acc: {correct / total}, img_size:{img_size}')
df = pd.DataFrame.from_dict(results)
df.to_csv('vs.csv')