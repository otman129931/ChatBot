import json 
from Helper import NLP_helper
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chat_module import ChatNeuralNet

nlp_h=NLP_helper()
with open("data.json",'r') as f:
    intents= json.load(f)
tags=[]
pattern_words=[]
X_Y=[]
for intenet in intents['intents']:
    tag=intenet['tag']
    tags.append(tag)
    for pattern in intenet['patterns']:
        words=nlp_h.tokenezation_remove_stop_word(pattern)
        pattern_words.extend(words)
        X_Y.append((tag, words))
tags=sorted(set(tags))
pattern_words=[nlp_h.word_stema(word) for word in pattern_words ]
pattern_words=sorted(set(pattern_words))

x_train=[]
y_train=[]
for (pattern_tag, patterne) in X_Y:
    vect=nlp_h.sentence_vector(patterne, pattern_words)
    x_train.append(vect)
    y_train.append(tags.index(pattern_tag))

x_train=np.array(x_train)
y_train=np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples
#hyper_parametres
batch_size=8
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ChatNeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')
