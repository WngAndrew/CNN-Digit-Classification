# CNN-Digit-Classification
After watching Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=435s) video I became curious about neural networks, I wanted to continue my exploration of deep learning via interactive learning. This project provided a simple way for me to learn how to write out simple neural network architectures in pytorch, while also giving me motivation to learn more about deep learning, specifically CNN's.

Before getting into any implementation, first I wanted to solidify my understanding of MLP's and also learn about the CNN architecture. [Here](https://docs.google.com/document/d/1QKHHfPdCDTgCw6Ko_slAYBEqdBgACYAhIxXS-kMe8M8/edit?usp=sharing) are my notes on MLP's, and [here](https://docs.google.com/document/d/1larZAdze_TzdZxS3wbQvTElkmNPsMFCemH9MevNQAuI/edit?usp=sharing) are my notes from learning how CNN's work. Now that I have a basic understanding of the theory behind the necessary architecture, let's get to implementing the model.

# Data setup and exploration
First let's import all the necessary libraries, load the datasets, and gain an understanding of what the data looks like
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from google.colab import drive
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#load the datasets
training_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

testing_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

#analyzing / understanding datasets
print(training_data.data.shape)
print(testing_data.data.shape)

image, label = training_data[0]
print(image.shape, label)
```

Looks like we have 60,000 samples for training and 10,000 for testing, and that our inputs will be of dimension 28x28x1

Now we can create data loaders for the two datasets

```python
training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

#learning how DataLoaders work
data_iter = iter(training_dataloader)

while True:
    try:
      image, label = next(data_iter)
      print(image.shape, label)
    except StopIteration:
      break
```

# Writing out the ConvNet
```python
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3) #output shape = 26x26x32
    self.pool1 = nn.MaxPool2d(2, 2) #output shape = 13x13x32
    self.conv2= nn.Conv2d(32, 64, 3) #output shape = 11x11x64
    self.pool2 = nn.MaxPool2d(2, 2) #output shape = 5x5x64
    self.fc1 = nn.Linear(5*5*64, 4608) #first fc layer
    self.fc2 = nn.Linear(4608, 10) #output layer to classify 10 digits

  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = x.view(-1, 5*5*64)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
```

# Training setup
```python
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

def train(epoch):
    model.train()
    running_loss = 0.0

    for (images, labels) in training_dataloader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()

      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(training_dataloader)}")

def validate(epoch):
  model.eval()
  val_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in testing_dataloader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)
      val_loss += loss.item()

      _, predicted = torch.max(outputs.data, 1) #fetch predicted value
      total += labels.size(0) #track total number of samples
      correct += (predicted == labels).sum().item() #track correct predictions

    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(testing_dataloader)}, Accuracy: {accuracy}%")
```

# Trainig loop and results
```python
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(epoch)
    validate(epoch)

print("Training complete")
```
| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|-------|---------------|-----------------|---------------------|
| 1/10  | 0.9089181232951216 | 0.31046995083997203 | 90.69% |
| 2/10  | 0.2421455085992432 | 0.21130713137092105 | 93.34% |
| 3/10  | 0.15945094818277145 | 0.12179995527502838 | 96.35% |
| 4/10  | 0.11895404360406021 | 0.08938162365726604 | 97.38% |
| 5/10  | 0.09728459076983716 | 0.0817339878031023 | 97.60% |
| 6/10  | 0.08330662681190952 | 0.0679879461373952 | 97.92% |
| 7/10  | 0.07396662203438167 | 0.06419518417055914 | 97.88% |

It seemed like the model was starting to overfit, and the accuracy was satisfactory, so I stopped the training loop on the 8th epoch.

Now let's see the model work real time with a GUI

<img src="https://github.com/WngAndrew/CNN-Digit-Classification/assets/108479242/58ed2189-93a3-410d-992b-78a9072aedba" alt="Digit Classifier GUI" width="400"/>

# Conclusion
It was a fun process working on this mini project and it provided a more intuitvely structured approach to deep learning, CNN's, and pytorch basics. 

Below are some of the resources I found helpful:
- [Understanding standard back prop](https://xnought.github.io/backprop-explainer/)
- [Andrej Karpathy's lecture on micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=435s)
- [Video on CNN backprop](https://www.youtube.com/watch?v=z9hJzduHToc&list=LL&index=1&t=3s&ab_channel=far1din)
- [Convolution Animation](https://www.youtube.com/watch?v=w4kNHKcBGzA&ab_channel=AnimatedAI)
- [Stanford's computer vision course website](https://cs231n.stanford.edu/)
- [CMU lecture on backprop in CNN's](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf)

