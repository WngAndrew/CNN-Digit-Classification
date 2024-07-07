# CNN-Digit-Classification
After watching Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=435s) video I became curious about neural networks, I wanted to continue my exploration of deep learning via interactive learning. This project provided a simple way for me to learn how to write out simple neural network architectures in pytorch, while also giving me motivation to learn more about deep learning, specifically CNN's.

Before getting into any implementation, first I wanted to solidify my understanding of MLP's and also learn about the CNN architecture. [Here](https://docs.google.com/document/d/1QKHHfPdCDTgCw6Ko_slAYBEqdBgACYAhIxXS-kMe8M8/edit?usp=sharing) are my notes on MLP's, and [here](https://docs.google.com/document/d/1larZAdze_TzdZxS3wbQvTElkmNPsMFCemH9MevNQAuI/edit?usp=sharing) are my notes from learning how CNN's work. Now that I have a basic understanding of the theory behind the necessary architecture, let's get to implementing the model.

# Data setup and exploration
First let's import all the necessary libraries
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

