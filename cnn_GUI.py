import tkinter as tk
from tkinter import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageGrab

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Digit Classifier")
        self.geometry("400x400")

        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()

        self.predict_button = tk.Button(self, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.clear_button.pack()

        self.label = tk.Label(self, text="Draw a digit and press 'Predict'")
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.model = self.load_model()

    def load_model(self):
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3)
                self.pool1 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.pool2 = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(5*5*64, 4608)
                self.fc2 = nn.Linear(4608, 10)

            def forward(self, x):
                x = self.pool1(F.relu(self.conv1(x)))
                x = self.pool2(F.relu(self.conv2(x)))
                x = x.view(-1, 5*5*64)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = CNN()
        model_load_path = '\\Users\\andre\\Downloads\\cnn_digit_model.pth'  # Update this path
        model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')

    def clear(self):
        self.canvas.delete("all")

    def preprocess_image(self):
        self.canvas.update()
        self.canvas.postscript(file='digit.eps', colormode='color')

        with Image.open('digit.eps') as img:
            img = img.convert('L')
            img = ImageOps.invert(img)
            img = img.resize((28, 28))
            img = np.array(img)
            img = img / 255.0  # Ensure normalization matches training
            img = img.reshape(1, 1, 28, 28)
            return torch.tensor(img, dtype=torch.float32)


    def predict(self):
        img = self.preprocess_image()
        output = self.model(img)
        _, predicted = torch.max(output.data, 1)
        self.label.config(text=f"Predicted digit: {predicted.item()}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
