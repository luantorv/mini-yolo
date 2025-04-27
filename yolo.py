# Simulación MUY simplificada de un YOLO mini en PyTorch
print("Inicio de ejecución")

import torch
import torch.nn as nn

class MiniYOLO(nn.Module):
    def __init__(self):
        super(MiniYOLO, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*64*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7*7*5)  # Por ejemplo: 7x7 cuadrícula, 5 predicciones por celda
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, 7, 7, 5)

modelo = MiniYOLO()
imagen_dummy = torch.randn(1, 3, 256, 256)  # Imagen falsa de entrada
salida = modelo(imagen_dummy)
print(f"Forma de la salida: {salida.shape}")  # Debería ser (1, 7, 7, 5)
