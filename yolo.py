# Simulación MUY simplificada de un YOLO mini en PyTorch

import torch
import torch.nn as nn

# Definimos una clase que representa nuestra red neuronal YOLO simplificada
# Hereda las características de nn.Module por lo que podrá utilizar sus métodos. 
class YOLOPequeno(nn.Module):
    def __init__(self):
        # super() llama al constructor de nn.Module para habilitar el funcionamiento interno de PyTorch.
        super(YOLOPequeno, self).__init__()
        
        # Primer bloque convolucional: extrae características simples (bordes, texturas)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # self.conv1 = nn.Conv2d(...) -> Crea una capa convolucional con:
        #       in_channels=3 -> entrada RGB (3 canales)
        #       out_channels=26 -> producirá 16 capas de activación (más profundidad)
        #       kernel_size=3 -> usa un filtro de 3x3 píxeles
        #       padding=1 -> mantiene un tamaño espacial 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        # MaxPool2d: reduce resolución dividiendo la imagen en bloques de 2x2 y eligiendo el valor máximo
        # (224x224 → 112x112)
        
        # Esto se repite en el resto de capas, bajando el tamaño espacial con MaxPool (hasta que coincida con el grid YOLO)
        # pero aumentando la cantidad de canales de salida (32, 64, 128)


        # Segundo bloque convolucional: extrae patrones más complejos
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112x112 → 56x56


        # Tercer bloque: seguimos profundizando la extracción de características
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56x56 → 28x28


        # Cuarto bloque
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(4, 4)  # 28x28 → 7x7 (nuestro "grid" de YOLO)


        # Capa completamente conectada: 
        # Linear(in_features, out_features) -> transforma la salida de la red convolucional en predicciones
        self.fc = nn.Linear(128 * 7 * 7, 7 * 7 * 5)  
        # entrada -> 128 * 7 *7 (salida de la última convolución aplanada)
        # salida -> 7 * 7 * 5  (x, y, w, h, C)

    # Toma un tensor de entrada x (una imagen) y devuelve la predicción.
    def forward(self, x):
        # Paso hacia adelante por las capas convolucionales y de pooling
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        # Aplica convolución → activa con ReLU → reduce con MaxPool.
        # torch.reulu() -> función de activación que introduce no linealidad y filtra valores negativos.


        # Aplanamos la salida para conectarla a la capa densa
        x = x.view(x.size(0), -1)  
        # convierte de [batch, canales, alto, ancho] → [batch, características planas]
        # Este paso es necesario antes de pasar a la capa densa (Linear), que espera un vector.


        # Pasamos por la capa lineal
        x = self.fc(x)

        # Reestructuramos el vector a forma [batch, 7, 7, 5]
        x = x.view(-1, 7, 7, 5)

        return x

# Si se ejecuta directamente, se hace una prueba rápida
if __name__ == "__main__":
    modelo = YOLOPequeno()  # se instancia el modelo
    entrada = torch.randn(1, 3, 224, 224)  # imagen de prueba, batch size = 1
    salida = modelo(entrada)  # paso la imagen de prueba al modelo y guardo la salida
    print("Forma de salida:", salida.shape)  
    # Muestra la salida del modelo
    # debería ser [1, 7, 7, 5]
