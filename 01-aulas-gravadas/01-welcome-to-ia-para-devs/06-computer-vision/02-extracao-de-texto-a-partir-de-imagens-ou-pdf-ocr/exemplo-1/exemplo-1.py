import os
import cv2
import matplotlib.pyplot as plt

# Carregar a imagem (usando caminho relativo ao script)
script_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(script_dir, 'imagem-1.jpeg'))

# Exibir imagem
cv2.imshow('Imagem', img)