import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)

y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Atualizando para aceitar apenas 1 valor de entrada, pois agora temos apenas a distância
        self.fc1 = nn.Linear(1, 5) # De 2 para 1 na entrada
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Net()

criterion = nn.MSELoss()
# TODO 4:15 https://on.fiap.com.br/mod/conteudoshtml/view.php?id=536601&c=14529&sesskey=egZAAHCu4h