import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definir a Rede GMMN com Regularização
class GMMN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GMMN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Adicionando dropout

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Função de Perda Alternativa: Distância de Wasserstein (não implementada aqui)
# Você pode buscar implementações de distâncias Wasserstein no PyTorch ou outras métricas.

# Função de Discrepância de Máxima Média (MMD) com sigma dinâmico
def mmd_rbf(X, Y, sigma=1.0):
    XX = torch.matmul(X, X.t())
    YY = torch.matmul(Y, Y.t())
    XY = torch.matmul(X, Y.t())
    
    rx = torch.diag(XX).unsqueeze(0).expand_as(XX)
    ry = torch.diag(YY).unsqueeze(0).expand_as(YY)
    
    K = torch.exp(- (rx.t() + rx - 2*XX) / (2*sigma**2))
    L = torch.exp(- (ry.t() + ry - 2*YY) / (2*sigma**2))
    P = torch.exp(- (rx.t() + ry - 2*XY) / (2*sigma**2))
    
    beta = 1. / (X.size(0) * X.size(0))
    gamma = 1. / (X.size(0) * Y.size(0))
    
    return beta * (K.sum() + L.sum()) - 2 * gamma * P.sum()

# Dados de exemplo (amostras reais e amostras latentes)
np.random.seed(42)
data_real = np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], 1000).astype(np.float32)
latent_data = np.random.normal(0, 1, (1000, 2)).astype(np.float32)

# Converting to PyTorch Tensors
data_real = torch.tensor(data_real)
latent_data = torch.tensor(latent_data)

# Criar o GMMN
input_dim = latent_data.shape[1]
hidden_dim = 256  # Aumentando ainda mais a complexidade
output_dim = data_real.shape[1]

gmmn = GMMN(input_dim, hidden_dim, output_dim)

# Otimizador com Regularização L2
optimizer = optim.Adam(gmmn.parameters(), lr=0.001, weight_decay=1e-5)

# Scheduler para Redução Gradual da Taxa de Aprendizado
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

# Treinamento do GMMN
epochs = 5000
sigma = 1.0  # Ajuste inicial de sigma
for epoch in range(epochs):
    gmmn.train()
    optimizer.zero_grad()

    # Introduzir ruído nas amostras latentes para evitar colapso de modos
    latent_data_noisy = latent_data + 0.1 * torch.randn_like(latent_data)

    generated_data = gmmn(latent_data_noisy)
    loss = mmd_rbf(generated_data, data_real, sigma=sigma)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 500 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Sigma: {sigma:.4f}')
        # Ajuste dinâmico mais agressivo de sigma
        if loss.item() < 1e-5:
            sigma = min(sigma * 0.9, 10.0)  # Redução mais moderada e valor mínimo para sigma
        elif loss.item() > 1e-2:
            sigma = min(sigma * 1.1, 10.0)  # Aumento mais moderado e valor máximo para sigma

# Verificar as amostras geradas
gmmn.eval()
generated_data = gmmn(latent_data).detach().numpy()

plt.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.5, label='Generated Data')
plt.scatter(data_real[:, 0], data_real[:, 1], alpha=0.5, label='Real Data')
plt.title('Amostras Geradas pelo GMMN vs. Amostras Reais')
plt.legend()
plt.save()
