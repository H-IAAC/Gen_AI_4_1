import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from standartized_balanced import StandardizedBalancedDataset

# Função para carregar os dados
def get_data(dataset_name, sensors, normalize_data):    
    data_folder = f"/HDD/dados/amparo/meta4/M4-Framework-Experiments/experiments/experiment_executor/data/standartized_balanced/{dataset_name}/"
    dataset = StandardizedBalancedDataset(data_folder, sensors=sensors)
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_all_data(normalize_data=normalize_data, resize_data=True)
    return X_train, y_train, X_test, y_test, X_val, y_val

# Função para pré-processar os dados
def preprocess_data(X):
    X = np.array([x.astype(np.float32) for x in X])
    return X

# Função para criar um DataLoader
def create_dataloader(X, batch_size):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Definição do Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        _, hidden = self.rnn(x)
        z = self.fc(hidden[-1])
        return z

# Definição do Decoder
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, output_dim, batch_first=True)

    def forward(self, z):
        x = self.fc(z)
        x = x.unsqueeze(1)  # Adicione uma dimensão de sequência
        x, _ = self.rnn(x)
        return x

# Definição do Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# Função para treinar o Autoencoder
def train_autoencoder(autoencoder, dataloader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for batch_data in dataloader:
            batch_data = batch_data[0]  # Ajustar conforme necessário

            optimizer.zero_grad()
            reconstructed = autoencoder(batch_data)
            loss = criterion(reconstructed, batch_data)  # MSE loss para reconstrução
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Função para reduzir a dimensionalidade
def reduce_dimensionality(encoder, data):
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32)
        z = encoder(data_tensor)
    return z.numpy()

# Função para plotar e salvar gráficos
def plot_and_save_embeddings(embeddings, labels, save_path):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title('Embeddings Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(save_path)
    plt.show()

# Carregar e preparar os dados
X_train, y_train, X_test, y_test, X_val, y_val = get_data("MotionSense", ['accel', 'gyro'], False)
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

batch_size = 64
train_loader = create_dataloader(X_train, batch_size)
test_loader = create_dataloader(X_test, batch_size)

# Inicializar e treinar o Autoencoder
z_dim = 6
autoencoder = Autoencoder(input_dim=6, hidden_dim=128, z_dim=z_dim)
train_autoencoder(autoencoder, train_loader, epochs=500)

# Reduzir a dimensionalidade dos dados de teste
X_test_reduced = reduce_dimensionality(autoencoder.encoder, X_test)

# Plotar e salvar as embeddings reduzidas
plot_and_save_embeddings(X_test_reduced, y_test, 'embeddings_reduced.png')
