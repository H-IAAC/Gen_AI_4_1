import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import phate
import trimap
import pacmap
from sklearn.decomposition import TruncatedSVD
from umap import UMAP
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from standartized_balanced import StandardizedBalancedDataset

# Function to load data using StandardizedBalancedDataset
def get_data(dataset_name, sensors, normalize_data):    
    data_folder = f"/HDD/dados/amparo/meta4/M4-Framework-Experiments/experiments/experiment_executor/data/standartized_balanced/{dataset_name}/"
    dataset = StandardizedBalancedDataset(data_folder, sensors=sensors)
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_all_data(normalize_data=normalize_data, resize_data=False)
    return X_train, y_train, X_test, y_test, X_val, y_val

dataset_n="MotionSense"
X_train, y_train, X_test, y_test, X_val, y_val = get_data(dataset_n, ['accel', 'gyro'], False)
input_shape = X_train[0].shape
print(f"Input shape: {input_shape}")

def apply_fft(X):
    # Aplicar FFT em cada amostra e retornar o módulo da FFT (magnitude)
    return np.abs(np.fft.fft(X, axis=-1))

X_train = apply_fft(X_train)
X_test = apply_fft(X_test)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))  # Reshape if needed

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Create 'plots' directory if it does not exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Define the Autoencoder using PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_encoded(self, x):
        return self.encoder(x)

# Build and train Autoencoder
input_dim = X_tensor.shape[1]
encoding_dim = 2  # Dimension to reduce to
autoencoder = Autoencoder(input_dim, encoding_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Convert data to DataLoader for batching
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Training loop
epochs = 50
autoencoder.train()
for epoch in range(epochs):
    for data in dataloader:
        inputs, _ = data
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Apply the encoder to reduce dimensionality
autoencoder.eval()
with torch.no_grad():
    X_autoencoder = autoencoder.get_encoded(X_tensor).numpy()

# Apply PHATE
phate_operator = phate.PHATE(n_components=2)
X_phate = phate_operator.fit_transform(X_scaled)

# Apply TRIMAP
X_trimap = trimap.TRIMAP(n_dims=2).fit_transform(X_scaled)

# PacMAP embedding class
class PacmapEmbedding:
    def __init__(self, n_components=2, n_neighbors=10):
        self.embedding_projector = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors)
        self.fitted = False

    def fit(self, data):
        emb = self.embedding_projector.fit_transform(data)
        self.fitted = True
        self.data_or = data
        return emb

    def transform(self, data):
        if not self.fitted:
            raise ValueError("Você precisa chamar o método fit antes de chamar o método transform")
        return self.embedding_projector.transform(data, self.data_or)

    def get_embedding(self):
        if not self.fitted:
            raise ValueError("Você precisa chamar o método fit antes de chamar o método get_embedding")
        return self.embedding_projector.transform(self.data_or)

# Apply PacMAP
pacmap_embedding = PacmapEmbedding(n_components=2, n_neighbors=10)
X_pacmap = pacmap_embedding.fit(X_scaled)

# Apply UMAP
umap_operator = UMAP(n_components=2)
X_umap = umap_operator.fit_transform(X_scaled)

# Apply TruncatedSVD
svd_operator = TruncatedSVD(n_components=2)
X_svd = svd_operator.fit_transform(X_scaled)

# Word2Vec Embedding
# Treating sensor data as sequences of "words"
# Each time series is flattened and converted to a sequence of strings
X_train_sequences = [list(map(str, x.flatten())) for x in X_train]

# Training Word2Vec model
w2v_model = Word2Vec(sentences=X_train_sequences, vector_size=2, window=5, min_count=1, sg=1)

# Generating embeddings for each sequence
X_w2v = np.array([w2v_model.wv[seq].mean(axis=0) for seq in X_train_sequences])

# Create subplots
fig = make_subplots(rows=4, cols=2, subplot_titles=("PHATE", "TRIMAP", "PacMAP", "UMAP", "TruncatedSVD", "Word2Vec", "Autoencoder"))

# Add each scatter plot to the subplot
# Adicionar cada gráfico de dispersão ao subplot
fig.add_trace(go.Scatter(x=X_phate[:, 0], y=X_phate[:, 1], mode='markers', marker=dict(color=y_train, colorscale='viridis')), row=1, col=1)
fig.add_trace(go.Scatter(x=X_trimap[:, 0], y=X_trimap[:, 1], mode='markers', marker=dict(color=y_train, colorscale='viridis')), row=1, col=2)
fig.add_trace(go.Scatter(x=X_pacmap[:, 0], y=X_pacmap[:, 1], mode='markers', marker=dict(color=y_train, colorscale='viridis')), row=2, col=1)
fig.add_trace(go.Scatter(x=X_umap[:, 0], y=X_umap[:, 1], mode='markers', marker=dict(color=y_train, colorscale='viridis')), row=2, col=2)
fig.add_trace(go.Scatter(x=X_svd[:, 0], y=X_svd[:, 1], mode='markers', marker=dict(color=y_train, colorscale='viridis')), row=3, col=1)
fig.add_trace(go.Scatter(x=X_w2v[:, 0], y=X_w2v[:, 1], mode='markers', marker=dict(color=y_train, colorscale='viridis')), row=3, col=2)
fig.add_trace(go.Scatter(x=X_autoencoder[:, 0], y=X_autoencoder[:, 1], mode='markers', marker=dict(color=y_train, colorscale='viridis')), row=4, col=1)

# Update layout
fig.update_layout(height=1200, width=900, title_text=f"{dataset_n} Comparative Visualization of Dimensionality Reduction Techniques Including Autoencoder (PyTorch)")

# Save the combined plot as an image
fig.write_html(f"plots/{dataset_n}_all_embeddings_with_autoencoder_pytorch_plotly.html")
fig.write_image(f"plots/{dataset_n}_all_embeddings_with_autoencoder_pytorch.jpg")
