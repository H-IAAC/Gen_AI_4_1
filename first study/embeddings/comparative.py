from gensim.models import Word2Vec




import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import phate
import trimap
import pacmap
from standartized_balanced import StandardizedBalancedDataset

# Function to load data using StandardizedBalancedDataset
def get_data(dataset_name, sensors, normalize_data):    
    data_folder = f"/HDD/dados/amparo/meta4/M4-Framework-Experiments/experiments/experiment_executor/data/standartized_balanced/{dataset_name}/"
    dataset = StandardizedBalancedDataset(data_folder, sensors=sensors)
    X_train, y_train, X_test, y_test, X_val, y_val = dataset.get_all_data(normalize_data=normalize_data, resize_data=False)
    return X_train, y_train, X_test, y_test, X_val, y_val

# Load the dataset (UCI in this example)
X_train, y_train, X_test, y_test, X_val, y_val = get_data("UCI", ['accel', 'gyro'], False)
input_shape = X_train[0].shape
print(f"Input shape: {input_shape}")



# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))  # Reshape if needed



X_train_tokens = [[f"sensor_{i}_{val}" for i, val in enumerate(x)] for x in X_scaled]

# Training Word2Vec model
w2v_model = Word2Vec(sentences=X_train_tokens, vector_size=2, window=5, min_count=1, sg=1)


# Generating embeddings for each sequence
X_w2v = np.array([w2v_model.wv[seq].mean(axis=0) for seq in X_train_tokens])
print(X_w2v.shape)
# Visualize Word2Vec result
plt.figure(figsize=(8, 6))
plt.scatter(X_w2v[:, 0], X_w2v[:, 1], c=y_train, cmap='viridis')
plt.title("Word2Vec")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.colorbar()
plt.savefig("word2vec.jpg")
plt.show()


# Apply PHATE
phate_operator = phate.PHATE(n_components=2)
X_phate = phate_operator.fit_transform(X_scaled)

# Visualize PHATE result
plt.figure(figsize=(8, 6))
plt.scatter(X_phate[:, 0], X_phate[:, 1], c=y_train, cmap='viridis')
plt.title("PHATE")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.colorbar()
plt.savefig("phate.jpg")
plt.show()

# Apply TRIMAP
X_trimap = trimap.TRIMAP(n_dims=2).fit_transform(X_scaled)

# Visualize TRIMAP result
plt.figure(figsize=(8, 6))
plt.scatter(X_trimap[:, 0], X_trimap[:, 1], c=y_train, cmap='viridis')
plt.title("TRIMAP")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.colorbar()
plt.savefig("trimap.jpg")
plt.show()

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

# Visualize PacMAP result
plt.figure(figsize=(8, 6))
plt.scatter(X_pacmap[:, 0], X_pacmap[:, 1], c=y_train, cmap='viridis')
plt.title("PacMAP")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.colorbar()
plt.savefig("pacmap.jpg")
plt.show()



