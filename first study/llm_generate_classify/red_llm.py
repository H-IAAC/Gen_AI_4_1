from standartized_balanced import StandardizedBalancedDataset




def get_data(dataset_name,sensors,normalize_data):    
    working_directory=f"result/{dataset_name}/"
    
    dataset = StandardizedBalancedDataset(data_folder, sensors=sensors)
    X_train, y_train,X_test, y_test,X_val, y_val = dataset.get_all_data(normalize_data=normalize_data, resize_data=False)
    return X_train, y_train,X_test, y_test,X_val, y_val

dataset_name="UCI"
# Carregue os dados
data_folder = f"/HDD/dados/amparo/meta4/M4-Framework-Experiments/experiments/experiment_executor/data/standartized_balanced/{dataset_name}/"
X_train, y_train, X_test, y_test, X_val, y_val = get_data("UCI", ['accel', 'gyro'], False)
#print(len(X_train[0]))
print(X_train.shape)
import numpy as np
fft_result = np.fft.fft(X_train, axis=1)
X_real = np.real(fft_result).astype(np.float32)
print(fft_result.shape)
from  utlis_llm import PacmapEmbedding
pacmapEmbedding= PacmapEmbedding(n_components=2)
data_emb=pacmapEmbedding.fit(X_real)
#print(data_emb)
#data_t=pacmapEmbedding.transform(X_test)

from collections import defaultdict
embeddings_l = defaultdict(list)
for vector, label in zip(data_emb, y_train):
    
    embeddings_l['activity_'+str(label)].append(list(vector))

#print(embeddings_l)
text_1=f'The following given embeddings correspond to "walking upstairs": {embeddings_l["activity_1"]}'
text_2=f'The following given embeddings correspond to "“jogging”": {embeddings_l["activity_2"]}'

#text =f'{text_1} and {text_2}classify the embedding pergunta  as either "walking upstairs" or "jogging" considering the minimum distance to the example embeddings provided that the distance metric chosen is euclidean distance. Answer in one word.'
text =f'use {text_1} and {text_2} to create a new embedding for the “jogging” return a vector same size of “jogging” your answer is a vector similar to [2,3]'

import ollama

answers=[]
for a in range(0,1):
    pergunta=embeddings_l['activity_1'][a]
    #print(pergunta)
    response = ollama.chat(model='llama3', messages=[
    {
        'role': 'user',
        'content':text,
    },
    ])
    answers.append(response['message']['content'])

print(answers)
from collections import Counter
counter = Counter(answers)
print(counter)
# Exibir a contagem de cada string
#print(counter)