#pip install fastapi uvicorn

#1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel
import torch
from InputSentences import InputSentences
from InputSentences import MyResponse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import pandas as pd

model_name = 'sentence-transformers/bert-base-nli-mean-tokens'

#2. Create the app object
app = FastAPI()
pickle_in = open("Cosine_Similarity.pkl", "rb")
Similarity = pickle.load(pickle_in)

#3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Please change the url to http://127.0.0.1:8000/docs'}

#4. Route with a single parameter, returns the parameter within a message
#       Located at: http://127.0.0.1:8000/name
@app.post('/similarityscore')
def similarity_score(data: InputSentences):
    data = data.model_dump()
    sentence1 = data['sentence1']
    sentence2 = data['sentence2']
    sentences = []
    # sentences = [item for sublist in zip(sentence1, sentence2) for item in sublist]
    sentences.append(sentence1)
    sentences.append(sentence2)
    print(sentences)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    tokens = {'input_ids': [], 'attention_mask': []}
    
    for sentence in sentences:
     new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                       truncation=True, padding='max_length', return_tensors='pt')
     tokens['input_ids'].append(new_tokens['input_ids'][0])
     tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    outputs = model(**tokens)
    
    print(outputs)
    
    embeddings = outputs.last_hidden_state
    
    attention = tokens['attention_mask']
    
    mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
    
    mask_embeddings = embeddings * mask
    
    summed = torch.sum(mask_embeddings, 1)

    counts = torch.clamp(mask.sum(1), min=1e-9)
    
    mean_pooled = summed/counts
    print(mean_pooled)
    
    mean_pooled = mean_pooled.detach().numpy()
    #for i in range(0, len(mean_pooled)-1, 2):
    similarity = cosine_similarity([mean_pooled[0]], [mean_pooled[1]])
    print(f"Cosine Similarity between text1 and text2 : {similarity}")
    
    # rounded_similarity = np.round(similarity, decimals=2).tolist()
    
    return MyResponse(result=similarity)
    

#5. Run the API with uvicorn
#   Will run on http://127.0.0.1:8000

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    #uvicorn app:app --reload
