import torch.nn as nn 
import torch.nn.functional as F 

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.vocab_size = vocab_size 
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim)
        
        self.linear = nn.Linear(
            self.embedding_dim, self.vocab_size)

        
    def forward(self, contexts):

        # Get the embeddings for each sequence of 
        # context vectors
        embedding = self.embedding(contexts)

        # Get the average embedding for each context
        avg_embedding = embedding.mean(dim = 1)

        # Apply the linear weights
        outputs = self.linear(avg_embedding)

        return outputs 

    def get_embedding_weights(self):
        return self.embedding.weight.detach().numpy()