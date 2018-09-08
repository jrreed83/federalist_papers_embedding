from models import cbow
from trainers import training
from data import federalist_papers

def main():
    # Grab the dataset
    print('Building the Dataset')
    dataset = federalist_papers.FederalistPapers()
    vocab_size = len(dataset.word2id) 

    # Build the model
    print('Initializing the model')
    model = cbow.CBOW(vocab_size = vocab_size, embedding_dim = 50)

    # Train the model
    print('Training')
    train = training.train
    train(model=model, dataset=dataset, num_epochs=500, batch_size=32)

if __name__ == '__main__':
    main()