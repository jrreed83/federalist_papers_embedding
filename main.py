import models
import trainers
import data

def main():
    # Grab the dataset
    dataset = data.federalist_papers.FederalistPapers()
    vocab_size = len(dataset.word2id) 

    # Build the model
    model = models.cbow.CBOW(vocab_size = vocab_size, embedding_dim = 50)

    # Train the model
    train = trainers.training.train
    train(model=model, dataset=dataset, num_epochs=500, batch_size=32)

if __name__ == '__main__':
    main()