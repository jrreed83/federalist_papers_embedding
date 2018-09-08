import models.cbow as cbow 
import trainers.training as training
import data.federalist_papers as fed

import matplotlib.pyplot as  plt 

def main():
    # Grab the dataset
    print('Building the Dataset')
    dataset = fed.FederalistPapers()
    vocab_size = len(dataset.word2id) 

    # Build the model
    print('Initializing the model')
    model = cbow.CBOW(vocab_size = vocab_size, embedding_dim = 50)

    # Train the model
    print('Training')
    train = training.train
    losses = train(model=model, dataset=dataset, num_epochs=500, batch_size=512)

    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    main()