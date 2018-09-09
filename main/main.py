import models.cbow as cbow 
import trainers.training as training
import data.dataset as d

import sklearn.manifold as manifold
import matplotlib.pyplot as  plt 

def main():
    # Grab the dataset
    print('Fetching the Dataset')
    dataset = d.FederalistPapers()
    vocab_size = len(dataset.word2id)  

    # Build the model
    print('Initializing the model')
    model = cbow.CBOW(vocab_size = vocab_size, embedding_dim = 50)

    # Train the model
    print('Training')
    train = training.train
    losses = train(model=model, dataset=dataset, num_epochs=500, batch_size=32)

    plt.plot(losses)
    plt.show()

    X = model.get_embedding_weights()

    states = dataset.word2id['states']
    constitution = dataset.word2id['constitution']
    Y = manifold.TSNE(n_components=2).fit_transform(X)
 
    #Y = Y[:100]
    #a = [x for x, y in Y]
    #b = [y for x, y in Y]

    #fig, ax = plt.subplots()
    #ax.scatter(a, b)
    #for i in range(len(Y)):
    #    ax.annotate(dataset.id2word[i], (a[i], b[i]))
    #plt.show()
if __name__ == '__main__':
    main()