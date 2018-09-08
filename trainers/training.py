import torch.optim as optim
import torch.nn as nn 
import torch.utils.data as data
import sklearn.manifold as manifold

def train(model, dataset, num_epochs = 100, batch_size = 32, lr = 1e-2):
    # Data loader
    loader = data.DataLoader(dataset, batch_size = batch_size)

    # Loss function being minimized
    loss_fn = nn.CrossEntropyLoss()

    # Optimization algorithm used to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in loader:

            # Reshape the targets so that it's a 1D tensor, i.e.
            # a vector
            targets = targets.view(-1)

            # Reinitialize the gradients
            optimizer.zero_grad()

            preds = model(inputs)

            loss = loss_fn(preds, targets)

            total_loss += loss.item()
            # Back propagation algorithm
            loss.backward() 

            # Take a step in the weights/ biases
            optimizer.step()
            break
        print(f'Epoch: {epoch} Loss: {total_loss}')
        losses.append(total_loss)


    return losses





