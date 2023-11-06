# Code reference: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
import torch
import time
from datetime import datetime


def train_step(model, loss_fn, optimizer, train_loader):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print('  batch {0} loss: {1:0.4f}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


def train(model, train_loader, test_loader, loss_fn, optimizer, epochs):
    epoch_number = 0
    best_val_loss = 1_000_000.
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_step(model, loss_fn, optimizer, train_loader)

        running_val_loss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, val_data in enumerate(test_loader):
                val_inputs, val_labels = val_data
                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_labels)
                running_val_loss += val_loss

        avg_val_loss = running_val_loss / (i + 1)
        print('LOSS train {0:0.4f} test {1:0.4f}'.format(avg_loss, avg_val_loss))

        # Track the best performance, and save the model's state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    end_time = time.time()
    print("Elapsed time:", end_time - start_time)
