# Code reference: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
import torch
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def train_step(model, loss_fn, optimizer, train_loader):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print('  batch {0} loss: {1:0.4f}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


def train(hypes, model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    flag: bool = (hypes is not None)
    epoch_number = 0
    patience = 0  # For early-stopping
    best_val_loss = 1_000_000.
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if flag is True:
        writer = SummaryWriter(comment=' '.join(str(h) for h in hypes))
    else:
        writer = SummaryWriter()

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        if patience > 1:
            print("Early stopping at step: %d" % (epoch_number + 1))
            break

        model.train(True)
        avg_loss = train_step(model, loss_fn, optimizer, train_loader)
        scheduler.step()

        running_val_loss: float = 0.0
        running_acc: int = 0
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, val_data in enumerate(test_loader):
                val_inputs, val_labels = val_data
                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_labels)
                running_val_loss += val_loss
                running_acc += torch.sum(val_data == val_outputs)

        avg_val_loss = running_val_loss / (i + 1)
        avg_acc = running_acc / (i + 1)
        print('LOSS train {0:0.4f} test {1:0.4f}'.format(avg_loss, avg_val_loss))
        print('Test accuracy: %d' % avg_acc)
        writer.add_scalars('Train vs val loss', {'Train': avg_loss, 'Val': avg_val_loss}, epoch_number + 1)
        writer.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = './models/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
            patience = 0

        else:
            patience += 1

        epoch_number += 1

    end_time = time.time()
    print("Elapsed time:", end_time - start_time)
