import torch
import copy
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd


def train(model, criterion, optimizer, train_data_loader, max_epochs):
    model.train()

    for epoch in range(0, max_epochs):
        train_loss = 0

        for bg, target in train_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_data_loader.dataset)

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))


def train_emodel(model, criterion, optimizer, train_data_loader, max_epochs, valid_data_loader):

    train_losses = [] # Train loss 저장
    valid_losses = [] # Vaild loss 저장

    for epoch in range(0, max_epochs):
        model.train()
        train_loss = 0

        for bg, self_feat, target in train_data_loader:
            pred = model(bg, self_feat)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_data_loader.dataset)
        train_losses.append(train_loss)  # Save loss for this epoch

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))

        model.train()
        valid_loss = 0

        with torch.no_grad():
            for bg, self_feat, target in valid_data_loader:
                pred = model(bg, self_feat)
                loss = criterion(pred, target)
                valid_loss += loss.detach().item()

            valid_loss /= len(valid_data_loader.dataset)
            valid_losses.append(valid_loss)  # Save loss for this epoch

            print('Epoch {}, train loss {:.4f}'.format(epoch + 1, valid_loss))

    return train_losses, valid_losses  # Epoch별 train loss, valid loss 반환 


def test(model, criterion, test_data_loader, accs=None):
    preds = None
    model.eval()

    with torch.no_grad():
        test_loss = 0
        correct = 0

        for bg, target in test_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            test_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)

            if accs is not None:
                correct += torch.eq(torch.max(pred, dim=1)[1], target).sum().item()

        test_loss /= len(test_data_loader.dataset)

        print('Test loss: ' + str(test_loss))

    if accs is not None:
        accs.append(correct / len(test_data_loader.dataset) * 100)
        print('Test accuracy: ' + str((correct / len(test_data_loader.dataset) * 100)) + '%')
    
    return test_loss, preds


def test_emodel(model, criterion, test_data_loader, accs=None):
    preds = None
    model.eval()

    targets = None
    self_feats = None

    with torch.no_grad():
        test_loss = 0
        correct = 0

        for bg, self_feat, target in test_data_loader:
            pred = model(bg, self_feat)
            loss = criterion(pred, target)
            test_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
                targets = target.clone().detach()
                self_feats = self_feat.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)
                targets = torch.cat((targets, target), dim=0)
                self_feats = torch.cat((self_feats, self_feat), dim=0)

            if accs is not None:
                correct += torch.eq(torch.max(pred, dim=1)[1], target).sum().item()

        test_loss /= len(test_data_loader.dataset)

        print('Test loss: ' + str(test_loss))

    if accs is not None:
        accs.append(correct / len(test_data_loader.dataset) * 100)
        print('Test accuracy: ' + str((correct / len(test_data_loader.dataset) * 100)) + '%')

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    self_feats = self_feats.cpu().numpy()
    np.savetxt('result.csv', np.concatenate((targets, preds, self_feats), axis=1), delimiter=',')

    return test_loss, preds


def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, train, test, collate, accs=None):
    num_data_points = len(dataset)
    size_fold = int(len(dataset) / float(num_folds))
    folds = []
    models = []
    optimizers = []
    test_losses = []

    for k in range(0, num_folds - 1):
        folds.append(dataset[k * size_fold:(k + 1) * size_fold])

    folds.append(dataset[(num_folds - 1) * size_fold:num_data_points])

    for k in range(0, num_folds):
        models.append(copy.deepcopy(model))
        optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01))

    # Fold마다 손실 기록
    fold_train_losses = []  # 각 fold별 train loss
    fold_valid_losses = []  # 각 fold별 train loss

    
    for k in range(num_folds):
        print('--------------- fold {} ---------------'.format(k + 1))

        train_dataset = []
        test_dataset = folds[k]

        for i in range(0, num_folds):
            if i != k:
                train_dataset += folds[i]


        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


        # Train the model for this fold
        train_losses, valid_losses = train(models[k], criterion, optimizers[k], train_data_loader, max_epochs, test_data_loader)  # 한 번에 max_epochs 실행
        fold_train_losses.append(train_losses)
        fold_valid_losses.append(valid_losses)


        # Test the model for this fold
        test_loss, pred = test(models[k], criterion, test_data_loader, accs)
        test_losses.append(test_loss)

    # Plot fold별 loss
    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)
    for k in range(num_folds):
        epochs = list(range(1, max_epochs + 1))
        axes[k].plot(epochs, fold_train_losses[k], label="Train Loss")
        axes[k].plot(epochs, fold_valid_losses[k], label="Validation Loss")

        axes[k].set_xlabel("Epoch")
        axes[k].set_ylabel("Loss")
        axes[k].legend()
        axes[k].set_title(f"Fold {k+1}")

    plt.suptitle("Train and Validation Loss Across Folds")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # df 저장
    Loss_df = pd.DataFrame({'Fold': list(range(1, len(fold_train_losses) + 1)), 'Train Loss': fold_train_losses, 'Validation Loss': fold_valid_losses})
    Loss_df.to_csv('Loss.csv', index = False)

    if accs is None:
        return np.mean(test_losses)
    else:
        return np.mean(test_losses), np.mean(accs)