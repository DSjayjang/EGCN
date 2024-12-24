import torch
import copy
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# def train(model, criterion, optimizer, train_data_loader, max_epochs):
#     model.train()


#     for epoch in range(0, max_epochs):
#         train_loss = 0

#         for bg, target in train_data_loader:
#             pred = model(bg)
#             loss = criterion(pred, target)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.detach().item()

#         train_loss /= len(train_data_loader.dataset)

#         print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))
    

def train(model, criterion, optimizer, train_data_loader, max_epochs):
    model.train()
    train_losses = []

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
        train_losses.append(train_loss)  # Append train loss for each epoch
        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))
    
    return train_losses  # Return epoch-wise train losses


def train_emodel(model, criterion, optimizer, train_data_loader, max_epochs):
    model.train()

    train_losses = [] # Train loss 저장

    for epoch in range(0, max_epochs):
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
    return train_losses  # Epoch별 train loss 반환 


# def test(model, criterion, test_data_loader, accs=None):
#     preds = None
#     model.eval()

#     with torch.no_grad():
#         test_loss = 0
#         correct = 0

#         for bg, target in test_data_loader:
#             pred = model(bg)
#             loss = criterion(pred, target)
#             test_loss += loss.detach().item()

#             if preds is None:
#                 preds = pred.clone().detach()
#             else:
#                 preds = torch.cat((preds, pred), dim=0)

#             if accs is not None:
#                 correct += torch.eq(torch.max(pred, dim=1)[1], target).sum().item()

#         test_loss /= len(test_data_loader.dataset)

#         print('Test loss: ' + str(test_loss))

    

#     if accs is not None:
#         accs.append(correct / len(test_data_loader.dataset) * 100)
#         print('Test accuracy: ' + str((correct / len(test_data_loader.dataset) * 100)) + '%')
    

def test(model, criterion, test_data_loader, accs=None):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for bg, target in test_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            test_loss += loss.detach().item()

        test_loss /= len(test_data_loader.dataset)
        print('Test loss: {:.4f}'.format(test_loss))

    return test_loss  # Return test loss


# def test_emodel(model, criterion, test_data_loader, accs=None):
#     preds = None
#     model.eval()

#     targets = None
#     self_feats = None

#     with torch.no_grad():
#         test_loss = 0
#         correct = 0

#         for bg, self_feat, target in test_data_loader:
#             pred = model(bg, self_feat)
#             loss = criterion(pred, target)
#             test_loss += loss.detach().item()

#             if preds is None:
#                 preds = pred.clone().detach()
#                 targets = target.clone().detach()
#                 self_feats = self_feat.clone().detach()
#             else:
#                 preds = torch.cat((preds, pred), dim=0)
#                 targets = torch.cat((targets, target), dim=0)
#                 self_feats = torch.cat((self_feats, self_feat), dim=0)

#             if accs is not None:
#                 correct += torch.eq(torch.max(pred, dim=1)[1], target).sum().item()

#         test_loss /= len(test_data_loader.dataset)

#         print('Test loss: ' + str(test_loss))

#     if accs is not None:
#         accs.append(correct / len(test_data_loader.dataset) * 100)
#         print('Test accuracy: ' + str((correct / len(test_data_loader.dataset) * 100)) + '%')

#     preds = preds.cpu().numpy()
#     targets = targets.cpu().numpy()
#     self_feats = self_feats.cpu().numpy()
#     np.savetxt('result.csv', np.concatenate((targets, preds, self_feats), axis=1), delimiter=',')

#     return test_loss, preds

def test_emodel(model, criterion, test_data_loader, max_epochs, accs=None):
    preds = None
    model.eval()

    targets = None
    self_feats = None
    test_losses = []  # 에포크별 테스트 손실을 저장

    with torch.no_grad():
        for epoch in range(max_epochs):  # 에포크마다 테스트 실행
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

            test_loss /= len(test_data_loader.dataset)  # 에포크마다 손실 계산

            test_losses.append(test_loss)  # 손실 저장

            if accs is not None:
                accs.append(correct / len(test_data_loader.dataset) * 100)

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        self_feats = self_feats.cpu().numpy()
        np.savetxt('result.csv', np.concatenate((targets, preds, self_feats), axis=1), delimiter=',')

    return test_losses, preds  # 에포크별 손실 반환



# def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, train, test, collate, accs=None):
#     num_data_points = len(dataset)
#     size_fold = int(len(dataset) / float(num_folds))
#     folds = []
#     models = []
#     optimizers = []
#     test_losses = []

#     for k in range(0, num_folds - 1):
#         folds.append(dataset[k * size_fold:(k + 1) * size_fold])

#     folds.append(dataset[(num_folds - 1) * size_fold:num_data_points])

#     for k in range(0, num_folds):
#         models.append(copy.deepcopy(model))
#         optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01))

#     # Fold마다 손실 기록
#     fold_train_losses = []  # 각 fold별 train loss
#     fold_test_losses = []   # 각 fold별 test loss

    
#     for k in range(num_folds):
#         print('--------------- fold {} ---------------'.format(k + 1))

#         train_dataset = []
#         test_dataset = folds[k]

#         for i in range(0, num_folds):
#             if i != k:
#                 train_dataset += folds[i]


#         train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
#         test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


#         # Train the model for this fold
#         train_losses = train(models[k], criterion, optimizers[k], train_data_loader, max_epochs)  # 한 번에 max_epochs 실행
#         fold_train_losses.append(train_losses)

#         # Test the model for this fold
#         test_loss, pred = test(models[k], criterion, test_data_loader, accs)
#         test_losses.append(test_loss)

#         fold_test_losses.append([test_loss] * max_epochs)  # 테스트 손실은 에포크당 동일하게 기록


 
#         print(f"Fold {k+1}, Test Loss: {test_loss:.4f}")

#     # Plot fold별 loss
#     fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)
#     for k in range(num_folds):
#         epochs = list(range(1, max_epochs + 1))
#         axes[k].plot(epochs, fold_train_losses[k], label="Train Loss")
#         axes[k].plot(epochs, fold_test_losses[k], label="Test Loss")
#         axes[k].set_xlabel("Epoch")
#         axes[k].set_title(f"Fold {k+1}")

#         if k == 0:
#             axes[k].set_ylabel("Loss")
#         axes[k].legend()
#     plt.suptitle("Train and Test Loss Across Folds")
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()


#     if accs is None:
#         return np.mean(test_losses)
#     else:
#         return np.mean(test_losses), np.mean(accs)

# def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, train, test, collate, accs=None):
#     num_data_points = len(dataset)
#     size_fold = int(len(dataset) / float(num_folds))
#     folds = []
#     models = []
#     optimizers = []
#     test_losses = []

#     for k in range(0, num_folds - 1):
#         folds.append(dataset[k * size_fold:(k + 1) * size_fold])

#     folds.append(dataset[(num_folds - 1) * size_fold:num_data_points])

#     for k in range(0, num_folds):
#         models.append(copy.deepcopy(model))
#         optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01))

#     # Fold별 손실 기록
#     fold_train_losses = []  # 각 fold별 train loss
#     fold_test_losses = []   # 각 fold별 test loss

#     for k in range(num_folds):
#         print('--------------- fold {} ---------------'.format(k + 1))

#         train_dataset = []
#         test_dataset = folds[k]

#         for i in range(0, num_folds):
#             if i != k:
#                 train_dataset += folds[i]

#         train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
#         test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

#         # Train the model for this fold
#         train_losses = train(models[k], criterion, optimizers[k], train_data_loader, max_epochs)
#         fold_train_losses.append(train_losses)

#         # Test the model for this fold
#         test_loss,_ = test(models[k], criterion, test_data_loader, accs)
#         fold_test_losses.append([test_loss] * max_epochs)  # 테스트 손실을 각 에포크별로 동일하게 유지
        
#         print(f"Fold {k+1}, Test Loss: {test_loss:.4f}")

#     # Loss 시각화
#     fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)
#     for k in range(num_folds):
#         epochs = list(range(1, max_epochs + 1))
#         axes[k].plot(epochs, fold_train_losses[k], label="Train Loss")
#         axes[k].plot(epochs, fold_test_losses[k], label="Test Loss")
#         axes[k].set_xlabel("Epoch")
#         axes[k].set_title(f"Fold {k+1}")

#         if k == 0:
#             axes[k].set_ylabel("Loss")
#         axes[k].legend()
#     plt.suptitle("Train and Test Loss Across Folds")
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

#     if accs is None:
#         return np.mean([loss[-1] for loss in fold_train_losses]), np.mean([loss[-1] for loss in fold_test_losses])
#     else:
#         return np.mean([loss[-1] for loss in fold_train_losses]), np.mean([loss[-1] for loss in fold_test_losses]), np.mean(accs)


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

    fold_train_losses = []  # 각 fold별 train loss
    fold_test_losses = []   # 각 fold별 test loss

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
        train_losses = train(models[k], criterion, optimizers[k], train_data_loader, max_epochs)
        fold_train_losses.append(train_losses)

        # Test the model for this fold
        test_losses_epoch, _ = test(models[k], criterion, test_data_loader, max_epochs, accs)
        fold_test_losses.append(test_losses_epoch)  # 각 에포크별 테스트 손실 저장

        print(f"Fold {k+1}, Test Loss: {test_losses_epoch[-1]:.4f}")

    # Plot fold별 loss
    fig, axes = plt.subplots(1, num_folds, figsize=(5 * num_folds, 5), sharey=True)
    for k in range(num_folds):
        epochs = list(range(1, max_epochs + 1))
        axes[k].plot(epochs, fold_train_losses[k], label="Train Loss")
        axes[k].plot(epochs, fold_test_losses[k], label="Test Loss")
        axes[k].set_xlabel("Epoch")
        axes[k].set_title(f"Fold {k+1}")

        if k == 0:
            axes[k].set_ylabel("Loss")
        axes[k].legend()
    plt.suptitle("Train and Test Loss Across Folds")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    if accs is None:
        return np.mean(test_losses)
    else:
        return np.mean(test_losses), np.mean(accs)
