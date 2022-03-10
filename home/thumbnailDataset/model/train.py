import  torch

import time
import copy
from tqdm import tqdm
import wandb
import sklearn.metrics
import numpy as np

def train_thumbnail(model, dataloaders, criterion, optimizer, scheduler,
                dataset_sizes, device='cpu', num_epochs=25, wandb_log=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0

    print('-'*5 + 'Training the model' + '-'*5)
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
                if wandb_log:
                    wandb.watch(model)
            elif phase == 'val':
                val_loss = epoch_loss
                val_acc = epoch_acc

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        if wandb_log:
            wandb.log({'train_loss': train_loss,
                       'val_loss': val_loss,
                       'train_acc': train_acc,
                       'val_acc': val_acc,
                       'best_val_loss': best_loss})

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_model(model, dataloaders, criterion, optimizer, scheduler,
                dataset_sizes, device='cpu', num_epochs=25, wandb_log=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0

    print('-'*5 + 'Training the model' + '-'*5)
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0.0
            running_labels = None
            running_outputs = None

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = torch.sigmoid(outputs)
                    # _, preds = torch.max(outputs, 1)
                    preds = outputs > 0.5
                    if idx == 5:
                        print('labels', labels[0])
                        print('outputs', outputs[0])
                        print('preds', preds[0])
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.all(preds == (labels>0.5), dim=1).float().mean()
                if running_labels is not None:
                    running_labels = np.vstack([running_labels, labels.cpu().numpy()])
                else:
                    running_labels = labels.cpu().numpy()
                if running_outputs is not None:
                    running_outputs = np.vstack([running_outputs, outputs.cpu().numpy()])
                else:
                    running_outputs = outputs.cpu().numpy()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / len(dataloaders[phase])# dataset_sizes[phase]
            epoch_roc_auc = sklearn.metrics.roc_auc_score(running_labels, running_outputs, average=None)

            print('{} Loss: {:.4f} ROC_AUC: {}'.format(
                phase, epoch_loss, epoch_roc_auc))
            if phase == 'train':
                train_loss = epoch_loss
                train_roc_auc = epoch_roc_auc
                if wandb_log:
                    wandb.watch(model)
            elif phase == 'val':
                val_loss = epoch_loss
                val_roc_auc = epoch_roc_auc

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        if wandb_log:
            wandb.log({'train_loss': train_loss,
                       'val_loss': val_loss,
                       'train_roc_auc': train_roc_auc,
                       'train_roc_auc_mean': train_roc_auc.mean(),
                       'val_roc_auc': val_roc_auc,
                       'val_roc_auc_mean': val_roc_auc.mean(),
                       'best_val_loss': best_loss})

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model