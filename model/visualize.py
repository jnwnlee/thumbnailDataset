import torch
import matplotlib.pyplot as plt
import os
import wandb
import numpy as np
import sklearn

from utils.utils import imshow

def visualize_thumb(model, dataloaders, class_names, args, device='cpu', num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(8, 8*num_images//2))

    with torch.no_grad():
        for i, (inputs, labels, tags) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}\nlabel: {labels[j]}\ntag: {tags[j]}')

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)

                    fname = [args.learning_type, 'batch', str(args.batch_size), 'n_epochs', str(args.n_epochs),
                             'lr', str(args.lr), 'step_size', str(args.step_size), 'seed', str(args.seed)]
                    fname = '_'.join(fname) + '.jpg'
                    fig.tight_layout()
                    #fig.subplots_adjust(top=0.8)
                    # plt.suptitle(str(label_list[:num_images]) + str(tag_list[:num_images]), y=0.98)
                    plt.savefig(os.path.join('examples/', fname))
                    return
        model.train(mode=was_training)

        fname = [args.learning_type, 'batch', str(args.batch_size), 'n_epochs', str(args.n_epochs),
                 'lr', str(args.lr), 'step_size', str(args.step_size), 'seed', str(args.seed)]
        fname = '_'.join(fname) + '.jpg'
        fig.tight_layout()
        #fig.subplots_adjust(top=0.8)
        # plt.suptitle(str(label_list) + str(tag_list), y=0.98)
        plt.savefig(fname)
        if args.wandb_log:
            wandb.log({"pred": fig})

def visualize_model(model, dataloader, class_names, args, device='cpu', data_dir=None, task=None, num_images=6):
    if task not in ['multiclass', 'multilabel']:
        raise ValueError(f'task {task} should be one of the following: \"multiclass\" or \"multilabel\"')

    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(8, 15*num_images//2))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if task == 'multiclass':
                _, preds = torch.max(outputs, 1)
            elif task == 'multilabel':
                outputs = torch.sigmoid(outputs)
                preds = outputs > 0.5

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                if task == 'multiclass':
                    ax.set_title(f'predicted: {class_names[preds[j]]}\nlabel: {class_names[labels[j]]}',
                                 fontsize=6)
                elif task == 'multilabel':
                    ax.set_title(f'predicted: {np.array(class_names)[np.where(preds[j].cpu().numpy() > 0.5)]}'
                                 + f'\nlabel: {np.array(class_names)[np.where(labels[j].cpu().numpy() > 0.5)]}',
                                 fontsize=6)

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)

                    fname = ['['+data_dir.split('/')[-1]+']', args.learning_type, 'batch', str(args.batch_size),
                             'n_epochs', str(args.n_epochs),
                             'lr', str(args.lr), 'step_size', str(args.step_size), 'seed', str(args.seed)]
                    fname = '_'.join(fname) + '.png'
                    fig.tight_layout()
                    #fig.subplots_adjust(top=0.8)
                    # plt.suptitle(str(label_list[:num_images]) + str(tag_list[:num_images]), y=0.98)
                    plt.savefig(os.path.join('examples/', fname))
                    if args.wandb_log:
                        wandb.log({"pred": fig})
                    return
        model.train(mode=was_training)

        fname = ['['+data_dir.split('/')[-1]+']', args.task, args.learning_type, 'batch', str(args.batch_size),
                 'n_epochs', str(args.n_epochs),
                 'lr', str(args.lr), 'step_size', str(args.step_size), 'seed', str(args.seed)]
        fname = '_'.join(fname) + '.png'
        fig.tight_layout()
        #fig.subplots_adjust(top=0.8)
        # plt.suptitle(str(label_list) + str(tag_list), y=0.98)
        plt.savefig(os.path.join('examples/', fname))
        if args.wandb_log:
            wandb.log({"pred": fig})

def visualize_confusion_matrix(model, dataloader, class_names, args, device='cpu', data_dir=None):
    was_training = model.training
    model.eval()

    labels_stack = None
    preds_stack = None

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if labels_stack is not None:
                labels_stack = np.hstack([labels_stack, labels.clone().detach().cpu().numpy()])
            else:
                labels_stack = labels.clone().detach().cpu().numpy()
            if preds_stack is not None:
                preds_stack = np.hstack([preds_stack, preds.clone().detach().cpu().numpy()])
            else:
                preds_stack = preds.clone().detach().cpu().numpy()

        model.train(mode=was_training)

        confusion_matrix = sklearn.metrics.confusion_matrix(labels_stack, preds_stack)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                      display_labels=class_names)
        disp.plot()
        plt.xticks(rotation=90)

        fname = ['['+data_dir.split('/')[-1]+']', 'confusionMtrx', args.task, args.learning_type,
                 'batch', str(args.batch_size), 'n_epochs', str(args.n_epochs),
                 'lr', str(args.lr), 'step_size', str(args.step_size), 'seed', str(args.seed)]
        fname = '_'.join(fname) + '.png'

        plt.savefig(os.path.join('examples/', fname))
        if args.wandb_log:
            wandb.log({"confusion_matrix": plt.gcf()})