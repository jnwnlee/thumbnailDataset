import torch
import matplotlib.pyplot as plt
import os

from utils.utils import imshow

def visualize_model(model, dataloaders, class_names, args, device='cpu', num_images=6):
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