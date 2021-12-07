from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from dataset.thumbnail import ThumbnailDataset

def main():
    data_dir = 'data/thumbnail'
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    thumbnail_dataset = ThumbnailDataset(data_dir, extensions=IMG_EXTENSIONS, transform=data_transform)

    tags = set(tag for tag in thumbnail_dataset.tags)
    count_per_tag = {key: [0, 0] for key in tags}

    for i in range(len(thumbnail_dataset)):
        count_per_tag[thumbnail_dataset.tags[i]][thumbnail_dataset.targets[i]] += 1

    x = np.arange(len(tags))
    width = 0.4

    plt.figure(figsize=(10, 8))
    plt.bar(x - width / 2., [l[0] for l in count_per_tag.values()], width, label='bad')
    plt.bar(x + width / 2., [l[1] for l in count_per_tag.values()], width, label='good')
    plt.xticks(x, list(count_per_tag.keys()), rotation=45)
    plt.xlabel('tags')
    plt.ylabel('count #')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('data_portion.jpg')

if __name__ == '__main__':
    main()