import time
import fiftyone as fo
from tqdm import tqdm

# Create a dataset from a directory of images
dataset = fo.Dataset.from_dir(
                    dataset_dir='/media/daftpunk2/home/seungheon/thumbnailDataset_unsplash/',
                    dataset_type=fo.types.ImageClassificationDirectoryTree,
                    name='thumbnailDataset_unsplash'
            )

import fiftyone.zoo as foz
import torch
import fiftyone.brain as fob

# Load a resnet from the model zoo
model = foz.load_zoo_model("resnet50-imagenet-torch")

# Verify that the model exposes embeddings
print('model.has_embeddings', model.has_embeddings)
# True

# Compute embeddings for each image
embeddings = dataset.compute_embeddings(model, gpus=torch.cuda.device_count())
# print(embeddings.shape)
# 10000 x 2048

# Compute 2D representation using pre-computed embeddings
results = fob.compute_visualization(
    dataset,
    embeddings=embeddings,
    num_dims=2,
    brain_key="image_embeddings",
    verbose=True,
    seed=51,
)

print('len(embeddings)', len(embeddings))

session = fo.launch_app(dataset, remote=True, port=10000)

# Visualize image embeddings colored by time of day
plot = results.visualize(
    labels="ground_truth.label",
    labels_title="class",
    axis_equal=True,
)
plot.show(height=512)
plot.save('./embedding_thumbnailDataset_unsplash.jpg')

# Attach plot to session
session.plots.attach(plot)

session.wait()
print(session)

# from torchvision import models
# from torchvision.datasets import ImageFolder
# import torch
# import cv2
#
# def is_valid_img(path):
#     try:
#         im = cv2.imread(path)
#     except:
#         return False
#     else:
#         if im is None:
#             return False
#         return True
#
# def main():
#     data_dir = ??????
#     batch_size = 4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     seed = 10
#     random.seed(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     data_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     thumbnail_dataset = ImageFolder(data_dir, transform=data_transform, is_valid_file=is_valid_img)
#     dataloader = torch.utils.data.DataLoader(thumbnail_dataset, batch_size=batch_size,
#                                 shuffle=False, num_workers=4)
#
#     model = models.resnet50(pretrained=True, progress=True)
#     # fix model weights
#     for param in model.parameters():
#         param.requires_grad = False
#
#     model.eval()
#     for inputs, labels in tqdm(dataloader, desc='Running the model inference'):
#         inputs = batch['image'].to(device)
#         labels += batch['label'].to(device)
#         # image_paths += batch['image_path']
#
#         output = model.forward(inputs)
#
#         current_outputs = output.cpu().numpy()
#         features = np.concatenate((outputs, current_outputs))
#
