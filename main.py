import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import accuracy_score
from torchvision import models,transforms,datasets
from torchvision.utils import make_grid

import pickle as pk
from torch.nn import Linear, Sequential, CrossEntropyLoss
import torch.optim as optim

from images_processing import*

from PIL import Image ; Image.MAX_IMAGE_PIXELS = 1000000000
from PIL import ImageFile ; ImageFile.LOAD_TRUNCATED_IMAGES = True


## Set up de PyTorch
print("Versions:")
print("torch version:", torch.__version__)
print("system version: ", sys.version)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

## Dataset Loading
data_path = "data/"

# Note: commencer par exécuter metadata_main
barcode_to_cat = pk.load(open('barcode_to_cat', 'rb'))
cat_to_name = pk.load(open('cat_to_name', 'rb'))
batch_size = 64

# Voir si on a besoin de normaliser (ça dépend du model utilisé)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

vgg_transform = transforms.Compose([
                transforms.RandomAffine(180, fillcolor=(255, 255, 255)),
                transforms.RandomGrayscale(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.RandomPerspective(fill=(255, 255, 255)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

dataset = ImageFolderWithPaths(data_path, vgg_transform, barcode_to_cat) # our custom dataset
train_size = int(len(dataset)*0.7)
test_size = len(dataset) - train_size
train_dsets, test_dsets = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(
    train_dsets,
    batch_size,
    num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dsets,
    batch_size,
    num_workers=0
)

## Exemple d'implémentation
epochs = 1
lr = 0.0005
n_outputs = len(cat_to_name)            # Nombre de catégories
# loading the pretrained model
model = models.vgg16_bn(pretrained=True)
# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Add on classifier
model.classifier[6] = Sequential(
    Linear(4096, n_outputs)
)
print("Model Summary:")
print(model.classifier)

for param in model.classifier[6].parameters():
    param.requires_grad = True

# specify loss function (categorical cross-entropy)
criterion = CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate
optimizer = optim.Adam(model.classifier[6].parameters(), lr=lr)

# training step
for epoch in range(epochs):

    # keep track of training and validation loss
    train_loss = 0.0

    training_loss = []
    for i, x, y in enumerate(train_loader):
        print('batch', i)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    training_loss = np.average(training_loss)
    print('epoch:', epoch, 'training loss: %.3f' %training_loss)

##
# eval step
prediction_val = []
target_val = []
for x, y in test_loader:

    with torch.no_grad():
        output = model(x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_val.append(predictions)
    target_val.append(y)

# validation accuracy
accuracy_val = []
for i in range(len(prediction_val)):
    accuracy_val.append(accuracy_score(target_val[i], prediction_val[i]))

print('validation accuracy: \t', np.average(accuracy_val))

## visualisation des premières images, avec leurs catégories et prédictions respectives:

imshow(make_grid(x[:8]))
for i in range(16):
    print("Nom de la catégorie:", cat_to_name[y[i].item()])
    print("Prédiction:", cat_to_name[predictions[i].item()])
##

