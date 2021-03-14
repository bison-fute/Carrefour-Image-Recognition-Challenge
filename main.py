from sklearn.metrics import accuracy_score
from torchvision import models
from torchvision.utils import make_grid
import time
import pickle as pk
import torch
import sys
from torchvision import transforms
from torch.nn import Linear, Sequential, CrossEntropyLoss
import torch.optim as optim
from images_processing import *
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


## PyTorch setup
print("Versions:")
print("torch version:", torch.__version__)
print("system version: ", sys.version)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

## Dataset Loading
data_path = "data/"

# Note: start by running metadata_main
barcode_to_cat = pk.load(open('barcode_to_cat', 'rb'))
cat_to_name = pk.load(open('cat_to_name', 'rb'))
batch_size = 64

# See if we need to normalize (it depends on the model used)
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

dataset = ImageFolderWithPaths(data_path, vgg_transform, barcode_to_cat)  # our custom dataset
train_size = int(len(dataset)*0.7)
test_size = len(dataset) - train_size
train_dsets, test_dsets = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(
    train_dsets,
    batch_size,
    num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    test_dsets,
    batch_size,
    num_workers=4
)

## Example of implementation
epochs = 1
lr = 0.0005
n_outputs = len(cat_to_name)  # Number of categories
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
start = time.time()
for epoch in range(epochs):

    # keep track of training and validation loss
    train_loss = 0.0
    training_loss = []
    for i, (x, y) in enumerate(train_loader):
        print('batch', i, 'over', len(train_loader))
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    training_loss_avg = np.average(training_loss)
    print('epoch:', epoch+1, 'training loss: %.3f' %training_loss_avg)
end_time = time.time() - start

## eval step
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

## visualization of the first images, with their respective categories and predictions:
imshow(make_grid(x[:8]))
for i in range(16):
    print("Name of the category:", cat_to_name[y[i].item()])
    print("Prediction:", cat_to_name[predictions[i].item()])

## visualization of learning curves
f, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.set_title("Learning curve")
ax.set_xlabel('batches')
ax.set_ylabel('Training loss')
ax.plot(np.arange(0, len(train_loader)), training_loss, color="gray", linestyle="--")
ax.legend('Training')
plt.show()
plt.savefig('learning_curve_train_{}.png'.format(int(time.time())))
print("Time elapsed: {} seconds".format(end_time))
print("Final training loss: {}".format(train_loss[-1]))
