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
train_size = int(len(dataset) * 0.7)
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

## Example of implementation
epochs = 4
lr = 0.0005
n_outputs = len(cat_to_name)  # Number of categories

# loading the pretrained model, adaption to the head to be made to run other model vgg16 and resnet50
# model = models.vgg16_bn(pretrained=True)
model = models.densenet161(pretrained=True)
# model = models.wide_resnet50_2(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Add on classifier
model.classifier = Sequential(
    Linear(2208, n_outputs)
)
print("Model Summary:")
print(model.classifier)

model = model.to(device)

for param in model.classifier.parameters():
    param.requires_grad = True

# specify loss function (categorical cross-entropy)
criterion = CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# training step
nb_of_batches = len(train_loader)
start = time.time()
training_loss = []  # keep track of training loss
for epoch in range(epochs):
    train_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # print('batch', i, 'over', nb_of_batches)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        # if i // 50 == 0 :
        #     print(time.time() - start, "7")
        #     validation_loss.append(get_validation_loss(test_loader, model, criterion, device))
        training_loss_avg = np.average(training_loss)
    print('epoch:', epoch + 1, 'training loss: %.3f' % training_loss_avg)
end_time = time.time() - start

## eval step
prediction_val = []
target_val = []
position_of_true_label_val = []
for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        output = model(x)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_val.append(predictions)
    target_val.append(y)
    position_of_true_label = len(cat_to_name) - (np.argwhere(np.array(prob).argsort()[::-1] == y.cpu().numpy().reshape(-1, 1))[:, 1] + 1)
    position_of_true_label_val.append(position_of_true_label.mean())  # add average position over the batch

# validation accuracy
accuracy_val = []
for i in range(len(prediction_val)):
    accuracy_val.append(accuracy_score(target_val[i].cpu(), prediction_val[i]))
print('validation accuracy: \t', np.average(accuracy_val))
avg_position_of_true_label = np.mean(position_of_true_label_val)
# print("Average rank of the correct prediction: %.1f" % avg_position_of_true_label)

## printing accuracy
# print_save_accuracy(train_loader, test_loader, model, accuracy_score, device, train=True, test=True)

## visualization of the first images, with their respective categories and predictions:
imshow(make_grid(x[:8]))
for i in range(16):
    print("Name of the category:", cat_to_name[y[i].item()])
    print("Prediction:", cat_to_name[predictions[i].item()])

## visualization of learning curves
f, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.set_title("Learning curve")
ax.set_xlabel('batches (1 epoch = 122 batches)')
ax.set_ylabel('Training loss')
ax.plot(np.arange(0, len(training_loss)), training_loss, color="gray", linestyle="--")
ax.legend(['Training', 'Test'])
plt.show()
plt.savefig('learning_curves_{}.png'.format(int(time.time())))
print("Time elapsed: {} seconds".format(end_time))

##
position_of_true_label_el = []
for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        output = model(x)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_val.append(predictions)
    target_val.append(y)
    position_of_true_label = np.argwhere(np.array(prob).argsort()[::-1] == y.cpu().numpy().reshape(-1, 1))[:, 1] + 1
    position_of_true_label_el.append(position_of_true_label)  # add average position over the batch
