import os
import sys
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import train, evaluate
from mymodels import loadDataSet,MyCNN, MyRNN

torch.manual_seed(0)
if torch.cuda.is_available():
	torch.cuda.manual_seed(0)

# Set a correct path to the seizure data file you downloaded
PATH_TRAIN_FILE = './data/train.csv'
PATH_TEST_FILE = './data/test.csv'
PATH_OUTPUT = "./mimic/sepsis/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

# Some parameters
MODEL_TYPE = 'RNN'  # TODO: Change this to 'MLP', 'CNN', or 'RNN' according to your task
NUM_EPOCHS = 1	
BATCH_SIZE = 42
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

train_dataset = loadDataSet(pd.read_csv(PATH_TRAIN_FILE), MODEL_TYPE)
# valid_dataset = load_seizure_dataset(PATH_VALID_FILE, MODEL_TYPE)
test_dataset = loadDataSet(pd.read_csv(PATH_TEST_FILE), MODEL_TYPE)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


if MODEL_TYPE == 'CNN':
	model = MyCNN()
	save_file = 'MyCNN.pth'
elif MODEL_TYPE == 'RNN':
	model = MyRNN()
	save_file = 'MyRNN.pth'
else:
	raise AssertionError("Wrong Model Type!")

lrate = 0.1

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)


for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)

torch.save(model, os.path.join(PATH_OUTPUT, save_file))

print('**************** done with training and validation ********************')
plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)

# print(test_results)

print('**************** done with testing ********************')

class_names = ['Seizure', 'TumorArea', 'HealthyArea', 'EyesClosed', 'EyesOpen']
plot_confusion_matrix(test_results, class_names)
