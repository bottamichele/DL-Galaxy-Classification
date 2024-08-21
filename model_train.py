import torch as tc
import numpy as np
import os

from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from dataset import GalaxyDataset, NUM_CLASSES
from xception import Xception
from vgg16 import VGG16
from resnet50 import ResNet50

# ========================================
# ========================================
# ========================================

PATH_MODEL = "./model/"
DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")

# ========================================
# ============ EARLY STOPPING ============
# ========================================

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# ========================================
# ============ TRAIN FUNCTIONS ===========
# ========================================

def train_step(dataloader, model, loss_function, optimizer):
    size_dataset = len(dataloader.dataset)
    labels_pred = []        #Labels predicted by neural network.
    labels_true = []        #Labels of training set.
    losses = []

    model.train()           #Model is set to training mode.
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        #Compute prediction and loss.
        y_pred = model(x)
        loss = loss_function(tc.log(y_pred), y)
        losses.append(loss.cpu().item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            print("Loss: {:>.5f}  [{:>5d}/{:>5d}]".format(loss.item(), len(x) * (batch + 1), size_dataset))

        labels_pred += y_pred.argmax(dim=1).cpu().detach().numpy().tolist()
        labels_true += y.cpu().detach().numpy().tolist()

    #Print training stats
    accuracy = 100 * accuracy_score(labels_true, labels_pred, normalize=True)
    print("- Training Accuracy = {:.2f}%".format(accuracy))

    avg_loss = np.mean(losses)
    print("- Training Loss: {:>.5f}".format(avg_loss))

    return accuracy, avg_loss


def validation_step(dataloader, model, loss_function):
    size_validset = len(dataloader.dataset)
    labels_pred = []            #Labels predicted by neural network.
    labels_true = []            #Labels of validation set.
    losses = []

    if size_validset <= 0:
        return

    model.eval()        #Model is set to evaluetion mode.
    with tc.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            #Predict validition points
            y_pred = model(x)       

            labels_pred += y_pred.argmax(dim=1).cpu().detach().numpy().tolist()
            labels_true += y.cpu().detach().numpy().tolist()
            losses.append( loss_function(tc.log(y_pred), y).cpu().item() )

    #Print validation stats.
    accuracy = 100 * accuracy_score(labels_true, labels_pred, normalize=True)
    print("- Validation Accuracy = {:.2f}%".format(accuracy))
    
    avg_loss = np.mean(losses)
    print("- Validation Loss = {:>.5f}".format(avg_loss))

    return accuracy, avg_loss

def train_model(model, model_name, epochs=50, batch_size=64, learning_rate=10**-4, early_stop_count=10, early_stop_delta=0):
    model.to(DEVICE)

    #Load training and validation set.
    training_data = GalaxyDataset(mode_dataset="train", device=DEVICE)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    
    validation_data = GalaxyDataset(mode_dataset="valid", device=DEVICE)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    #Optimizer and loss function.
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = NLLLoss()

    #Set other settings.
    early_stopper = EarlyStopper(patience=early_stop_count, min_delta=early_stop_delta)
    train_history = []
    valid_history = []

    #Start training.
    os.makedirs(PATH_MODEL, exist_ok=True)
    os.makedirs(PATH_MODEL + "training/")
    
    print("Training Phase")
    print("--------------------------------------------------")
    print("Training set size = {}; Validation set size = {}".format(len(training_data), len(validation_data)))
    print("")

    for t in range(1, epochs+1):
        print("Epoch {}".format(t))
        print("--------------------------------------------------")

        train_acc, train_loss = train_step(train_dataloader, model, loss_function, optimizer)
        valid_acc, valid_loss = validation_step(validation_dataloader, model, loss_function)

        train_history.append((train_acc, train_loss))
        valid_history.append((valid_acc, valid_loss))

        model.save_model(f"{PATH_MODEL}training/{model_name}_ep_{t}.pth")
        np.save(PATH_MODEL + "training/train_history.npy", train_history)
        np.save(PATH_MODEL + "training/valid_history.npy", valid_history)
        
        print("")

        if early_stopper.early_stop(valid_loss):
           break

    #Save model trained on disk.
    model.save_model(PATH_MODEL + model_name + ".pth")

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    TRAIN_XCEPTION_MODEL = True
    TRAIN_VGG16_MODEL = False
    TRAIN_RESNET50_MODEL = False

    if TRAIN_XCEPTION_MODEL and not TRAIN_VGG16_MODEL and not TRAIN_RESNET50_MODEL:
        #Train a Xception's model
        xception_model = Xception(NUM_CLASSES)

        train_model(xception_model, "xception_model")
    elif TRAIN_VGG16_MODEL and not TRAIN_XCEPTION_MODEL and not TRAIN_RESNET50_MODEL:
        #Train a VGG16's model.
        vgg16_model = VGG16(NUM_CLASSES)

        train_model(vgg16_model, "vgg16_model")
    elif TRAIN_RESNET50_MODEL and not TRAIN_XCEPTION_MODEL and not TRAIN_VGG16_MODEL:
        #Train a ResNet34's model.
        resnet50_model = ResNet50(NUM_CLASSES)

        train_model(resnet50_model, "resnet50_model")
    else:
        print("Invalid choice of training!")