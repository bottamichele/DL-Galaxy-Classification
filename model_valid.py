import torch as tc
import matplotlib.pyplot as plt

from dataset import GalaxyDataset, NUM_CLASSES
from xception import Xception
from vgg16 import VGG16
from resnet50 import ResNet50

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# ========================================
# ========== VALIDATION FUNCTION =========
# ========================================

DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")

def validation_model(model, batch_size):
    #Load validation set from disk.
    valid_data = GalaxyDataset(mode_dataset="valid", device=DEVICE)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    #Model is set to eval mode.
    model.to(DEVICE)
    model.eval()

    #Validation phase.
    labels_pred = []        #Labels predicted by neural networks.
    labels_true = []        #Labels of validation set.
    
    print("Validation Phase")
    print("--------------------------------------------------")
    print("Validation set size = {}".format(len(valid_data)))
    print("")
    
    with tc.no_grad():
        for x, y in valid_dataloader:
            x = x.to(DEVICE)
            y_pred = model(x)

            labels_pred += y_pred.argmax(dim=1).cpu().detach().numpy().tolist()
            labels_true += y.cpu().detach().numpy().tolist()

    #Print validation accuracy.
    print("- Validation Accuracy = {:.2f}%".format(100 * accuracy_score(labels_true, labels_pred, normalize=True))) 

    #Print precision score.
    print("- Precision score = {}".format(100 * precision_score(labels_true, labels_pred, average=None)))

    #Print recall score.
    print("- Recall score = {}".format(100 * recall_score(labels_true, labels_pred, average=None))) 

    #Print confusion matrix
    print("- Confusion matrix:")
    cm = confusion_matrix(labels_true, labels_pred)
    
    cmd = ConfusionMatrixDisplay(cm, display_labels=["Sm_r", "Sm_in", "Sm_c", "Lent_e", "Sp_b", "Sp_nob"])
    cmd.plot()

    print(cm)
    plt.show()

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    VALID_XCEPTION_MODEL = True
    VALID_VGG16_MODEL = False
    VALID_RESNET50_MODEL = False

    if VALID_XCEPTION_MODEL and not VALID_VGG16_MODEL and not VALID_RESNET50_MODEL:
        #Validation for Xception's model. 
        model = Xception(NUM_CLASSES)
        model.load_model("./model/xception_model.pth")

        validation_model(model, batch_size=256)
    elif VALID_VGG16_MODEL and not VALID_XCEPTION_MODEL and not VALID_RESNET50_MODEL:
        #Validation for VGG16's model. 
        model = VGG16(NUM_CLASSES)
        model.load_model("./model/vgg16_model.pth")

        validation_model(model, batch_size=256)
    elif VALID_RESNET50_MODEL and not VALID_XCEPTION_MODEL and not VALID_VGG16_MODEL:
        #Validation for VGG16's model. 
        model = ResNet50(NUM_CLASSES)
        model.load_model("./model/resnet50_model.pth")

        validation_model(model, batch_size=256)
    else:
        print("Invalid choice for validation!")