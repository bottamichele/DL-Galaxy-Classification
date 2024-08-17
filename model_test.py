import torch as tc
import matplotlib.pyplot as plt

from dataset import GalaxyDataset, NUM_CLASSES
from xception import Xception
from vgg16 import VGG16
from resnet50 import ResNet50

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# ========================================
# ============= TEST FUNCTION ============
# ========================================

DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")

def test_model(model, batch_size):
    #Load test set from disk.
    test_data = GalaxyDataset(mode_dataset="test", device=DEVICE)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    #Set model to eval mode.
    model.to(DEVICE)
    model.eval()

    #Test phase.
    labels_pred = []        #Labels predicted by neural network.
    labels_true = []        #Labels of test set.

    print("Test Phase")
    print("--------------------------------------------------")
    print("Test set size = {}".format(len(test_data)))
    print("")

    with tc.no_grad():
        for x, y in test_dataloader:
            x = x.to(DEVICE)
            y_pred = model(x)

            labels_pred += y_pred.argmax(dim=1).cpu().detach().numpy().tolist()
            labels_true += y.cpu().detach().numpy().tolist()

    #Print test accuracy
    print("- Test Accuracy = {:.2f}%".format(100 * accuracy_score(labels_true, labels_pred, normalize=True)))

    #Print precision score.
    print("- Precision score = {}".format(100 * precision_score(labels_true, labels_pred, average=None)))

    #Print recall score.
    print("- Recall score = {}".format(100 * recall_score(labels_true, labels_pred, average=None)))

    #Print f1 score.
    print("- F1 score = {}".format(100 * f1_score(labels_true, labels_pred, average=None)))    

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
    TEST_XCEPTION_MODEL = True
    TEST_VGG16_MODEL = False
    TEST_RESNET50_MODEL = False

    if TEST_XCEPTION_MODEL and not TEST_VGG16_MODEL and not TEST_RESNET50_MODEL:
        #Test Xception's model trained.
        model = Xception(NUM_CLASSES)
        model.load_model("./model/xception_model.pth")

        test_model(model, batch_size=256)
    elif TEST_VGG16_MODEL and not TEST_XCEPTION_MODEL and not TEST_RESNET50_MODEL:
        #Test VGG16's model trained.
        model = VGG16(NUM_CLASSES)
        model.load_model("./model/vgg16_model.pth")

        test_model(model, batch_size=256)
    elif TEST_RESNET50_MODEL and not TEST_XCEPTION_MODEL and not TEST_VGG16_MODEL:
        #Test ResNet50's model trained.
        model = ResNet50(NUM_CLASSES)
        model.load_model("./model/resnet50_model.pth")

        test_model(model, batch_size=256)
    else:
        print("Invalid choice for testing!")