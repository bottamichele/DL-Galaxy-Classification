import torch as tc

from torch.nn import Module
from torch.nn.functional import softmax
from torchvision.models import resnet50

class ResNet50(Module):
    """A ResNet50 neural network."""

    def __init__(self, n_classes):
        """Create a new ResNet50 network."""

        super(ResNet50, self).__init__()
        self._my_resnet50 = resnet50(num_classes=n_classes)

    def forward(self, x):
        return softmax(self._my_resnet50(x))
    
    def save_model(self, filename):
        """Save model on disk."""

        tc.save(self.state_dict(), filename)

    def load_model(self, filename):
        """Load model from disk."""
        
        self.load_state_dict(tc.load(filename))