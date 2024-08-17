import torch as tc

from torch.nn import Module
from torch.nn.functional import softmax
from torchvision.models import vgg16

class VGG16(Module):
    """A VGG16 neural network."""

    def __init__(self, n_classes):
        """Create new VGG16 network."""
        
        super(VGG16, self).__init__()
        self._my_vgg16 = vgg16(num_classes=n_classes)

    def forward(self, x):
        return softmax(self._my_vgg16(x))
    
    def save_model(self, filename):
        """Save model on disk."""

        tc.save(self.state_dict(), filename)

    def load_model(self, filename):
        """Load model from disk."""

        self.load_state_dict(tc.load(filename))