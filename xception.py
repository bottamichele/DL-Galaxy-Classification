import torch as tc

from torch.nn import Module, Sequential, ReLU, Softmax, Conv2d, BatchNorm2d, MaxPool2d, Linear, Identity

# ========================================
# ======= SEPARABLE CONVOLUTION 2D =======
# ========================================

class SeparableConv2d(Module):
    """A Xception's separable convolution 2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """Create new Xception's separable convolution 2d layer."""

        super(SeparableConv2d, self).__init__()
        self._pointwise = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same")
        self._depthwise = Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=out_channels)

    def forward(self, x):
        """Compute x by separable convolution 2d layer."""

        x = self._pointwise(x)
        x = self._depthwise(x)
        return x

# ========================================
# =========== XCEPTION'S BLOCK ===========
# ========================================

class Block(Module):
    """A Xception's block."""

    def __init__(self, in_channels, out_channels, start_with_relu, end_with_pooling):
        """Create a new Xception's block."""
        
        super(Block, self).__init__()

        #New skip connection.
        self._skip_connection = Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0) if end_with_pooling else Identity()

        #Layers in cascade. 
        self._layers_seq = Sequential()
        
        if start_with_relu:
            self._layers_seq.append(ReLU())

        self._layers_seq.append(SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self._layers_seq.append(BatchNorm2d(out_channels))
        self._layers_seq.append(ReLU())
        self._layers_seq.append(SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self._layers_seq.append(BatchNorm2d(out_channels))

        if end_with_pooling:
            self._layers_seq.append(MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        return self._skip_connection(x) + self._layers_seq(x)

# ========================================
# =============== XCEPTION ===============
# ========================================

class Xception(Module):
    """A deep neural network that implements Xception architecture."""

    def __init__(self, n_classes):
        """Crate new Xception network."""

        super(Xception, self).__init__()

        self._conv_1 = Sequential(Conv2d(3, 32, kernel_size=3, stride=2, padding=1), BatchNorm2d(32), ReLU())        # (3, 299, 299)     ==> (32, 149, 149)
        self._conv_2 = Sequential(Conv2d(32, 64, kernel_size=3, stride=1, padding=1), BatchNorm2d(64), ReLU())       # (32, 149, 149)    ==> (64, 147, 147)
        self._block_1 = Block(64, 128, start_with_relu=False, end_with_pooling=True)
        self._block_2 = Block(128, 256, start_with_relu=True, end_with_pooling=True)
        self._block_3 = Block(256, 728, start_with_relu=True, end_with_pooling=True)
        self._block_4_11 = Sequential()
        self._block_12 = Block(728, 1024, start_with_relu=True, end_with_pooling=True)
        self._sep_conv_1 = Sequential(SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1), BatchNorm2d(1536), ReLU())
        self._sep_conv_2 = Sequential(SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1), BatchNorm2d(2048), ReLU())
        self._fc1 = Sequential(Linear(2048, 2048), ReLU())
        self._fc2 = Sequential(Linear(2048, 2048), ReLU())
        self._out = Sequential(Linear(2048, n_classes), Softmax())

        for _ in range(8):
            self._block_4_11.append(Block(728, 728, start_with_relu=True, end_with_pooling=False))

    def forward(self, x):
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._block_1(x)
        x = self._block_2(x)
        x = self._block_3(x)
        x = self._block_4_11(x)
        x = self._block_12(x)
        x = self._sep_conv_1(x)
        x = self._sep_conv_2(x)
        x = tc.mean(x.view(x.size(0), x.size(1), -1), dim=2)        #Global Avarage Pooling
        x = self._fc1(x)
        x = self._fc2(x)
        x = self._out(x)

        return x
    
    def save_model(self, filename):
        """Save model on disk."""

        tc.save(self.state_dict(), filename)

    def load_model(self, filename):
        """Load model from disk."""

        self.load_state_dict(tc.load(filename))