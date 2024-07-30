import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torch.distributions import Normal
from pytorch_model_summary import summary

class Retina:
    """A visual retina.

    Extracts a foveated glimpse `phi` around location `l`
    from an image `x`.

    Concretely, encodes the region around `l` at a
    high-resolution but uses a progressively lower
    resolution for pixels further from `l`, resulting
    in a compressed representation of the original
    image `x`.

    Args:
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l: a 2D Tensor of shape (B, 2). Contains normalized
            coordinates in the range [-1, 1].
        g: size of the first square patch.
        k: number of patches to extract in the glimpse.
        s: scaling factor that controls the size of
            successive patches.

    Returns:
        phi: a 5D tensor of shape (B, k, g, g, C). The
            foveated glimpse of the image.
    """

    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def extract_patch(self, x, l, patch_size):
        """
        Extract patches using bilinear interpolation.
        x: input image tensor of shape (B, C, H, W)
        l: normalized coordinates tensor of shape (B, 2) with values in [-1, 1]
        patch_size: size of the patch (e.g., 5 for a 5x5 patch)
        """
        B, C, H, W = x.size()
        theta = torch.zeros(B, 2, 3).to(x.device)

        theta[:, 0, 0] = patch_size / W
        theta[:, 1, 1] = patch_size / H
        theta[:, :, 2] = l

        grid = F.affine_grid(theta, torch.Size((B, C, patch_size, patch_size)), align_corners=False)
        patches = F.grid_sample(x, grid, align_corners=False)
        return patches

    def foveate(self, x, l):
        """Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.g

        # extract k patches of increasing size
        for i in range(self.k):
            patch = self.extract_patch(x, l, size)
            #if size != self.g:
            #    patch = F.interpolate(patch, size=(self.g, self.g), mode='bilinear', align_corners=False)
            phi.append(patch)
            size = int(self.s * size)
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        return phi
class CustomResNetStep1(nn.Module):
    def __init__(self, input_channels=3, output_dim=256, initial_channels=64):
        super(CustomResNetStep1, self).__init__()
        self.in_channels = initial_channels
        
        # Simplified initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Residual layers
        self.layer1 = self._make_layer(models.resnet.BasicBlock, initial_channels, 2)
        self.layer2 = self._make_layer(models.resnet.BasicBlock, initial_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(models.resnet.BasicBlock, initial_channels * 4, 2, stride=2)
        
        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(initial_channels * 4 * models.resnet.BasicBlock.expansion, output_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CustomConvNet(nn.Module):
    def __init__(self, input_channels, output_dim, initial_channels=16):
        super(CustomConvNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(initial_channels, initial_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(initial_channels * 2, initial_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(initial_channels * 4, output_dim)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_dim=256):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # Load a pretrained ResNet18
        
        # Modify the first convolutional layer to accept N input channels
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the classification layer
        self.fc = nn.Linear(512, output_dim)  # Add a new fully connected layer with 512 input features (resnet18 specific)

    def forward(self, x):
        x = self.resnet(x)  # Pass through ResNet
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.fc(x)  # Pass through the new fully connected layer
        return x

class GlimpseNetwork(nn.Module):
    """The glimpse network.

    Combines the "what" and the "where" into a glimpse
    feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args:
        h_g: hidden layer size of the fc layer for `phi`.
        h_l: hidden layer size of the fc layer for `l`.
        g: size of the square patches in the glimpses extracted
        by the retina.
        k: number of patches to extract per glimpse.
        s: scaling factor that controls the size of successive patches.
        c: number of channels in each image.
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
            coordinates [x, y] for the previous timestep `t-1`.

    Returns:
        g_t: a 2D tensor of shape (B, hidden_size).
            The glimpse representation returned by
            the glimpse network for the current
            timestep `t`.
    """

    def __init__(self, h_g, h_l, g, k, s, c):
        super().__init__()

        self.retina = Retina(g, k, s)
        self.k = k #needed for the forward to know how many patches to concatenate probably dont need to know XD

        # glimpse layer
        D_in = k * g * g * c
        self.fc1 = nn.Linear(D_in, h_g)
            # convnet instead of fully connected
        self.feature_extractor = CustomResNetStep1(input_channels=1 * k, output_dim=h_g, initial_channels=16)
        #self.feature_extractor = CustomConvNet(input_channels=1 * k, output_dim=h_g, initial_channels=16)
        # self.feature_extractor = ResNetFeatureExtractor(input_channels=1 * k, output_dim=h_g)
        # get an an overviewl of the ResnetSize
        print(summary(self.feature_extractor, torch.zeros((1,1 * k, 224, 224))))

        # location layer
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_l)

        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x in different res as a list
        phi = self.retina.foveate(x, l_t_prev)
        # concatenate all the patches to (B, Glimpses x Channels,H,W)
        phi = torch.cat(phi, 1)
        # extract features from the concatenated patches
        if True:
            phi_out = self.feature_extractor(phi)

        if False:
            # Flatten use when using Linear layer 
            phi = phi.view(phi.shape[0], -1)
            phi_out = F.relu(self.fc1(phi))

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return g_t


class CoreNetwork(nn.Module):
    """The core network.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The
            hidden state vector for the previous timestep `t-1`.

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class ActionNetwork(nn.Module):
    """The action network.

    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class LocationNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std

        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t):
        # compute mean
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))

        # reparametrization trick
        l_t = torch.distributions.Normal(mu, self.std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu, self.std).log_prob(l_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t


class BaselineNetwork(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        b_t: a 2D vector of shape (B, 1). The baseline
            for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t
