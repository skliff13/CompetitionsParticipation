import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19
from torch.nn import Sequential
from torch.nn.modules.conv import Conv2d, Conv3d
from torch.nn.modules.pooling import MaxPool2d, MaxPool3d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.activation import ReLU
from torchviz import make_dot
from torchsummary import summary


class VGGlike(nn.Module):
    def _build_vgg_like(self, anchors, base, cut_layers, down_conv_z, out_channels, up_conv_z2):
        down_blocks = []
        block_layers = []
        sel_layers = base.features[:len(base.features) - cut_layers]
        for i, layer in enumerate(sel_layers):
            if layer.__class__ == Conv2d:
                ws, bs = layer.weight, layer.bias
                ws = ws.sum(axis=1, keepdims=True) if ws.shape[1] == 3 else ws
                ws = torch.unsqueeze(ws, -1)
                shp = ws.shape[:-1] + (3,) if down_conv_z else ws.shape
                pz = int(down_conv_z)
                print(f'Creating Conv3D layer with weights shape {tuple(shp)}')
                new_layer = Conv3d(in_channels=shp[1], out_channels=shp[0], kernel_size=shp[2:], padding=(1, 1, pz))
                new_layer.weight.data[..., shp[-1] // 2] = ws[..., 0]
                new_layer.bias.data[...] = bs[...]
                block_layers.append(new_layer)
            elif layer.__class__ == MaxPool2d:
                print('Creating MaxPooling3D layer')
                block_layers.append(MaxPool3d((2, 2, 1)))
            elif layer.__class__ == ReLU:
                print('Creating ReLU layer')
                block_layers.append(ReLU(inplace=True))
                if i in anchors:
                    down_blocks.append(DownBlock(block_layers, block_i=len(down_blocks)))
                    block_layers = []

        up_blocks = []
        last_filters = -1
        for i in range(len(down_blocks) - 2, -1, -1):
            block_d = down_blocks[i + 1]
            block_l = down_blocks[i]
            filters = block_l[-2].weight.shape[0]
            last_filters = filters
            in_channels = block_l[-2].weight.shape[1] + block_d[-2].weight.shape[1]
            up_blocks.append(UpBlock(filters, in_channels, block_i=len(up_blocks), up_conv_z2=up_conv_z2))

        for i, block in enumerate(down_blocks):
            setattr(self, f'down_{i}', block)

        for i, block in enumerate(up_blocks):
            setattr(self, f'up_{i}', block)

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.final_conv = Conv3d(in_channels=last_filters, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        xs = [x]
        for block in self.down_blocks:
            xs.append(block(xs[-1]))

        x = xs[-1]
        for i, block in enumerate(self.up_blocks):
            x = block(x, xs[- i - 2])

        predictions = self.final_conv(x)
        return predictions

    def freeze_pretrained(self):
        for block in self.down_blocks:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_pretrained(self):
        for block in self.down_blocks:
            for param in block.parameters():
                param.requires_grad = True


class DownBlock(nn.Sequential):
    def __init__(self, layers, block_i=0):
        super().__init__()
        for i, layer in enumerate(layers):
            name = f'd{block_i}-{i}_{layer.__class__.__name__.lower()}'
            self.add_module(name, layer)


class UpBlock(nn.Module):
    def __init__(self, filters, in_channels, block_i=0, up_conv_z2=False):
        super().__init__()
        print(f'Creating UpBlock with {filters} filters')
        self.conv_part = Sequential()
        conv0 = Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv_part.add_module(f'u{block_i}-0_conv3d', conv0)
        self.conv_part.add_module(f'u{block_i}-0_relu', ReLU())
        uz = int(up_conv_z2)
        conv1 = Conv3d(in_channels=filters, out_channels=filters, kernel_size=(3, 3, 1 + 2 * uz), padding=(1, 1, uz))
        self.conv_part.add_module(f'u{block_i}-1_conv3d', conv1)
        self.conv_part.add_module(f'u{block_i}-1_relu', ReLU())

        self.up_sample = Upsample(scale_factor=(2, 2, 1))

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.up_sample(x)
        both = torch.cat([x_e, x_0], dim=1)
        x = self.conv_part(both)
        return x


class VGG16J(VGGlike):
    def __init__(
        self,
        out_channels: int = 2,
        cut_layers: int = 1,
        down_conv_z: bool = False,
        up_conv_z2: bool = False
    ):
        """
        A VGG16-based version of UNet-like network.

        Args:
            out_channels: number of output channels. Defaults to 2.
            cut_layers: number of last VGG layers to be ignored. Good values are 1, 8, 15
            down_conv_z: if True, the convolution kernels in DownBlocks are 3x3x3; otherwise 3x3x1 (slice-wise)
            up_conv_z2: if True, both convolutional layers in each UpBlock have 3x3x3 kernel size; otherwise there is
                one layer with 3x3x3 and one layer with 3x3x1

        """
        super().__init__()

        base = vgg16(pretrained=True)
        anchors = {3, 8, 15, 22, 29}

        self._build_vgg_like(anchors, base, cut_layers, down_conv_z, out_channels, up_conv_z2)


class VGG19J(VGGlike):
    def __init__(
        self,
        out_channels: int = 2,
        cut_layers: int = 1,
        down_conv_z: bool = False,
        up_conv_z2: bool = False
    ):
        """
        A VGG19-based version of UNet-like network.

        Args:
            out_channels: number of output channels. Defaults to 2.
            cut_layers: number of last VGG layers to be ignored. Good values are 1, 10, 19
            down_conv_z: if True, the convolution kernels in DownBlocks are 3x3x3; otherwise 3x3x1 (slice-wise)
            up_conv_z2: if True, both convolutional layers in each UpBlock have 3x3x3 kernel size; otherwise there is
                one layer with 3x3x3 and one layer with 3x3x1

        """
        super().__init__()

        base = vgg19(pretrained=True)
        anchors = {3, 8, 17, 26, 35}

        self._build_vgg_like(anchors, base, cut_layers, down_conv_z, out_channels, up_conv_z2)


def main():
    device = 'cpu'
    m = VGG16J(cut_layers=1, down_conv_z=False, up_conv_z2=True).to(device)
    summary(m, (1, 192, 192, 32), device=device)
    x = torch.rand((1, 1, 192, 192, 32), device=device)
    out = m(x)
    make_dot(out).render('vgg16j', view=True)


if __name__ == '__main__':
    main()
