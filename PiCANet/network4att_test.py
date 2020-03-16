# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time

cfg = {'PicaNet': "GGLLL",
       'Size': [28, 28, 28, 56, 112, 224],
       'Channel': [1024, 512, 512, 256, 128, 64],
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}


class Unet(nn.Module):
    def __init__(self, cfg={'PicaNet': "GGLLL",
                            'Size': [28, 28, 28, 56, 112, 224],
                            'Channel': [1024, 512, 512, 256, 128, 64],
                            'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}):
        super(Unet, self).__init__()
        self.encoder = Encoder()
        self.decoder = nn.ModuleList()
        self.cfg = cfg
        for i in range(5):
            assert cfg['PicaNet'][i] == 'G' or cfg['PicaNet'][i] == 'L'
            self.decoder.append(
                DecoderCell(size=cfg['Size'][i],
                            in_channel=cfg['Channel'][i],
                            out_channel=cfg['Channel'][i + 1],
                            mode=cfg['PicaNet'][i]))
        self.decoder.append(DecoderCell(size=cfg['Size'][5],
                                        in_channel=cfg['Channel'][5],
                                        out_channel=1,
                                        mode='C'))

    def forward(self, *input):
        if len(input) == 2:
            x = input[0]
            tar = input[1]
            test_mode = False
        elif len(input) == 3:
            x = input[0]
            tar = input[1]
            test_mode = input[2]
        elif len(input) == 1:
            x = input[0]
            tar = None
            test_mode = True
        else:
            assert 0
        en_out = self.encoder(x)
        dec = None
        pred = []
        attention = []
        for i in range(6):
            # print(En_out[5 - i].size())
            dec, _pred, _attention = self.decoder[i](en_out[5 - i], dec)
            pred.append(_pred)
            if _attention is not None:
                attention.append(_attention)
        loss = 0
        if not test_mode:
            for i in range(6):
                loss += F.binary_cross_entropy(pred[5 - i], tar) * self.cfg['loss_ratio'][5 - i]
                # print(float(loss))
                if tar.size()[2] > 28:
                    tar = F.max_pool2d(tar, 2, 2)
        return pred, loss, attention


def make_layers(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        configure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'm', 512, 512, 512, 'm']
        self.seq = make_layers(configure, 3)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6 in paper
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, 1)  # fc7 in paper

    def forward(self, *input):
        x = input[0]
        conv1 = self.seq[:4](x)
        conv2 = self.seq[4:9](conv1)
        conv3 = self.seq[9:16](conv2)
        conv4 = self.seq[16:23](conv3)
        conv5 = self.seq[23:](conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)

        return conv1, conv2, conv3, conv4, conv5, conv7


class DecoderCell(nn.Module):
    def __init__(self, size, in_channel, out_channel, mode):
        super(DecoderCell, self).__init__()
        self.bn_en = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'G':
            self.picanet = PicanetG(size, in_channel)
        elif mode == 'L':
            self.picanet = PicanetL(in_channel)
        elif mode == 'C':
            self.picanet = None
        else:
            assert 0
        if not mode == 'C':
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        else:
            self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0)

    def forward(self, *input):
        assert len(input) <= 2
        if input[1] is None:
            en = input[0]
            dec = input[0]  # not specified in paper
        else:
            en = input[0]
            dec = input[1]

        if dec.size()[2] * 2 == en.size()[2]:
            dec = F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        elif dec.size()[2] != en.size()[2]:
            assert 0
        en = self.bn_en(en)
        en = F.relu(en)
        fmap = torch.cat((en, dec), dim=1)  # F
        fmap = self.conv1(fmap)
        fmap = F.relu(fmap)
        if not self.mode == 'C':
            # print(fmap.size())
            fmap_att, attention = self.picanet(fmap)  # F_att
            x = torch.cat((fmap, fmap_att), 1)
            x = self.conv2(x)
            x = self.bn_feature(x)
            dec_out = F.relu(x)
            _y = self.conv3(dec_out)
            _y = torch.sigmoid(_y)
        else:
            dec_out = self.conv2(fmap)
            _y = torch.sigmoid(dec_out)
            attention = None

        return dec_out, _y, attention


class PicanetG(nn.Module):
    def __init__(self, size, in_channel):
        super(PicanetG, self).__init__()
        self.renet = Renet(size, in_channel, 100)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1)
        # print(kernel.size())
        x = F.unfold(x, [10, 10], dilation=[3, 3])
        x = x.reshape(size[0], size[1], 10 * 10)
        kernel = kernel.reshape(size[0], 100, -1)
        x = torch.matmul(x, kernel)
        x = x.reshape(size[0], size[1], size[2], size[3])

        # for attention visualization

        # print(torch.cuda.memory_allocated() / 1024 / 1024)
        attention = kernel.data
        attention = attention.requires_grad_(False)
        attention = torch.reshape(attention, (size[0], -1, 10, 10))
        # attention = F.conv_transpose2d(torch.ones((1, 1, 1, 1)).cuda(), attention, dilation=3)
        attention = F.interpolate(attention, 224, mode='bilinear', align_corners=True)
        # attention = F.interpolate(attention, 224, mode='area')
        attention = torch.reshape(attention, (size[0], size[2], size[3], 224, 224))
        return x, attention


class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        attention = kernel.data
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        # print("Before unfold", x.shape)
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        # print("After unfold", x.shape)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        # print(x.shape, kernel.shape)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])

        # attention = kernel.data
        attention = attention.requires_grad_(False)
        # attention = torch.reshape(attention, (size[0], -1, 7, 7))
        attention = torch.reshape(attention, (size[0], -1, 7, 7))
        # attention = F.conv_transpose2d(torch.ones((1, 1, 1, 1)).cuda(), attention, dilation=2)
        attention = F.interpolate(attention, int(12 * 224 / size[2] + 1), mode='bilinear', align_corners=True)
        # attention = F.interpolate(attention, int(12 * 224 / size[2] + 1), mode='area')
        attention = torch.reshape(attention,
                                  (size[0], size[2], size[3], int(12 * 224 / size[2] + 1), int(12 * 224 / size[2] + 1)))
        # attention = attention.permute(0, 2, 1, 3, 4)
        # attention = attention.permute(0, 1, 2, 4, 3)
        return x, attention


class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                                  bidirectional=True)  # each column
        self.conv = nn.Conv2d(512, out_channel, 1)
        # self.fc = nn.Linear(512 * size * size, 10)

    def forward(self, *input):
        x = input[0]
        temp = []
        size = x.size()  # batch, in_channel, height, width
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        # x = torch.reshape(x, (-1, 512 * self.size * self.size))
        x = self.conv(x)
        return x


if __name__ == '__main__':
    vgg = torchvision.models.vgg16(pretrained=True)
    # model = Encoder()
    # model.seq.load_state_dict(vgg.features.state_dict())
    # print(model.state_dict().keys())
    # print(vgg.features.state_dict().keys())
    # print(vgg.features)
    device = torch.device("cuda")
    batch_size = 1
    noise = torch.randn((batch_size, 3, 224, 224)).type(torch.cuda.FloatTensor)
    target = torch.randn((batch_size, 1, 224, 224)).type(torch.cuda.FloatTensor)

    # print(vgg.features(noise))
    # print(model(noise))
    # print(model.seq)
    # print(vgg.features)
    # print(F.mse_loss(model.seq[:8](noise), vgg.features[:8](noise)))
    model = Unet(cfg).cuda()
    model.encoder.seq.load_state_dict(vgg.features.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    print('Time: {}'.format(time.clock()))
    _, loss = model(noise, target)
    loss.backward()
    """
    for i in range(1000):
        opt.zero_grad()
        time_spend = time.clock()
        _, loss = model(noise, target)
        print('Time_Spend: {}'.format(time.clock() - time_spend))
        loss.backward()
        opt.step()

        print(float(loss))
        print('Time: {}'.format(time.clock()))
    """
