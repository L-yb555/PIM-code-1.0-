from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class EMGNet(nn.Module):

    def __init__(self, args):
        super(EMGNet, self).__init__()
        self.num_channels = args.num_channels
        self.num_forces = args.num_forces
        self.num_force_levels = args.num_force_levels
        self.num_frames = args.num_frames
        self.num_frequencies = args.num_frequencies
        self.window_length = args.window_length
        self.hop_length = args.hop_length
        # STFT
        hann_window = torch.hann_window(self.window_length)
        if args.cuda:
            hann_window = hann_window.cuda()
        self.hann_window = hann_window
        # Layers
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        self.linear = nn.Linear(32, self.num_force_levels * self.num_forces)

    def _encoder(self):
        layers = []
        layers.append(self._encoder_block(self.num_channels, 32, (2, 4)))
        layers.append(self._encoder_block(32, 128, (2, 4)))
        layers.append(self._encoder_block(128, 256, (2, 4)))
        return nn.Sequential(*layers)

    def _encoder_block(self, in_channel, out_channel, downsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=downsample, stride=downsample))
        return nn.Sequential(*layers)

    def _decoder(self):
        layers = []
        layers.append(self._decoder_block(256, 256, (self.num_frames // 4, 1)))
        layers.append(self._decoder_block(256, 128, (self.num_frames // 2, 1)))
        layers.append(self._decoder_block(128, 32, (self.num_frames, 1)))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channel, out_channel, upsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Upsample(size=upsample, mode='bilinear', align_corners=False))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size * self.num_channels, x.size(2))
        x = torch.stft(x, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, window=self.hann_window,
                       center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        x = x.view(batch_size, self.num_channels, x.size(1), x.size(2))
        x = x.transpose(2, 3)[..., :self.num_frequencies]
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), self.num_force_levels, self.num_forces, x.size(2))
        return x
    
    def forward_feature(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size * self.num_channels, x.size(2))
        x = torch.stft(x, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, window=self.hann_window,
                       center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        x = x.view(batch_size, self.num_channels, x.size(1), x.size(2))
        x = x.transpose(2, 3)[..., :self.num_frequencies]

        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(x.size(0), -1)
    
class EMGNet_pose_pps(nn.Module):

    def __init__(self, args):
        super(EMGNet_pose_pps, self).__init__()
        self.num_channels = args.num_channels
        self.num_angles = args.num_angles
        self.num_forces = args.num_forces
        self.num_force_levels = args.num_force_levels
        self.num_frames = args.num_frames
        self.num_frequencies = args.num_frequencies
        self.window_length = args.window_length
        self.hop_length = args.hop_length
        # STFT
        hann_window = torch.hann_window(self.window_length)
        hann_window = hann_window.cuda()
        self.hann_window = hann_window
        # Layers
        self.encoder = self._encoder_emg()
        self.decoder = self._decoder()
        
        self.mlp = self._mlp_manus()
        self.emg_pose_out = self._emg_pose_out()
        self.emg_out = self._emg_out()
        self.linear_output = nn.Linear(128, self.num_force_levels * self.num_forces)

    def _encoder_emg(self):
        layers = []
        layers.append(self._encoder_block_emg(self.num_channels, 32, (2, 4)))
        layers.append(self._encoder_block_emg(32, 128, (2, 4)))
        layers.append(self._encoder_block_emg(128, 256, (2, 4)))
        return nn.Sequential(*layers)

    def _encoder_block_emg(self, in_channel, out_channel, downsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=downsample, stride=downsample))
        return nn.Sequential(*layers)

    def _decoder(self):
        layers = []
        layers.append(self._decoder_block(256, 512, (self.num_frames // 4, 1)))
        layers.append(self._decoder_block(512, 256, (self.num_frames // 2, 1)))
        layers.append(self._decoder_block(256, 256, (self.num_frames, 1)))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channel, out_channel, upsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Upsample(size=upsample, mode='bilinear', align_corners=False))
        return nn.Sequential(*layers)

    def _mlp_manus(self):
        layers = []
        layers.append(nn.Linear(self.num_angles * 1248 , 1024)) # 32 
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(1024, 512)) 
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(512, 64*32))
        return nn.Sequential(*layers)

    def _emg_out(self):
        layers = []
        layers.append(nn.Linear(256, 512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512, 512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512, 512))
        return nn.Sequential(*layers)

    def _emg_pose_out(self):
        layers = []
        layers.append(nn.Linear((512+64), 512)) 
        layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512, 256))
        layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 128))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        batch_size = x1.size(0)
 
        x1 = x1.reshape(batch_size * self.num_channels, x1.size(2))
        x1 = torch.stft(x1, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, window=self.hann_window,
                       center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        x1 = x1.view(batch_size, self.num_channels, x1.size(1), x1.size(2))
        x1 = x1.transpose(2, 3)[..., :self.num_frequencies]
        
        x1 = self.encoder(x1)
        x1 = self.decoder(x1)
        x1 = x1.view(x1.size(0), x1.size(1), x1.size(2))
        x1 = x1.transpose(1, 2)
        x1 = self.emg_out(x1) 
        x_emg = x1.transpose(1, 2)

        x2 = x2.reshape(x2.size(0), x2.size(1) * x2.size(2)) 
        x2 = self.mlp(x2)
        x_pose = x2.reshape(x2.size(0), -1, 32)
        x_emg_pose = torch.cat((x_emg, x_pose), dim=1)
        x_emg_pose = x_emg_pose.transpose(1, 2)
        x_emg_pose = self.emg_pose_out(x_emg_pose)   
        x_emg_pose = self.linear_output(x_emg_pose)
        x_emg_pose = x_emg_pose.transpose(1, 2)

        x_emg_pose = x_emg_pose.view(x_emg_pose.size(0), self.num_force_levels, self.num_forces, x_emg_pose.size(2))
        return x_emg_pose


class EMGNet_angles(nn.Module):

    def __init__(self, args):
        super(EMGNet_angles, self).__init__()
        self.num_channels = args.num_channels
        self.num_forces = args.num_forces
        self.num_frames = args.num_frames
        self.num_frequencies = args.num_frequencies
        self.window_length = args.window_length
        self.hop_length = args.hop_length
        # STFT
        hann_window = torch.hann_window(self.window_length)
        if args.cuda:
            hann_window = hann_window.cuda()
        self.hann_window = hann_window
        # Layers
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        self.linear = nn.Linear(64, self.num_forces)

    def _encoder(self):
        layers = []
        layers.append(self._encoder_block(self.num_channels, 32, (2, 4)))
        layers.append(self._encoder_block(32, 128, (2, 4)))
        layers.append(self._encoder_block(128, 256, (2, 4)))
        return nn.Sequential(*layers)

    def _encoder_block(self, in_channel, out_channel, downsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=downsample, stride=downsample))
        return nn.Sequential(*layers)

    def _decoder(self):
        layers = []
        layers.append(self._decoder_block(256, 256, (self.num_frames // 4, 1)))
        layers.append(self._decoder_block(256, 128, (self.num_frames // 2, 1)))
        layers.append(self._decoder_block(128, 64, (self.num_frames, 1)))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channel, out_channel, upsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Upsample(size=upsample, mode='bilinear', align_corners=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size * self.num_channels, x.size(2))
        x = torch.stft(x, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, window=self.hann_window,
                       center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        x = x.view(batch_size, self.num_channels, x.size(1), x.size(2))
        x = x.transpose(2, 3)[..., :self.num_frequencies]
        x = self.encoder(x)
        
        x_pps = self.decoder(x)
        x_pps = x_pps.view(x_pps.size(0), x_pps.size(1), x_pps.size(2))
        x_pps = x_pps.transpose(1, 2)
        x_pps = self.linear(x_pps)
        x_pps = x_pps.view(x_pps.size(0), self.num_forces, -1)
        return x_pps


class Pose_pps(nn.Module):

    def __init__(self, args):
        super(Pose_pps, self).__init__()
        self.num_angles = args.num_angles
        self.num_forces = args.num_forces
        self.num_force_levels = args.num_force_levels
        self.num_frames = args.num_frames
        self.num_frequencies = args.num_frequencies
        self.window_length = args.window_length
        self.hop_length = args.hop_length

        # Layers
        self.mlp = self._mlp_manus()
        self.linear_mlp = nn.Linear(128, 128)
        self.linear_output = nn.Linear(2240, self.num_force_levels * self.num_forces)

    def _mlp_manus(self):
        layers = []
        layers.append(nn.Linear(self.num_angles, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(64, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(256, 512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(512, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(256, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(64, 32))
        return nn.Sequential(*layers)

    def forward(self, x2):
        x2 = x2.transpose(1, 2)       
        x2 = self.mlp(x2)
        x2 = x2.transpose(1, 2)
        x_emg_pose = self.linear_output(x2)

        x_emg_pose = x_emg_pose.view(x_emg_pose.size(0), self.num_force_levels, self.num_forces, 32)
        return x_emg_pose


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def forward_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        return x


class NewModel(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
        self.emgnet = EMGNet(args)
        self.fc1 = nn.Linear(32 * 32, 512)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, self.emgnet.num_force_levels * self.emgnet.num_forces * 32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, emg, pose):
        emg_feature = self.emgnet.forward_feature(emg)
        emg_feature = self.fc1(emg_feature)
        pose_feature = self.resnet.forward_feature(pose)

        feature = torch.cat((emg_feature, pose_feature), -1)
        feature = self.relu(self.bn2(self.fc2(feature)))
        feature = self.relu(self.bn3(self.fc3(feature)))
        output = self.fc4(feature)
        output = output.view(output.size(0), self.emgnet.num_force_levels, self.emgnet.num_forces, 32)
        return output

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def forward_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        return x


class NewModel_3d(nn.Module):
    def __init__(self, args, **kwargs):
        super(NewModel_3d, self).__init__()
        self.num_forces = args.num_forces
        self.num_force_levels = args.num_force_levels
        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
        self.fc1 = nn.Linear(32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256) 
        self.fc4 = nn.Linear(256, self.num_force_levels * self.num_forces * 32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pose):
        # emg_feature = self.emgnet.forward_feature(emg)
        # emg_feature = self.fc1(emg_feature)
        pose_feature = self.resnet.forward_feature(pose)

        # feature = torch.cat((pose_feature, pose_feature), -1)
        feature = self.relu(self.bn2(self.fc2(pose_feature)))
        feature = self.relu(self.bn3(self.fc3(feature)))
        output = self.fc4(feature)
        output = output.view(output.size(0), self.num_force_levels, self.num_forces, 32)
        return output

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


'''
# 可视化结构图
from torchviz import make_dot

# 实例化模型（需要构造 args）
class Args:
    num_channels = 8
    num_forces = 10
    num_force_levels = 2
    num_frames = 32
    num_frequencies = 128
    window_length = 256
    hop_length = 32
    cuda = False

args = Args()

model = NewModel(args)

# 创建虚拟输入
emg_input = torch.randn(1, args.num_channels, 2048)  # batch_size=1, channels=8, length=2048
pose_input = torch.randn(1, 3, 16, 112, 112)  # 根据你的 ResNet 配置调整
model.eval()
# 前向传播 + 生成图
y = model(emg_input, pose_input)

# 获取模型的类名
model_name = model.__class__.__name__

# 生成计算图
dot = make_dot(y, params=dict(model.named_parameters()))

# 保存为 PDF 文件，文件名基于模型名称
dot.render(f"{model_name}", format="pdf")
print(f"Saved {model_name} structure to {model_name}.pdf")

'''


if __name__ == "__main__":


     '''
    def forward(self, x):
        batch_size = x.size(0)
        # x = x.view(batch_size * self.num_channels, x.size(2))
        x = x.reshape(batch_size * self.num_channels, x.size(2))
        x = torch.stft(x, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, window=self.hann_window,
                       center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        x = x.view(batch_size, self.num_channels, x.size(1), x.size(2))
        x = x.transpose(2, 3)[..., :self.num_frequencies]
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), self.num_force_levels, self.num_forces, x.size(2))
        return x
'''

