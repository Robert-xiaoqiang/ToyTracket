import torch  # pytorch 0.4.0! fft
import torch.nn as nn
import numpy as np
import pdb


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.yf = config.yf.clone()
        self.lambda0 = config.lambda0

    def forward(self, z, x, label):
        # z = self.feature(z)
        # x = self.feature(x)
        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)

        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
                  
        alphaf = label.to(device=z.device) / (kzzf + self.lambda0)  
        #alphaf = self.yf.to(device=z.device) / (kzzf + self.lambda0) # very Ugly
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
           
        return response


class DCFNetCollection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.p0net = DCFNet(config)
        self.p1net = DCFNet(config)
        self.p2net = DCFNet(config)
        self.p3net = DCFNet(config)

        self._initialize_weight()

    def get_feature(self, template, search1, search2):
        feature_t = [ ]
        feature_s1 = [ ]
        feature_s2 = [ ]
        for i in range(4):
            feature_t.append(getattr(self, 'p{}net'.format(i)).feature(template))
        for i in range(4):
            feature_s1.append(getattr(self, 'p{}net'.format(i)).feature(search1))
        for i in range(4):
            feature_s2.append(getattr(self, 'p{}net'.format(i)).feature(search2))
        return feature_t, feature_s1, feature_s2

    def forward(self, z, x, label):
        assert len(z) == len(x) == len(label) == 4, 'differ in length of input feature'
        feature = [ ]
        for i in range(len(z)):
            feature.append(getattr(self, 'p{}net'.format(i))(z[i], x[i], label[i]))
        return tuple(feature)

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class TopModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        self.config = config
        self.model = DCFNetCollection(self.config)

        # initial_y is the same for 4 points now
        self.initial_y = torch.FloatTensor(self.config.y.copy()).cuda()
        label = self.config.yf.repeat(self.args.replicata_batch_size, 1, 1, 1, 1).cuda(non_blocking=True)
        self.label = [ label ] * 4

    @staticmethod
    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def reponse_to_fake_yf(self, response):
        assert len(response) == 4, 'missing response, expected 4'
        ret = [ ]
        for cur_point_response in response:
            value, index = torch.max(cur_point_response.view(self.args.replicata_batch_size, -1), dim = 1) # B
            fake_y = torch.zeros((self.args.replicata_batch_size, 1, self.config.output_sz, self.config.output_sz)).cuda()
            r_max, c_max = TopModel.unravel_index(index, [self.config.output_sz, self.config.output_sz])
            for j in range(self.args.replicata_batch_size):
                shift_y  = torch.roll(self.initial_y, r_max[j].int().item(), dims = 0)
                fake_y[j, 0] = torch.roll(shift_y, c_max[j].int().item(), dims = 1)
            fake_yf = torch.rfft(fake_y.view(self.args.replicata_batch_size, 1, self.config.output_sz, self.config.output_sz), signal_ndim = 2)
            fake_yf = fake_yf.cuda(non_blocking=True)
            ret.append(fake_yf)
        return tuple(ret)

    def forward(self, template, search1, search2):
        template_feat, search1_feat, search2_feat = self.model.get_feature(template, search1, search2)
        # forward tracking 1
        with torch.no_grad():
            s1_response = self.model(template_feat, search1_feat, self.label)

        fake_yf = self.reponse_to_fake_yf(s1_response)

        # forward tracking 2
        with torch.no_grad():
            s2_response = self.model(search1_feat, search2_feat, fake_yf)

        fake_yf = self.reponse_to_fake_yf(s2_response)
    
        # backward tracking
        output = self.model(search2_feat, template_feat, fake_yf)

        return output

if __name__ == '__main__':

    # network test
    net = DCFNet()
    net.eval()



