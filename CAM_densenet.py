from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import densenet121
import torchvision.transforms as transforms
import util
from train import DenseNet
# %matplotlib inline

class CAM(nn.Module):
    def __init__(self, model_to_convert, get_fc_layer=lambda m: m.classifier,score_fn=F.softmax, resize=True):
        super().__init__()
        self.backbone = nn.Sequential(*list(model_to_convert.children())[:-1])
        self.fc = get_fc_layer(model_to_convert)
        self.conv  =  nn.Conv2d(self.fc.in_features, self.fc.out_features, kernel_size=1)
        self.conv.weight = nn.Parameter(self.fc.weight.data.unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = self.fc.bias
        self.score_fn = score_fn
        self.resize = resize
        self.eval()
        
    def forward(self, x, out_size=None):
        batch_size, c, *size = x.size()
        feat = self.backbone(x)
        cmap = self.score_fn(self.conv(feat))
        if self.resize:
            if out_size is None:
                out_size = size
            cmap = F.upsample(cmap, size=out_size, mode='bilinear')
        pooled = F.adaptive_avg_pool2d(feat,output_size=1)
        flatten = pooled.view(batch_size, -1)
        cls_score = self.score_fn(self.fc(flatten))
        weighted_cmap =  (cmap*cls_score.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return cmap, cls_score, weighted_cmap

target_size = (224,224)
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.Scale(target_size),transforms.CenterCrop(target_size),
                                transforms.ToTensor(), normalize])



def load_model(args, load_path, nclasses):
    model = DenseNet(args, nclasses)
    model = densenet121(pretrained=False)
    model.load_state_dict(torch.load(load_path), strict=False)
    model = model.cuda()
    model.eval()
    print(type(model))
    return model


parser = util.get_parser()
args = parser.parse_args()
densenet = load_model(args, '/content/GP/val0.020903_train0.020714_epoch10', 5)
cam = CAM(densenet)
assert not cam.training

if torch.cuda.is_available():
    print("use gpu")
    cam = cam.cuda()
    def to_var(x, requires_grad=False, volatile=False):
        return Variable(x.cuda(), requires_grad=requires_grad, volatile=volatile)
else:
    def to_var(x, requires_grad=False, volatile=False):
        return Variable(x, requires_grad=requirs_grad, volatile=volatile)

path = '/content/images/00000025_000.png'
img = Image.open(path).convert('RGB')
img_v = to_var(transform(img).unsqueeze(0),volatile=True)

cmap, score, weighted_cmap = cam(img_v)
print(cmap.size())
print(score.size())
print(weighted_cmap.size())


background = np.array(img.resize(target_size))
color_map = weighted_cmap.data.cpu().numpy()[0]
# color_map = cmap.data.cpu().numpy()[0,55]
plt.imshow(background)
plt.imshow(color_map,alpha=0.4)
plt.savefig('myfig.png')