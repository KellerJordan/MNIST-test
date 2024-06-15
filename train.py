import os
import sys
import uuid
from math import ceil

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

#############################################
#                DataLoader                 #
#############################################

MNIST_MEAN = torch.tensor((0.1306,))
MNIST_STD = torch.tensor((0.3081,))

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class MnistLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, gpu=0):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10('/tmp', train=train, download=True)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device(gpu))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(MNIST_MEAN, MNIST_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        else:
            images = self.proc_images['norm']

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(32*32*3, dim)
        self.lin2 = nn.Linear(dim, 10)
    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


def train(train_loader):
    dim = 2048
    model = MLP(dim).cuda().half()
    optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9)

    base_lr = 0.4
    epochs = 5
    total_steps = epochs * len(train_loader)
    def get_lr(step):
        if step < 100: # warmup
            return base_lr * (step / 100)
        return base_lr * (1 - (step / total_steps))

    step = 0
    losses = []
    for _ in (range(epochs)):
        for inputs, labels in train_loader:

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            losses.append(loss.item())

    with torch.no_grad():
        inputs, labels = next(iter(val_loader))
        outputs = model(inputs)
        return outputs
    
N = 40000 # base train set size
def train0():
    train_loader = MnistLoader('cifar10', batch_size=1000, train=True, aug=dict(translate=2))
    train_loader.images = train_loader.images[:N]
    train_loader.labels = train_loader.labels[:N]
    return train(train_loader)

def train1(i):
    train_loader = MnistLoader('cifar10', batch_size=1000, train=True, aug=dict(translate=2))
    train_loader.images = torch.cat([train_loader.images[:N], train_loader.images[i:i+1]])
    train_loader.labels = torch.cat([train_loader.labels[:N], train_loader.labels[i:i+1]])
    return train(train_loader)

os.makedirs('cifar10', exist_ok=True)
loader0 = MnistLoader('cifar10', batch_size=1000, train=True)
val_loader = MnistLoader('cifar10', batch_size=10000, train=False)
val_loader.images = loader0.images[40000:50000]
val_loader.labels = loader0.labels[40000:50000]

outputs = torch.stack([train0() for _ in tqdm(range(1000))])

out_dir = 'logsc0'
os.makedirs(out_dir, exist_ok=True)
mu = outputs.float().mean(0)
torch.save(mu, '%s/%s.pt' % (out_dir, uuid.uuid4()))

