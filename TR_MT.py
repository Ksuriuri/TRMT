from torchvision import models
from torch.autograd import Variable
import torch.optim as optim
from os.path import exists, join
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from ops import *
from glob import glob

img_w = 32  # 160 32
img_h = 32  # 192 32
num_epochs = 200  # 200
batch_size = 64  # 128
patience = num_epochs  # 100
repeat_num = 5  # 10
learning_rate = 1e-3  # 1e-5
# dropout = 0.5

fold_start = 1
fold_end = 4
data_path = r'/home/public/Documents/hhy/ivim_new/data/fold/mvi2.npz'
save_path = r'/home/public/Documents/hhy/ivim_new/result_new/new_experiment/tfm_base_multi_net_share_2_4_1'
all_bmap_path = r'/home/public/Documents/hhy/ivim_new/data/mvi2'
bmap_save_path = r'/home/public/Documents/hhy/ivim_new/result_new/new_experiment/tfm_base_multi_net_share_b_2_4_1.npz'
is_save_all = True
if not exists(save_path):
    os.makedirs(save_path)

# tfm_base_multi_net_1_2: sc + sf; weight of loss_rec set to 1e-2; lr set to 1e-3, decay 5e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18 = torch.load('./model/resnet18-5c106cde.pth')

# define b values
b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000]).astype(np.float32)
b_fit = np.expand_dims(b_values, -1)  # .repeat(batch_size, axis=0)
b_fit = Variable(torch.from_numpy(b_fit).to(device))
# b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
# b_fit = torch.FloatTensor(b_values)
print(b_values.shape)


def evaluate(pred_choice, target, p_):
    TP = ((pred_choice == 1) & (target.data == 1)).cpu().sum()
    TN = ((pred_choice == 0) & (target.data == 0)).cpu().sum()
    FN = ((pred_choice == 0) & (target.data == 1)).cpu().sum()
    FP = ((pred_choice == 1) & (target.data == 0)).cpu().sum()

    # p = TP / (TP + FP + 1e-8)
    # r = TP / (TP + FN + 1e-8)
    # F1 = 2 * r * p / (r + p)

    sen = TP / (TP + FN + 1e-8)
    spe = TN / (TN + FP + 1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN)

    pre_, lb_ = [], []
    p_ = nn.Softmax(dim=1)(p_)
    for ii in range(target.shape[0]):
        lb_.append(target[ii].item())
        pre_.append(p_[ii, 1].item())
    auc = roc_auc_score(lb_, pre_)

    return acc, sen, spe, auc


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)


def loaddata():
    train_data = np.load(data_path)['train_data'].transpose([0, 3, 1, 2]).astype(np.float32)
    train_lb = np.load(data_path)['train_lb'].astype(np.float32).argmax(1)
    test_data = np.load(data_path)['test_data'].transpose([0, 3, 1, 2]).astype(np.float32)
    test_lb = np.load(data_path)['test_lb'].astype(np.float32).argmax(1)
    print(train_data.shape, ' ', test_data.shape)
    print(train_lb.shape, ' ', test_lb.shape)

    trainset = Mydataset(train_data, train_lb)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=2, drop_last=True)

    testset = Mydataset(test_data, test_lb)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_lb.shape[0], shuffle=False,
                                             num_workers=2, drop_last=True)
    return trainloader, testloader, train_data.shape[0] // batch_size


class TRMT(nn.Module):
    def __init__(self):
        super(TRMT, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.load_state_dict(resnet18)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, 2)

        self.d1 = DownsampleLayer(9, 64)  # 9-64
        self.d2 = DownsampleLayer(64, 128)  # 64-128
        self.d3 = DownsampleLayer(128, 256)  # 128-256
        self.d4 = DownsampleLayer(256, 512)  # 256-512

        self.u1 = UpSampleLayer(512, 512)  # 512-1024-512
        self.u2 = UpSampleLayer(1024, 256)  # 1024-512-256
        self.u3 = UpSampleLayer(512, 128)  # 512-256-128
        self.u4 = UpSampleLayer(256, 64)  # 256-128-64

        self.o = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )

        self.flattener = nn.Flatten(2, 3)
        self.positional_emb_c = Parameter(torch.zeros(1, 4, 512),
                                          requires_grad=True)
        init.trunc_normal_(self.positional_emb_c, std=0.2)
        self.positional_emb_f = Parameter(torch.zeros(1, 4, 512),
                                          requires_grad=True)
        init.trunc_normal_(self.positional_emb_f, std=0.2)
        self.drop_e = nn.Dropout(p=0.1)
        # self.e1 = TransformerEncoderLayer(d_model=512, nhead=2, dim_feedforward=128)
        self.ec = TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=128)
        self.ef = TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=128)
        self.e_share = TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=128)
        self.norm = LayerNorm(512)
        self.attention_pool = Linear(512, 1)
        # self.fc0 = nn.Linear(1024, 128)

    def forward(self, inputs):
        x_c = self.model.conv1(inputs)
        x_c = self.model.bn1(x_c)
        x_c = self.model.relu(x_c)
        x_c = self.model.maxpool(x_c)
        x_c = self.model.layer1(x_c)
        x_c = self.model.layer2(x_c)
        x_c = self.model.layer3(x_c)
        x_c = self.model.layer4(x_c)
        out_c = self.model.avgpool(x_c)
        out_c = out_c.view(out_c.size(0), -1)

        d_1, d1 = self.d1(inputs)
        d_2, d2 = self.d2(d1)
        d_3, d3 = self.d3(d2)
        d_4, d4 = self.d4(d3)

        x_c_flat = self.drop_e(self.flattener(x_c).transpose(-2, -1) + self.positional_emb_c)
        d4_flat = self.drop_e(self.flattener(d4).transpose(-2, -1) + self.positional_emb_f)
        sc1 = self.ec(x_c_flat, x_c_flat)
        sf1 = self.ef(d4_flat, d4_flat)
        sc2 = self.norm(self.e_share(sc1, sf1))
        sf2 = self.norm(self.e_share(sf1, sc1)).transpose(-1, -2).view(d4.size(0), 512, 2, 2)
        # s_ = torch.cat((sc, sf), -2)
        # s_ = sc + sf

        u1 = self.u1(d4 + sf2, d_4)
        u2 = self.u2(u1, d_3)
        u3 = self.u3(u2, d_2)
        u4 = self.u4(u3, d_1)
        x_f = self.o(u4)
        mask = (inputs[:, :1] > 0).float()
        params = torch.abs(x_f) * mask  # 64,32,32
        out_rec = torch.clamp(self.ivim_matmul(params) * inputs[:, :1], 0.0, 1.0)

        sp = torch.matmul(F.softmax(self.attention_pool(sc2), dim=1).transpose(-1, -2), sc2).squeeze(-2)
        # out_c = torch.cat((out_c, sp), 1)
        # out_c = self.fc0(out_c)
        out_c = out_c + sp
        out_c = self.model.fc(out_c)

        return out_c, out_rec, params

    def ivim_matmul(self, params):
        flat = params.view(params.size(0), 3, params.size(2) * params.size(3))
        dp = flat[:, 0].unsqueeze(1)
        dt = flat[:, 1].unsqueeze(1)
        fp = flat[:, 2].unsqueeze(1)
        b_fit_ = b_fit.unsqueeze(0).repeat(params.size(0), 1, 1)
        outputs = fp * torch.exp(-torch.bmm(b_fit_, dp)) + (1 - fp) * torch.exp(-torch.bmm(b_fit_, dt))
        outputs = outputs.view(params.size(0), b_values.shape[0], img_w, img_h)
        # print(outputs.shape)
        return outputs


def train():
    trainloader, testloader, num_batch = loaddata()

    test_img, test_lb = None, None
    for data in testloader:
        test_img, test_lb = data
        test_img, test_lb = Variable(test_img.to(device)), Variable(test_lb.to(device))

    model = TRMT()
    model = model.to(device)

    total = sum([param.nelement() for param in model.parameters()])
    print('params ', total, ' flops ', 0)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()

    l1_loss = nn.L1Loss(reduce=True, size_average=True)

    lc_, acc_ = [], []
    sen_, spe_, auc_ = [], [], []
    auc_max = 0
    for epoch in range(num_epochs):
        running_loss, running_loss_rec = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            model.train()
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs, outputs_rec, ivim_pre = model(inputs)
            _, predicted = torch.max(outputs, 1)
            acc = torch.eq(predicted, labels).sum().float().item() / labels.shape[0]
            loss_c = criterion(outputs, labels)
            loss_rec = l1_loss(outputs_rec, inputs)
            loss = loss_c + 1e-2 * loss_rec
            loss.backward()
            optimizer.step()
            running_loss += loss_c.item()
            running_loss_rec += loss_rec.item()
            if i == num_batch - 1:
                model.eval()
                with torch.no_grad():
                    outputs_t, _, _ = model(test_img)
                    loss_t = criterion(outputs_t, test_lb).item()
                    _, predicted_t = torch.max(outputs_t, 1)
                    # acc_t = torch.eq(predicted_t, test_lb).sum().float().item() / test_lb.shape[0]
                    acc_t, sen_t, spe_t, auc_t = evaluate(predicted_t, test_lb, outputs_t)
                    lc_.append(loss_t), acc_.append(acc_t)
                    sen_.append(sen_t), spe_.append(spe_t), auc_.append(auc_t)
                    print('[%3d, %3d] lc %.4f lrec %.4f acc %.4f\ttest: loss %.4f acc %.4f sen %.4f spe %.4f auc %.4f' %
                          (epoch + 1, i + 1, running_loss / num_batch, running_loss_rec / num_batch,
                           acc, loss_t, acc_t, sen_t, spe_t, auc_t))
                    running_loss, running_loss_rec = 0.0, 0.0

    print('Finished Training')

    model.eval()
    with torch.no_grad():
        b_files = glob(join(all_bmap_path, '*'))
        b_files.sort(key=lambda x: (int(x.split('/')[-1].split('.')[0].split('_')[-1])))
        test_data_org = []
        for idx, bf in enumerate(b_files):
            img = np.load(bf)['x']
            test_data_org.append(img)
        test_data_org = np.array(test_data_org).transpose([0, 3, 1, 2]).astype(np.float32)
        # b0 = test_data_org[..., :1]
        test_data = test_data_org
        dataset = Mydataset(test_data, np.ones((test_data.shape[0], 2)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=test_data.shape[0], shuffle=False,
                                                 num_workers=2, drop_last=True)
        ivim_pre_all, x_fit_pre_all = [], []
        for data in dataloader:
            test_img, test_lb = data
            test_img = Variable(test_img.to(device))
            _, outputs_rec, ivim_p = model(test_img)
            ivim_pre_all = ivim_p.cpu().detach().numpy().transpose([0, 2, 3, 1]).astype(np.float32)
            x_fit_pre_all = outputs_rec.cpu().detach().numpy().transpose([0, 2, 3, 1]).astype(np.float32)

        np.savez(bmap_save_path, ivim=ivim_pre_all, x=x_fit_pre_all)

    return lc_, acc_, sen_, spe_, auc_

# print(torch.__version__)
# print(torch.cuda.device_count())
# print(torch.cuda.is_available())


lc_fold, acc_fold = [], []
sen_fold, spe_fold, auc_fold = [], [], []
temp_path = data_path
for fold_ in range(fold_start - 1, fold_end):
    data_path = data_path.split('.')[0] + '_%d' % (fold_ + 1) + '.npz'
    lc_all, acc_all = [], []
    sen_all, spe_all, auc_all = [], [], []
    for _ in range(repeat_num):
        loss_c, acc_c, sen_c, spe_c, auc_c = train()
        lc_all.append(loss_c), acc_all.append(acc_c)
        sen_all.append(sen_c), spe_all.append(spe_c), auc_all.append(auc_c)
    if fold_end - fold_start + 1 == 1:
        acc_fold, lc_fold, sen_fold, spe_fold, auc_fold = acc_all, lc_all, sen_all, spe_all, auc_all
    else:
        lc_fold.append(lc_all), acc_fold.append(acc_all)
        sen_fold.append(sen_all), spe_fold.append(spe_all), auc_fold.append(auc_all)
    data_path = temp_path

if is_save_all:
    np.savez(join(save_path, r'result.npz'), acc=acc_fold, loss=lc_fold, sen=sen_fold, spe=spe_fold, auc=auc_fold)
