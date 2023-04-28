import torch
import numpy as np
from PIL import Image
import random
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from model import *
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='CORSD CIFAR Training')
parser.add_argument('--model', default="resnet18", type=str, help="resnet18|resnet34|resnet50|resnet101|resnet152|")
parser.add_argument('--channel', default=64, type=int, help="feature channel in the first resblock")
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100|cifar10")
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--epoch', default=100, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.16, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.01, type=float) 
parser.add_argument('--dataset_path', default="data", type=str)
parser.add_argument('--temperature', default=3.0, type=float)
parser.add_argument('--batchsize', default=43, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
parser.add_argument('--alpha', default=0.01, type=float)   
parser.add_argument('--margin', default=1, type=float)
parser.add_argument('--lg_loss', default=0.01, type=float)
parser.add_argument('--task_loss_coefficient', default=0.2, type=float)
parser.add_argument('--task_loss_coefficient_smooth', default=0.05, type=float)
parser.add_argument('--loss_coefficient_smooth', default=0.04, type=float)
args = parser.parse_args()
print(args)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/args.temperature, dim=1)
    softmax_targets = F.softmax(targets/args.temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_modified_dataset(base, num_classes):
    class Own_dataset(base):                  
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.target = [
                torch.nonzero(torch.tensor(self.targets) == i).view(-1)
                for i in range(num_classes)                                                 
            ]

        def __getitem__(self, idx):
            ret_data_list = []
            ret_target_list = []

            def _append(i):
                sample, target = super(Own_dataset, self).__getitem__(i)
                ret_data_list.append(sample)
                ret_target_list.append(target)

            def _get_idx(cls):
                rand = random.randint(0, len(self.target[cls])-1) 
                return self.target[cls][rand]

            def _append_random_sample(cls):
                _append(_get_idx(cls))

            _append(idx)

            _append_random_sample(self.targets[idx])

            while True:
                diff_cls = random.randint(0, num_classes-1)            
                if diff_cls != super().__getitem__(idx)[1]:
                    break
            
            _append_random_sample(diff_cls)
            
            return torch.stack(ret_data_list), torch.tensor(ret_target_list)

    return Own_dataset


def collect_fn(batch):
    data_lists, target_lists = zip(*batch)
    data_lists = torch.stack([x for lists in data_lists for x in lists])
    target_lists = torch.stack([x for lists in target_lists for x in lists])
    return data_lists, target_lists

if args.dataset == "cifar100":
    own_dataset = get_modified_dataset(torchvision.datasets.CIFAR100, 100)
    trainset = own_dataset(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=transform_train,
    )

    testset = torchvision.datasets.CIFAR100(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=transform_test,
    )

elif args.dataset == "cifar10":
    own_dataset = get_modified_dataset(torchvision.datasets.CIFAR10, 10)
    trainset = own_dataset(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=transform_test
    )

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=4,
    collate_fn=collect_fn
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=4,
    collate_fn=collect_fn
)

if args.model == "resnet18":
    net = resnet18(args)
if args.model == "resnet34":
    net = resnet34(args)
if args.model == "resnet50":
    net = resnet50(args)
if args.model == "resnet101":
    net = resnet101(args)
if args.model == "resnet152":
    net = resnet152(args)

net = nn.DataParallel(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.init_lr, steps_per_epoch=len(trainloader), epochs=args.epoch)


if __name__ == "__main__":
    set_seed(0)
    best_acc = 0
    logits_ema = torch.eye(args.num_classes, args.num_classes)
    logits_ema = logits_ema.to(device)
    for epoch in range(args.epoch):
        correct = 0
        predicted = []
        net.train()
        sum_ce_loss, sum_loss, total = 0.0, 0.0, 0.0
            for i, data in enumerate(trainloader, 0):
                length = len(trainloader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, outputs_feature = net(inputs)
                
                loss = torch.FloatTensor([0.]).to(device)
  
                ce_loss = criterion(outputs, labels)
                loss += ce_loss
  
                rn_nets = [net.module.rn4, net.module.rn3, net.module.rn2, net.module.rn1] 
  
                n, p = {}, {}
                n_outputs_logits, p_outputs_logits = {}, {}
            
                for index in range(0, 4):    
                    n_in = torch.stack([
                        torch.cat([outputs_feature[index][3*i+0], outputs_feature[index][3*i+2]])
                        for i in range(int(outputs_feature[index].size()[0]/3))
                    ])
                        
                    p_in = torch.stack([torch.cat([outputs_feature[index][3*i+0], outputs_feature[index][3*i+1]]) for i in range(int(outputs_feature[index].size()[0]/3))])
                       
                    rn_net = rn_nets[index]
                        
                    n[index], n_outputs_logits[index] = rn_net(n_in)
                    p[index], p_outputs_logits[index] = rn_net(p_in)
                   
                    loss += F.kl_div(p_outputs_logits[index], p_outputs_logits[0]) * args.loss_coefficient
                    loss += criterion(p_outputs_logits[index], labels[1::3]) * (args.task_loss_coefficient - args.loss_coefficient)
   
                    batch_size = labels.size(0) // 3
                    p_labels = torch.FloatTensor(batch_size, args.num_classes).to(device)
                    p_labels = p_labels.zero_().scatter_(1,labels[1::3].unsqueeze(1),1)
                    n_labels = torch.FloatTensor(batch_size, args.num_classes).to(device)
                    n_labels = n_labels.zero_().scatter_(1,labels[2::3].unsqueeze(1),1)
                    smooth_labels = (p_labels + n_labels) / 2
                    loss += F.kl_div(n_outputs_logits[index], n_outputs_logits[0]) * args.loss_coefficient_smooth
                    loss += CrossEntropy(n_outputs_logits[index], smooth_labels) * (args.task_loss_coefficient_smooth - args.loss_coefficient_smooth)
  
                    loss_triplet = torch.mean(torch.max(torch.cat(((p[index]-n[index] + args.margin),torch.tensor([0.0]).unsqueeze(0).repeat(int(outputs_feature[index].size()[0]/3),1).to(device)),1),dim=1)[0]) * args.alpha
                    
                    loss += loss_triplet
  
                for index in range(1, 4):   
                    loss_KD_n = (n[index] - n[0]).pow(2) * args.feature_loss_coefficient  
                    loss += loss_KD_n.mean()
                    loss_KD_p = (p[index] - p[0]).pow(2) * args.feature_loss_coefficient 
                    loss += loss_KD_p.mean()  
                        
                logits = nn.Softmax()(outputs)
                    
                for logit, label in zip(logits, labels):           
                    logits_ema[label] += logit.detach()                                                         
                    logits_combination = torch.cat((logit, logits_ema[label]),0)
                    logits_combination = logits_combination.unsqueeze(0)
                    logits_out = net.module.rn_logits(logits_combination)
                    loss += logits_out[0] * args.lg_loss  
                    rand = torch.randint(1,args.num_classes,[1]).item()
                    label = (label + rand)%(args.num_classes)                       
                    logits_combination = torch.cat((logit, logits_ema[label]),0)
                    logits_combination = logits_combination.unsqueeze(0)
                    logits_out = net.module.rn_logits(logits_combination)     
                    loss += -logits_out[0] * args.lg_loss
                    
  
                sum_loss += loss.item()
                sum_ce_loss += ce_loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += float(labels.size(0))
                
                _, predicted = torch.max(outputs.data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())
                
  
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.2f%% '
                        % (epoch + 1, (i + 1 + epoch * length), 100 * correct / total))
  
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            predicted = []
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, outputs_feature = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            print('Test Set AccuracyAcc:  %.4f%%'
                   % (100 * correct / total))

            if correct / total > best_acc:
                best_acc = correct/total
                print("Best Accuracy Updated: ", best_acc * 100)
                torch.save(net.state_dict(), "./checkpoints/"+str(args.model)+".pth")
        

    print("Training Finished, TotalEPOCH=%d, Best Accuracy=%.3f" % (args.epoch, best_acc))
