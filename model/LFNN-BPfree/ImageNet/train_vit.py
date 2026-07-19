#!/usr/bin/env python
# coding: utf-8
"""ViT-B/16 LFNN training script matching the released ImageNet configuration."""

# In[ ]:


'''Train ImageNet with PyTorch.'''
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from ViT import ViT
import torch.backends.cudnn as cudnn

from torchvision.models import vit_b_16, ViT_B_16_Weights

cudnn.benchmark = True

# Authoritative settings recorded in model/LFNN/vit_training_log.txt.
BATCH_SIZE = 256
NUM_WORKERS = 2
NUM_EPOCHS = 90
LOG_INTERVAL = 50
SCHEDULER_T_MAX = 200

# if torch.cuda.is_available():
#     device_ids = [0]
#     for device_id in device_ids:
#         torch.cuda.set_device(device_id)
# else:
#     device = 'cpu'

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device0)
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

temp_dir = 'temp'

# save temp pth
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

train_directory = './imagenet/train'
test_directory = './imagenet/val'

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.ImageFolder(root=train_directory, transform=transform_train)
testset = torchvision.datasets.ImageFolder(root=test_directory, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
#trainloader = tqdm(trainloader, total=len(trainloader))

testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# trainloader = tqdm(testloader, total=len(testloader))

def load_state_dict_ignore_mismatch(model, state_dict):
    model_dict = model.state_dict()

    matched_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    model_dict.update(matched_state_dict)

    model.load_state_dict(model_dict)

    return len(matched_state_dict), len(model_dict)



# load resnet101
net = ViT(num_classes=1000, image_size=224, patch_size=16, hidden_dim=768, num_heads=12, num_layers=12, mlp_dim=3072)
#net = ViT(num_classes=1000, img_size=224, patch_size=16, d_model=768, n_head=12, n_layers=12, d_mlp=3072)
import hashlib

def get_model_hash(model):
    md5 = hashlib.md5()
    for param in model.parameters():
        md5.update(param.data.cpu().numpy().tobytes())
    return md5.hexdigest()

# 加载预训练权重前的哈希值
initial_hash = get_model_hash(net)
print(f"Initial model hash: {initial_hash}")

pre = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
pretrained_state_dict = pre.state_dict()
#net.load_state_dict(state)
#pretrained_state_dict = torch.load('pretrained_image1k.pth')
#net.load_state_dict(pretrained_state_dict)
matched_params, total_params = load_state_dict_ignore_mismatch(net, pretrained_state_dict)
print(f"Loaded {matched_params} out of {total_params} parameters")

loaded_hash = get_model_hash(net)
print(f"Loaded model hash: {loaded_hash}")

# state_dict = torch.load('temp/vit_epoch1.pth')
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith("module."):
#         new_key = k[len("module."):]
#     else:
#         new_key = k
#     new_state_dict[new_key] = v
#net = nn.DataParallel(net, device_ids=device_ids)
if torch.cuda.device_count() > 1:
    #print(torch.cuda.device_count())
    net = nn.DataParallel(net)

net.to(device0)
model_ref = net.module if isinstance(net, nn.DataParallel) else net

criterion_1 = nn.CrossEntropyLoss().to(device0)

# Optimizer membership matches the released run. The patch embedding, class
# token, positional embedding, and final encoder norm were not optimized.
optimizer_1 = optim.SGD([
    {'params': model_ref.encoder.layers[:3].parameters()},
], lr=0.001, weight_decay=5e-4,momentum=0.9)  # update first layer

optimizer_2 = optim.SGD([
    {'params': model_ref.encoder.layers[3:6].parameters()},
], lr=0.001, weight_decay=5e-4,momentum=0.9)  # update second layer

optimizer_3 = optim.SGD([
     {'params': model_ref.encoder.layers[6:9].parameters()},
], lr=0.001, weight_decay=5e-4,momentum=0.9)  # update third layer

optimizer_4 = optim.SGD([
    {'params': model_ref.encoder.layers[9:].parameters()},
    {'params': model_ref.heads.parameters()},
], lr=0.001,weight_decay=5e-4,momentum=0.9)  # update fourth layer

scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=SCHEDULER_T_MAX)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=SCHEDULER_T_MAX)
scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=SCHEDULER_T_MAX)
scheduler_4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_4, T_max=SCHEDULER_T_MAX)



train_losses = []
test_losses = []

log_file = open("vit_training_log.txt", "w")
config_line = (
    f"ViT config: batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}, "
    f"cuda_devices={torch.cuda.device_count()}, train_batches={len(trainloader)}, "
    f"test_batches={len(testloader)}"
)
run_line = (
    f"Running ViT for {NUM_EPOCHS} epoch(s); batch_size={BATCH_SIZE}; "
    f"cuda_devices={torch.cuda.device_count()}"
)
print(config_line)
print(run_line)
log_file.write(config_line + "\n")
log_file.write(run_line + "\n")
log_file.flush()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device0), targets.to(device0)

        if targets.max() >= 1000 or targets.min() < 0:
            raise ValueError(f"Targets out of range: min {targets.min()}, max {targets.max()}")

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        optimizer_4.zero_grad()

        outputs, extra_1, extra_2, extra_3 = net(inputs)
        loss_1 = criterion_1(extra_1, targets)

        loss_2 = criterion_1(extra_2, targets)

        loss_3 = criterion_1(extra_3, targets)

        loss_4 = criterion_1(outputs, targets)

        loss = 0.5 * (loss_1 + loss_2 + loss_3) + loss_4
        loss.backward()

        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        optimizer_4.step()

        train_loss += loss_4.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % LOG_INTERVAL == 0:
            print(f'Step [{i}/{len(trainloader)}] | Loss: {loss.item():.4f}')
            log_file.write(f'Step [{i}/{len(trainloader)}] | Loss: {loss.item():.4f}\n')
            log_file.flush()
            print(f'Loss1:{loss_1.item():.4f}')
            print(f'Loss2:{loss_2.item():.4f}')
            print(f'Loss3:{loss_3.item():.4f}')
            print(f'Loss4:{loss_4.item():.4f}')
            log_file.write(f'loss1:{loss_1.item():.4f},loss2:{loss_2.item():.4f},loss3:{loss_3.item():.4f},loss4:{loss_4.item():.4f}\n ')
            log_file.flush()
    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / len(trainloader), 100. * correct / total, correct, total))
    log_file.write(
        f"Epoch {epoch}: Train Loss = {train_loss / len(trainloader):.3f}, Accuracy = {100. * correct / total:.3f}%")
    log_file.flush()
    train_losses.append(train_loss / len(trainloader))

def test(epoch):
    net.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            inputs, targets = inputs.to(device0), targets.to(device0)
            outputs, _, _, _ = net(inputs)
            loss = criterion_1(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.topk(5, 1)

            total += targets.size(0)
            correct_top1 += predicted[:, 0].eq(targets).sum().item()

            for i in range(targets.size(0)):
                if targets[i] in predicted[i]:
                    correct_top5 += 1

    top1_accuracy = 100. * correct_top1 / total
    top5_accuracy = 100. * correct_top5 / total

    print('Test Loss: %.3f | Top-1 Acc: %.3f%% | Top-5 Acc: %.3f%% (%d/%d)'
          % (test_loss / len(testloader), top1_accuracy, top5_accuracy, correct_top1, total))
    log_file.write(
        f" Test Loss = {test_loss / len(testloader):.3f}, Top-1 Accuracy = {top1_accuracy:.3f}%, Top-5 Accuracy = {top5_accuracy:.3f}%\n")
    log_file.flush()
    test_losses.append(test_loss / len(testloader))

for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    train(epoch)
    test(epoch)

    check_path = os.path.join('temp', f'vit_epoch{epoch + 1}.pth')
    torch.save(model_ref.state_dict(), check_path)

    scheduler_1.step()
    scheduler_2.step()
    scheduler_3.step()
    scheduler_4.step()

log_file.close()
# Save the trained weights
save_path = 'vit_4out_imagenet.pth'
torch.save(model_ref.state_dict(), save_path)
print("Trained weights saved to:", save_path)
