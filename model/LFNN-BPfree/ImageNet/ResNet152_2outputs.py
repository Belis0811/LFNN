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
import ResNet2 as ResNet
import torch.backends.cudnn as cudnn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True

if torch.cuda.is_available():
    device_ids = [0]
    for device_id in device_ids:
        torch.cuda.set_device(device_id)
else:
    device = 'cpu'

device0 = torch.device('cuda:0')

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
    trainset, batch_size=118, shuffle=True, num_workers=5, pin_memory=True)
#trainloader = tqdm(trainloader, total=len(trainloader))

testloader = torch.utils.data.DataLoader(
    testset, batch_size=118, shuffle=False, num_workers=5, pin_memory=True)

# trainloader = tqdm(testloader, total=len(testloader))

# load resnet152
net = ResNet.ResNet152()

net = nn.DataParallel(net, device_ids=device_ids)

net.to(device0)

criterion_1 = nn.CrossEntropyLoss().to(device0)

# define optimizer and loss function
optimizer_1 = optim.SGD([
    {'params': net.module.conv1.parameters()},
    {'params': net.module.bn1.parameters()},
    {'params': net.module.layer1.parameters()},
    {'params': net.module.layer2.parameters()},
    {'params': net.module.fc2.parameters()}
], lr=0.001, weight_decay=5e-4,momentum=0.9)  # update first two layer

optimizer_4 = optim.SGD([
    {'params': net.module.layer3.parameters()},
    {'params': net.module.layer4.parameters()},
    {'params': net.module.fc.parameters()}
], lr=0.001,weight_decay=5e-4,momentum=0.9)  # update layer3 and 4

scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=100)
scheduler_4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_4, T_max=100)


train_losses = []
test_losses = []

log_file = open("152_2out_training_log.txt", "w")


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device0), targets.to(device0)

        optimizer_1.zero_grad()
        optimizer_4.zero_grad()

        outputs, extra_1 = net(inputs)
        loss_1 = criterion_1(extra_1, targets)

        #loss_2 = criterion_1(extra_2, targets)

        #loss_3 = criterion_1(extra_3, targets)

        loss_4 = criterion_1(outputs, targets)

        loss = 0.5*loss_1 + loss_4
        loss.backward()

        optimizer_1.step()
        #optimizer_2.step()
        #optimizer_3.step()
        optimizer_4.step()

        train_loss += loss_4.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if i % 50 == 0:
            print(f'Step [{i}/{len(trainloader)}] | Loss: {loss.item():.4f}')
            log_file.write(f'Step [{i}/{len(trainloader)}] | Loss: {loss.item():.4f}\n')
            log_file.flush()
            print(f'Loss1:{loss_1.item():.4f}')
            #print(f'Loss2:{loss_2.item():.4f}')
            #print(f'Loss3:{loss_3.item():.4f}')
            print(f'Loss4:{loss_4.item():.4f}')
            log_file.write(f'loss1:{loss_1.item():.4f},loss4:{loss_4.item():.4f}\n ')
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
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device0), targets.to(device0)
            outputs, _ = net(inputs)
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


for epoch in range(start_epoch, start_epoch + 100):
    train(epoch)
    test(epoch)
    if epoch % 5 == 0:
        check_path = os.path.join('temp', f'ResNet152_2out_epoch{epoch + 1}.pth')
        torch.save(net.state_dict(), check_path)
    
    scheduler_1.step()
    #scheduler_2.step()
    #scheduler_3.step()
    scheduler_4.step()

log_file.close()
# Save the trained weights
save_path = 'resnet152_2out_imagenet.pth'
torch.save(net.state_dict(), save_path)
print("Trained weights saved to:", save_path)
