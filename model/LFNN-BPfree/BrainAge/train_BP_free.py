import numpy as np
import sys
import os
# from models_cnn import CNNSwinT
# from models import SwinT_4out
import model_pytorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from tqdm import tqdm
import csv
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math
import time

def seed_everything(seed=42):
    '''
    Set the seed for reproducibility.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def train_model(train_loader, test_loader, model, weight_dir, is_loaded, weight_path, start_epoch,
                criterion, loss_csv_path, num_epochs=100):
    start_time = time.time()
    save_interval = 10

    if is_loaded:
        weights = torch.load(weight_path)
        model.load_state_dict(weights)

    model.to(device)
    
    optimizer_1 = optim.AdamW([
        {'params': model.block1.parameters()},
        {'params': model.classifier1.parameters()}], 
                              lr=0.0001, weight_decay=0.01)

    optimizer_2 = optim.AdamW([
        {'params': model.block2.parameters()},
        {'params': model.classifier2.parameters()}], 
                              lr=0.0001, weight_decay=0.01)

    optimizer_3 = optim.AdamW([
        {'params': model.block3.parameters()},
        {'params': model.classifier3.parameters()}], 
                              lr=0.0001, weight_decay=0.01)

    optimizer_4 = optim.AdamW([
        {'params': model.block4.parameters()},
        {'params': model.classifier.parameters()}], 
                              lr=0.0001, weight_decay=0.01)

    # optimizer_1 = optim.AdamW([
    #     {'params': [param for name, param in model.named_parameters() if name.startswith('swinViT.patch_embed') or name.startswith('swinViT.layers1') or name.startswith('classifier1')]},
    # ], lr=0.0001, weight_decay=0.01)

    # optimizer_2 = optim.AdamW([
    #     {'params': [param for name, param in model.named_parameters() if name.startswith('swinViT.layers2') or name.startswith('classifier2')]},
    # ], lr=0.0001, weight_decay=0.01)

    # optimizer_3 = optim.AdamW([
    #     {'params': [param for name, param in model.named_parameters() if name.startswith('swinViT.layers3') or name.startswith('classifier3')]},
    # ], lr=0.0001, weight_decay=0.01)

    # optimizer_4 = optim.AdamW([
    #     {'params': [param for name, param in model.named_parameters() if name.startswith('swinViT.layers4') or name.startswith('classifier4')]},
    # ], lr=0.0001, weight_decay=0.01)


    # optimizer_1 = optim.AdamW([
    #     {'params': [param for name, param in model.named_parameters() if name.startswith('swinViT.patch_embed') or name.startswith('swinViT.layers1') or name.startswith('classifier1')]},
    # ], lr=0.0001, weight_decay=0.01)

    # optimizer_2 = optim.AdamW([
    #     {'params': [param for name, param in model.named_parameters() if name.startswith('swinViT.patch_embed') or name.startswith('swinViT.layers1') or name.startswith('classifier1') or 
    #                 name.startswith('swinViT.layers2') or name.startswith('classifier2')]},
    # ], lr=0.0001, weight_decay=0.01)

    # optimizer_3 = optim.AdamW([
    #     {'params': [param for name, param in model.named_parameters() if name.startswith('swinViT.patch_embed') or name.startswith('swinViT.layers1') or name.startswith('classifier1') or 
    #                 name.startswith('swinViT.layers2') or name.startswith('classifier2') or name.startswith('swinViT.layers3') or name.startswith('classifier3')]},
    # ], lr=0.0001, weight_decay=0.01)

    # optimizer_4 = optim.AdamW([
    #     {'params': model.parameters()},
    # ], lr=0.0001, weight_decay=0.01)

    warmup_steps = (len(train_loader)//100)*100

    scheduler_1 = WarmupCosineSchedule(optimizer_1, warmup_steps=warmup_steps, t_total=num_epochs*len(train_loader))
    scheduler_2 = WarmupCosineSchedule(optimizer_2, warmup_steps=warmup_steps, t_total=num_epochs*len(train_loader))
    scheduler_3 = WarmupCosineSchedule(optimizer_3, warmup_steps=warmup_steps, t_total=num_epochs*len(train_loader))
    scheduler_4 = WarmupCosineSchedule(optimizer_4, warmup_steps=warmup_steps, t_total=num_epochs*len(train_loader))

    for epoch in range(start_epoch, start_epoch+num_epochs):
        start=time.time()
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f'Epoch {epoch + 1}/{start_epoch+num_epochs}')

        for i, data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()
            optimizer_4.zero_grad()

            # outputs = model(inputs)
            out1, out2, out3, out4 = model(inputs)
            out1 = torch.squeeze(out1)
            out2 = torch.squeeze(out2)
            out3 = torch.squeeze(out3)
            out4 = torch.squeeze(out4)
            # print(f'model output shape {outputs.shape}, label shape {labels.shape}')
            # print(f'model output {outputs}, label {labels}')
            loss1 = criterion(out1, labels)
            loss1.backward(retain_graph=True)
            loss2 = criterion(out2, labels)
            loss2.backward(retain_graph=True)
            loss3 = criterion(out3, labels)
            loss3.backward(retain_graph=True)
            loss4 = criterion(out4, labels)
            loss4.backward()

            optimizer_1.step()
            optimizer_2.step()
            optimizer_3.step()
            optimizer_4.step()

            scheduler_1.step()
            scheduler_2.step()
            scheduler_3.step()
            scheduler_4.step()

            running_loss += loss4.item()

            progress_bar.set_postfix(loss=running_loss / (i + 1))

        avg_loss_train = running_loss / len(train_loader)

        avg_loss_test = test_model(test_loader, model, criterion)

        avg_loss = [avg_loss_train, avg_loss_test]

        with open(loss_csv_path, 'a', newline='') as f:
            w = csv.writer(f)

            w.writerow(avg_loss)

        if (epoch + 1) % save_interval == 0 or epoch == start_epoch+num_epochs - 1:

            torch.save(model.state_dict(), os.path.join(weight_dir, f'model_epoch_{epoch+1}.pth'))
            print(f'model_epoch_{epoch+1}.pth saved')

        print(f'train loss: {avg_loss_train},     test loss: {avg_loss_test}\n')
        print("Time elapsed: ", time.time()-start)

    print(f'Finished Training. Total time: {time.time() - start_time:.2f} seconds')


def test_model(test_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='Testing')
        for i, data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)
            out1, out2, out3, out4 = model(inputs)
            outputs = torch.squeeze(out4)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            progress_bar.set_postfix(test_loss=total_loss / (i + 1))

    avg_loss = total_loss / len(test_loader)
    print(f'Test MAE: {avg_loss:.4f}')

    return avg_loss

class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, t_total: int, cycles: float = 0.5, last_epoch: int = -1
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            warmup_steps: number of warmup iterations.
            t_total: total number of training iterations.
            cycles: cosine cycles parameter.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # torch.manual_seed(42)
    # np.random.seed(42)
    seed_everything(42)

    is_loaded = False
    start_epoch = 0
    num_epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")

    data_name = 'data_all_CNN_BP_free_dim_256_depth_1111'
    print(f'data_name {data_name}')

    weight_dir = f'./model weight/{data_name}'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    loss_csv_dir = f'./csv'
    if not os.path.exists(loss_csv_dir):
        os.makedirs(loss_csv_dir)

    loss_csv_path = os.path.join(loss_csv_dir, f'loss_MAE_{data_name}.csv')

    if not is_loaded:
        with open(loss_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['Train Loss (MAE)', 'Test Loss (MAE)'])

    # x1 = np.load(f'./data/VolsUKHCCAmonthNativeSpace.npy')
    # y1 = np.load(f'./data/CAUKHCCAmonthNativeSpace.npy')
    #
    # x2 = np.load(f'./VolsNACCHCCAmonthNativeSpace.npy')
    # y2 = np.load(f'./CANACCHCCAmonthNativeSpace.npy')
    #
    # x3 = np.load(f'./VolsADNIHCCAmonthNativeSpace.npy')
    # y3 = np.load(f'./CAADNIHCCAmonthNativeSpace.npy')
    #
    # x = np.concatenate((x1, x2, x3), axis=0)
    # y = np.concatenate((y1, y2, y3), axis=0)
    # x = np.load(f'../USC_BA_estimator/VolsNACCHCCAmonthNativeSpace.npy')
    # y = np.load(f'../USC_BA_estimator/CANACCHCCAmonthNativeSpace.npy')

    x = np.load(f'./data/VolsNACCHCCAmonthNativeSpace.npy')
    y = np.load(f'./data/CANACCHCCAmonthNativeSpace.npy')

    print(x.shape)
    print(y.shape)


    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(device)

    # model = SwinT_4out(spatial_dims=3, patch=4, window=8, in_channels=1, mlp_ratio=4.0,
    #             feature_size=16, dropout_path_rate=0.0, use_checkpoint=True)
    model = model_pytorch.CNN3D()

    criterion = nn.L1Loss()

    weight_name = f'model_epoch_200.pth'
    weight_path = os.path.join(weight_dir, weight_name)

    train_model(train_loader=train_loader, test_loader=test_loader, model=model, weight_dir=weight_dir, is_loaded=is_loaded, weight_path=weight_path,
                start_epoch=start_epoch, criterion=criterion, loss_csv_path=loss_csv_path, num_epochs=num_epochs)



