import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        # input (batch_size, depth, height, width, channels)
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(6, 6, 6), stride=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(16)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(6, 6, 6), stride=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(32),
            nn.Dropout(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(6, 6, 6), stride=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(64),
            nn.Dropout(0.2)
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(6, 6, 6), stride=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(128),
            nn.Dropout(0.2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        self.classifier1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.ReLU(inplace=True)
        )
                
        self.classifier2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.ReLU(inplace=True)
        )
        
        self.classifier3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.ReLU(inplace=True)
        )
                
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        ex1 = x
        ex1 = self.global_avg_pool(ex1)
        ex1 = torch.flatten(ex1,1)
        ex1 = self.classifier1(ex1)
        
        
        x = self.block2(x)
        ex2 = x
        ex2 = self.global_avg_pool(ex2)
        ex2 = torch.flatten(ex2,1)
        ex2 = self.classifier2(ex2)
        
        x = self.block3(x)
        ex3 = x
        ex3 = self.global_avg_pool(ex3)
        ex3 = torch.flatten(ex3,1)
        ex3 = self.classifier3(ex3)
        
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  
        x = self.classifier(x)
        return ex1, ex2, ex3, x



