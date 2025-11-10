import torch
import torch.nn as nn

### CNN for classifying raw EEG samples in shape (Batch size, num_channels, seq_lenth) or (num_channels, seq_length) ###
class EEGCNNLight(nn.Module):
    def __init__(self, num_classes, in_channels, seq_len, kernel_size = None, maxpool_size=None, dropout = 0.5, out_channel_1_dim = 3, out_channel_2_dim = 3):
        if not kernel_size:
            kernel_size = round(seq_len/12)
        super(EEGCNNLight, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel_1_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel_1_dim),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel_1_dim, out_channels=out_channel_2_dim, kernel_size=(1,kernel_size), bias=False),
            nn.BatchNorm2d(out_channel_2_dim),
            nn.Tanh(),
        )
        curr_size = seq_len-kernel_size+1
        
        # By default, select valid maxpool size closest to the value that splits the input into 150 segments.
        if not maxpool_size:
            maxpool_size = round(curr_size/150)
            for i in range(1,11):
                if maxpool_size - i > 0 and curr_size % (maxpool_size - i) == 0:
                    maxpool_size = maxpool_size - i
                    break 
                if maxpool_size + i <= curr_size and curr_size % (maxpool_size + i) == 0:
                    maxpool_size = maxpool_size + i 
                    break
        self.maxpool = nn.MaxPool2d([1,maxpool_size],stride=[1,maxpool_size],padding=0)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=(round(curr_size/maxpool_size)*out_channel_2_dim), out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


