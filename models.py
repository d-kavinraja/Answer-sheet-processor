# models.py

import torch.nn as nn

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN).
    This architecture must exactly match the one used to train the .pth weight files.
    """
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            # Standard CNN layers without the extra BatchNorm/Dropout that caused the mismatch
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=(2, 1)), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.cnn(x)
        
        # Prepare for RNN
        x = x.squeeze(2)  # Remove height dimension
        x = x.permute(2, 0, 1)  # Change to [width, batch, channels] for RNN

        # Recurrent layers
        x, _ = self.rnn(x)
        
        # Fully connected layer
        x = self.fc(x)
        return x
