import torch.nn as nn

INPUT_SHAPE = 3
HIDDEN_UNITS = 10
OUTPUT_SHAPE = 29

class TinyVGG(nn.Module):
    
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*8*8, out_features=output_shape)
        )
    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"Shape after conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Shape after conv_block_2: {x.shape}")
        x = self.conv_block_3(x)
        # print(f"Shape after conv_block_3: {x.shape}")
        x = self.fc_layer(x)
        # print(f"Shape after fc_layer: {x.shape}")
        return x
    def __str__(self):
        return f"TinyVGG(\n  conv_block_1={self.conv_block_1},\n  conv_block_2={self.conv_block_2},\n  conv_block_3={self.conv_block_3},\n  fc_layer={self.fc_layer}\n)"