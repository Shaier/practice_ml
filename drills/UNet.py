def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = double_conv(3, 64)
        self.down2 = double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = double_conv(128, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(256, 128) # 256 because of skip connection
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(128, 64)
        
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u2 = self.up2(b)
        merge2 = torch.cat([u2, c2], dim=1) # Skip Connection
        c3 = self.up_conv2(merge2)
        
        u1 = self.up1(c3)
        merge1 = torch.cat([u1, c1], dim=1) # Skip Connection
        c4 = self.up_conv1(merge1)
        
        return self.out(c4)