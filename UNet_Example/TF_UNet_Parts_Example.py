#Original code in pytorch: https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py
#Paper with code: https://paperswithcode.com/method/u-net 

import tensorflow as tf

class DoubleConv(tf.keras.layers.Layer):
     def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
    
        self.double_conv = tf.keras.Sequential(
            tf.keras.layers.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1),
            tf.keras.layers.BatchNormalization(mid_channels),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1),
            tf.keras.layers.BatchNormalization(mid_channels),
            tf.keras.layers.ReLU()
        )

     def call(self, x):
         return self.double_conv(x)
     
class Down(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = tf.keras.Sequential(
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(tf.keras.layers.Layer):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = tf.keras.layers.Conv2DTranspose(input_shape=in_channels , filters=in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = tf.keras.utils.pad_sequences(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = tf.concat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(in_channels, out_channels, kernel_size=1),

    def forward(self, x):
        return self.conv(x)