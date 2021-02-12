# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:24:57 2021

@author: danie
"""
import torch
import torch.nn as nn


##############################################################################
# Resblock
##############################################################################
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,
                               stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,
                               stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.convShortConnect = nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
                                 # x: batch*channels*rx*ry*rz
        x1 = self.conv1(x)    
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)

        x5s = self.convShortConnect(x)
        x6 = x5s + x5
        
        x7 = self.relu2(x6)
        
        return x7
    

##############################################################################
# FCN 3 Layer
##############################################################################
class FCNmodel_3pool(nn.Module):
    def __init__(self,n_class):
        super(FCNmodel_3pool,self).__init__()
        self.n_class3 = n_class
        
        self.conv = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.ResBlock1 = ResBlock(32,64).cuda()
        self.ResBlock2 = ResBlock(64,128).cuda()
        self.ResBlock3 = ResBlock(128,256).cuda()
        self.ResBlock4 = ResBlock(256,512).cuda()
        
        self.deconv1   = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,
                                            padding=1,dilation=1,
                                            output_padding=1)
        
        self.ResBlock_r1 = ResBlock(256,256).cuda()
        self.deconv_classifier1   = nn.ConvTranspose2d(256,n_class,
                                                       kernel_size=9,stride=8,
                                                       padding=1, dilation=1,
                                                       output_padding=1)

        self.deconv2   = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,
                                            padding=1, dilation=1,
                                            output_padding=1)
        self.ResBlock_r2 = ResBlock(128,128).cuda()
        self.deconv_classifier2   = nn.ConvTranspose2d(128,n_class,
                                                       kernel_size=5,stride=4,
                                                       padding=1, dilation=1,
                                                       output_padding=1)

        self.deconv3   = nn.ConvTranspose2d(128, 64,kernel_size=3,stride=2,
                                            padding=1, dilation=1,
                                            output_padding=1)
        
        self.ResBlock_r3 = ResBlock(64,64).cuda()
        self.deconv_classifier3 = nn.ConvTranspose2d(64,n_class,kernel_size=3,
                                                     stride=2,padding=1,
                                                     dilation=1,
                                                     output_padding=1)
        
        self.deconv4   = nn.ConvTranspose2d(64, 32,kernel_size=3,stride=2,
                                            padding=1,
                                            dilation=1,output_padding=1)
        self.ResBlock_r4 = ResBlock(32,32).cuda()
        self.deconv_classifier4 = nn.Conv2d(32,n_class,kernel_size=1)
        
        

    def forward(self,x):
                                    # x: batch*1*272*256
        # Encoder
        x1 = self.conv(x)           # X1: batch*32*272*256
        x2 = self.pool(x1)          # X2: batch*32*136*128
           
        x3 = self.ResBlock1(x2)     # X3: batch*64*136*128
        x4 = self.pool(x3)          # X4: batch*64*68*64
        
        x5 = self.ResBlock2(x4)     # X5: batch*128*68*64
        x6 = self.pool(x5)          # X6: batch*128*34*32
        
        x7 = self.ResBlock3(x6)     # x7: batch*256*34*32
        x8 = self.pool(x7)          # x8: batch*256*17*16
        
        x9 = self.ResBlock4(x8)     # x9: batch*512*17*16
        
        #deconv
        # Decoder
        y1 = self.deconv1(x9)        # y1: batch*256*56*20
        y1s = y1 + x7                # y1s: batch*256*56*20
        y2 = self.ResBlock_r1(y1s)     # y2: batch*256*56*20
            
        y3 = self.deconv2(y2)        # y3: batch*128*112*40
        y3s = y3 + x5                # y3s: batch*128*112*40
        y4 = self.ResBlock_r2(y3s)     # y4: batch*128*112*40
        
        y5 = self.deconv3(y4)        # y5: batch*64*224*80
        y5s = y5 + x3                # y5s: batch*64*224*80
        y6 = self.ResBlock_r3(y5s)     # y6: batch*64*224*80
        
        y7 = self.deconv4(y6)        # y7: batch*32*448*160
        y7s = y7 + x1                # y7s: batch*32*448*160
        y8 = self.ResBlock_r4(y7s)     # y8: batch*32*448*160
        
       
        z1 = self.deconv_classifier1(y2)
        z2 = self.deconv_classifier2(y4)
        z3 = self.deconv_classifier3(y6)    
        z4 = self.deconv_classifier4(y8)     

        z = z1 + z2 + z3 + z4
      
        z5 = torch.sigmoid(z)
        return z5


 #############################################################################
# FCN 2 Layer
##############################################################################   
class FCNmodel_2pool(nn.Module):
    def __init__(self,n_class):
        super(FCNmodel_2pool,self).__init__()
        self.n_class3 = n_class
        
        self.conv = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.ResBlock1 = ResBlock(32,64).cuda()
        self.ResBlock2 = ResBlock(64,128).cuda()

        self.deconv3   = nn.ConvTranspose2d(128, 64,kernel_size=3,stride=2,
                                            padding=1,
                                            dilation=1,output_padding=1)
        self.ResBlock_r3 = ResBlock(64,64).cuda()
        self.deconv_classifier3 = nn.ConvTranspose2d(64,n_class,kernel_size=3,
                                                     stride=2,padding=1,
                                                     dilation=1,
                                                     output_padding=1)
        
        self.deconv4   = nn.ConvTranspose2d(64, 32,kernel_size=3,stride=2,
                                            padding=1, dilation=1,
                                            output_padding=1)
        self.ResBlock_r4 = ResBlock(32,32).cuda()
        self.deconv_classifier4 = nn.Conv2d(32,n_class,kernel_size=1)
        
        

    def forward(self,x):
                                    # x: batch x 1 x 28 x 28
        x1 = self.conv(x)           # x1: batch x 32 x 28 x 28
        x2 = self.pool(x1)          # x2:batch x 32 x 14 x 14
           
        x3 = self.ResBlock1(x2)     # X3: batch x 64 x 14 x 14
        x4 = self.pool(x3)          # X4: batch x 64 x 7 x 7
        
        x5 = self.ResBlock2(x4)     # X5: batch x 128 x 7 x 7
        
        # The Bottleneck layer is x * batch * 128 * 7 * 7 
        
        #deconv
        y1 = self.deconv3(x5)        # y1: batch*64*14*14
        y1s = y1 + x3                # y1s: batch*64*14*14
        y2 = self.ResBlock_r3(y1s)     # y2: batch*64*14*14
            
        y3 = self.deconv4(y2)        # y3: batch*32*28*28
        y3s = y3 + x1                # y3s: batch*32*28*28
        y4 = self.ResBlock_r4(y3s)     # y4: batch*32*28*28
       
        z1 = self.deconv_classifier3(y2)
        z2 = self.deconv_classifier4(y4)    

        z = z1 + z2
      
        z5 = torch.sigmoid(z)
        return z5



 #############################################################################
# FCN 4 Layer
##############################################################################
class FCNmodel_4pool(nn.Module):
    def __init__(self,n_class):
        super(FCNmodel_4pool,self).__init__()
        self.n_class3 = n_class
        
        self.conv = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.ResBlock1 = ResBlock(32,64).cuda()
        self.ResBlock2 = ResBlock(64,128).cuda()
        self.ResBlock3 = ResBlock(128,256).cuda()
        self.ResBlock4 = ResBlock(256,512).cuda()
        self.ResBlock5 = ResBlock(512,1028).cuda()
        
        self.deconv1a =  nn.ConvTranspose2d(1028, 512, kernel_size=3, 
                                            stride=2,padding=1,
                                            dilation=1,output_padding=1)
        self.ResBlock_r1a = ResBlock(512,512).cuda()
        self.deconv_classifier1a   = nn.ConvTranspose2d(256,n_class, 
                                                       kernel_size=9,stride=8,
                                                       padding=1, dilation=1,
                                                       output_padding=1)
        
        
        
        self.deconv1   = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,
                                            padding=1, dilation=1,
                                            output_padding=1)
        
        self.ResBlock_r1 = ResBlock(256,256).cuda()
        self.deconv_classifier1   = nn.ConvTranspose2d(256,n_class, 
                                                       kernel_size=9,stride=8,
                                                       padding=1, dilation=1,
                                                       output_padding=1)

        self.deconv2   = nn.ConvTranspose2d(256,128,kernel_size=3,
                                            stride=2,padding=1,
                                            dilation=1,output_padding=1)
        
        self.ResBlock_r2 = ResBlock(128,128).cuda()
        self.deconv_classifier2   = nn.ConvTranspose2d(128,n_class,
                                                       kernel_size=5,stride=4,
                                                       padding=1, dilation=1,
                                                       output_padding=1)

        self.deconv3   = nn.ConvTranspose2d(128, 64,kernel_size=3,stride=2,
                                            padding=1, dilation=1,
                                            output_padding=1)
        
        self.ResBlock_r3 = ResBlock(64,64).cuda()
        self.deconv_classifier3 = nn.ConvTranspose2d(64,n_class,kernel_size=3,
                                                     stride=2,padding=1,
                                                     dilation=1,
                                                     output_padding=1)
        
        self.deconv4   = nn.ConvTranspose2d(64, 32,kernel_size=3,stride=2,
                                            padding=1, dilation=1, 
                                            output_padding=1)
        self.ResBlock_r4 = ResBlock(32,32).cuda()
        
        self.deconv_classifier4 = nn.Conv2d(32,n_class,kernel_size=1)
        
        

    def forward(self,x):
                                    # x: batch*1*256*256
        x1 = self.conv(x)           # X1: batch*32*256*256
        x2 = self.pool(x1)          # X2: batch*32*128*128
           
        x3 = self.ResBlock1(x2)     # X3: batch*64*128*128
        x4 = self.pool(x3)          # X4: batch*64*68*64
        
        x5 = self.ResBlock2(x4)     # X5: batch*128*64*64
        x6 = self.pool(x5)          # X6: batch*128*32*32
        
        x7 = self.ResBlock3(x6)     # x7: batch*256*32*32
        x8 = self.pool(x7)          # x8: batch*256*16*16
        
        x9 = self.ResBlock4(x8)     # x9: batch*512*16*16
        x10 = self.pool(x9)          # x10: batch*1028*8*8
        
        x11 = self.ResBlock5(x10)     # x9: batch*1028*8*8
        
        #deconv
        y1a = self.deconv1a(x11)        # y1: batch*512*16*16
        y1as = y1a + x9                # y1s: batch*256*56*20
        y2a = self.ResBlock_r1a(y1as)     # y2: batch*256*56*20
        
        y1 = self.deconv1(y2a)        # y1: batch*256*56*20
        y1s = y1 + x7                # y1s: batch*256*56*20
        y2 = self.ResBlock_r1(y1s)     # y2: batch*256*56*20
            
        y3 = self.deconv2(y2)        # y3: batch*128*112*40
        y3s = y3 + x5                # y3s: batch*128*112*40
        y4 = self.ResBlock_r2(y3s)     # y4: batch*128*112*40
        
        y5 = self.deconv3(y4)        # y5: batch*64*224*80
        y5s = y5 + x3                # y5s: batch*64*224*80
        y6 = self.ResBlock_r3(y5s)     # y6: batch*64*224*80
        
        y7 = self.deconv4(y6)        # y7: batch*32*448*160
        y7s = y7 + x1                # y7s: batch*32*448*160
        y8 = self.ResBlock_r4(y7s)     # y8: batch*32*448*160
        
 
        z1a = self.deconv_classifier1(y2a)
        z1 = self.deconv_classifier1(y2)
        z2 = self.deconv_classifier2(y4)
        z3 = self.deconv_classifier3(y6)    
        z4 = self.deconv_classifier4(y8)     

        z = z1a + z1 + z2 + z3 + z4
      
        z5 = torch.sigmoid(z)
        return z5




