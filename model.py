import torch.nn as nn
from torch.nn import init
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
import torch
from collections import OrderedDict 
import math


vf_w = 128
vf_h = 128
vf_d = 128

def vecAtPos3d(pos,vector_field):
    if(pos.data[0]<=0.0000001 or pos.data[0]>=vf_w-1.0000001 or pos.data[1]<=0.0000001 or pos.data[1]>=vf_h-1.0000001 or pos.data[2]<=0.0000001 or pos.data[2]>=vf_d-1.0000001):
        return False,None

    x = int(pos.data[0])
    y = int(pos.data[1])
    z = int(pos.data[2])

    vecf1 = torch.stack([vector_field[0][0][x][y][z],vector_field[0][1][x][y][z],vector_field[0][2][x][y][z]])
    vecf2 = torch.stack([vector_field[0][0][x+1][y][z],vector_field[0][1][x+1][y][z],vector_field[0][2][x+1][y][z]])
    vecf3 = torch.stack([vector_field[0][0][x][y+1][z],vector_field[0][1][x][y+1][z],vector_field[0][2][x][y+1][z]])
    vecf4 = torch.stack([vector_field[0][0][x+1][y+1][z],vector_field[0][1][x+1][y+1][z],vector_field[0][2][x+1][y+1][z]])
    vecf5 = torch.stack([vector_field[0][0][x][y][z+1],vector_field[0][1][x][y][z+1],vector_field[0][2][x][y][z+1]])
    vecf6 = torch.stack([vector_field[0][0][x+1][y][z+1],vector_field[0][1][x+1][y][z+1],vector_field[0][2][x+1][y][z+1]])
    vecf7 = torch.stack([vector_field[0][0][x][y+1][z+1],vector_field[0][1][x][y+1][z+1],vector_field[0][2][x][y+1][z+1]])
    vecf8 = torch.stack([vector_field[0][0][x+1][y+1][z+1],vector_field[0][1][x+1][y+1][z+1],vector_field[0][2][x+1][y+1][z+1]])

    facx = pos[0]-x
    facy = pos[1]-y
    facz = pos[2]-z

    ret = (1-facx)*(1-facy)*(1-facz)*vecf1+(facx)*(1-facy)*(1-facz)*vecf2+(1-facx)*(facy)*(1-facz)*vecf3+(facx)*(facy)*(1-facz)*vecf4+(1-facx)*(1-facy)*(facz)*vecf5+(facx)*(1-facy)*(facz)*vecf6+(1-facx)*(facy)*(facz)*vecf7+(facx)*(facy)*(facz)*vecf8
    return True,ret

def vec(streamline,vector_field):
    vecs = []
    #print('streamline shape', streamline.shape) # torch.Size([256, 3])
    for i in range(0,len(streamline)):
        pos = torch.stack([streamline[i][0],streamline[i][1],streamline[i][2]])
        #print(pos)
        _,vec = vecAtPos3d(pos,vector_field)
        #if vec is None:
        #print('vec', vec)
        vecs.append(vec)
    #if len(vecs)!=0:
    #print(vecs)
    v = torch.cat([vecs[j] for j in range(0,len(vecs))])
    #print('predicted velocity',v)
    #print("predicted velocity shape",v.shape) #torch.Size([sample-point * 3])
    v = v.reshape(streamline.shape[0], streamline.shape[1])
    #print("reshaped predicted v",v.shape) #torch.Size([768])
    #print("reshaped predicted v",v)
    return v

################################################### TSR InterpolationNet Model 
def BuildResidualBlock(channels,dropout,kernel,depth,bias):
  layers = []
  for i in range(int(depth)):
    layers += [nn.Conv3d(channels,channels,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias),
               nn.BatchNorm3d(channels),
               nn.ReLU(True)]
    if dropout:
      layers += [nn.Dropout(0.5)]
  layers += [nn.Conv3d(channels,channels,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias),
             nn.BatchNorm3d(channels)
           ]
  return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
  def __init__(self,channels,dropout,kernel,depth,bias):
    super(ResidualBlock,self).__init__()
    self.block = BuildResidualBlock(channels,dropout,kernel,depth,bias)

  def forward(self,x):
    out = x+self.block(x)
    return out

###################################################  Recurrent Residual Convolutional Neural Network based on U-Net
class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1



class Encoder1(nn.Module):
    def __init__(self,inc,init_channels,rb=True):
        super(Encoder1,self).__init__()
        self.conv1_0 = RRCNN_block(inc,init_channels,2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(init_channels,init_channels,4,2,1), # stride = 2, dowsscale 
            nn.BatchNorm3d(init_channels),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_0 = RRCNN_block(init_channels,2*init_channels,2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(2*init_channels,2*init_channels,4,2,1),
            nn.BatchNorm3d(2*init_channels),
            nn.ReLU(inplace=True)
            )
        self.conv3_0 = RRCNN_block(2*init_channels,4*init_channels,2)
        self.conv3 = nn.Sequential(
            nn.Conv3d(4*init_channels,4*init_channels,4,2,1),
            nn.BatchNorm3d(4*init_channels),
            nn.ReLU(inplace=True)
            )
        self.conv4_0 = RRCNN_block(4*init_channels,8*init_channels,2)
        self.conv4 = nn.Sequential(
            nn.Conv3d(8*init_channels,8*init_channels,4,2,1),
            nn.BatchNorm3d(8*init_channels),
            nn.ReLU(inplace=True)
            )
        self.rb = rb
        if self.rb:
            self.rb1 = RRCNN_block(8*init_channels,8*init_channels,2)
            self.rb2 = RRCNN_block(8*init_channels,8*init_channels,2)
            self.rb3 = RRCNN_block(8*init_channels,8*init_channels,2)
            

    def forward(self,x):
        x1_0 =self.conv1_0(x)
        x1 = self.conv1(x1_0)

        x2_0 = self.conv2_0(x1)
        x2 = self.conv2(x2_0)

        x3_0 = self.conv3_0(x2)
        x3 = self.conv3(x3_0)

        x4_0 = self.conv4_0(x3)
        x4 = self.conv4(x4_0)
        if self.rb:
            x4 = self.rb1(x4)
            x4 = self.rb2(x4)
            x4 = self.rb3(x4)
            
        return [x1,x2,x3,x4]

class Decoder1(nn.Module):
    def __init__(self,outc,init_channels,ac,num):
        super(Decoder1,self).__init__()
        self.deconv41 = nn.Sequential(
            nn.ConvTranspose3d(init_channels,init_channels//2,4,2,1), 
            nn.BatchNorm3d(init_channels//2)
            )
        self.conv_u41 = RRCNN_block(init_channels,init_channels//2,2)

        self.deconv31 = nn.Sequential(
            nn.ConvTranspose3d(init_channels//2,init_channels//4,4,2,1),
            nn.BatchNorm3d(init_channels//4) 
            )
        self.conv_u31 = RRCNN_block(init_channels//2,init_channels//4,2)

        self.deconv21 = nn.Sequential(
            nn.ConvTranspose3d(init_channels//4,init_channels//8,4,2,1),
            nn.BatchNorm3d(init_channels//8)
            )
        self.conv_u21 = RRCNN_block(init_channels//4,init_channels//8,2)

        self.deconv11 = nn.Sequential(
            nn.ConvTranspose3d(init_channels//8,init_channels//16,4,2,1),
            nn.BatchNorm3d(init_channels//16)
            )
        self.conv_u11 = nn.Sequential(
            nn.Conv3d(init_channels//16,outc,3,1,1),
            )
        self.ac = ac

    def forward(self,features):
        u11 = F.relu(self.deconv41(features[-1]))
        u11 = F.relu(self.conv_u41(torch.cat((features[-2],u11),dim=1)))
        u21 = F.relu(self.deconv31(u11))
        u21 = F.relu(self.conv_u31(torch.cat((features[-3],u21),dim=1)))
        u31 = F.relu(self.deconv21(u21)) 
        u31 = F.relu(self.conv_u21(torch.cat((features[-4],u31),dim=1)))
        u41 = F.relu(self.deconv11(u31))
        out = self.conv_u11(u41)
        return out

class R2UNet(nn.Module):
    def __init__(self,inc,outc,init_channels,num=1,ac=None,rb=True):
        super(R2UNet,self).__init__()
        self.encoder1 = Encoder1(inc,init_channels,rb=True)
        self.decoder1 = Decoder1(outc,init_channels*8,ac=None,num=1)

    def forward(self,x):
        return self.decoder1(self.encoder1(x))

class InterpolationNet_R2UNet(nn.Module): 
    def __init__(self,inc,outc,init_channels):
        super(InterpolationNet_R2UNet,self).__init__()
        self.U = R2UNet(inc,outc,init_channels,num=1,ac=None,rb=True)
        self.V = R2UNet(inc,outc,init_channels,num=1,ac=None,rb=True)
        self.W = R2UNet(inc,outc,init_channels,num=1,ac=None,rb=True)

    def forward(self,x): 
        #print('input shape', x.shape) # torch.Size([1, 9, 128, 128, 128])
        u = torch.cat((x[:,0:1,:,:,:],x[:,3:4,:,:,:],x[:,6:7,:,:,:]),1) 
        #print('u shape', u.shape) #torch.Size([1, 3, 128, 128, 128])
        v = torch.cat((x[:,1:2,:,:,:],x[:,4:5,:,:,:],x[:,7:8,:,:,:]),1)
        w = torch.cat((x[:,2:3,:,:,:],x[:,5:6,:,:,:],x[:,8:9,:,:,:]),1)
        U = self.U(u)
        V = self.V(v)
        W = self.W(w)
        #print('W shape', W.shape) # torch.Size([1, 1, 128, 128, 128])
        return U,V,W

class TSR_R2UNet(nn.Module): 
    def __init__(self,inc,outc,init_channels):
        super(TSR_R2UNet,self).__init__()
        self.Interpolation_r2unet = InterpolationNet_R2UNet(inc,outc,init_channels)
        
    def forward(self,x,seeds):
        if seeds is not None:
            #print('x shape', x.shape) #torch.Size([1, 9, 128, 128, 128])
            I_u,I_v,I_w = self.Interpolation_r2unet(x)
            # input the final vector field and the positions of the points in streamlines
            # apply vec interpolation function to compute the predicted vec in streamlines
            final = torch.cat((I_u,I_v,I_w),1)
            #print('final shape', final.shape) # torch.Size([1, 3, 128, 128, 128])
            vecs = vec(seeds,final)
            return final, vecs
        else:
            I_u,I_v,I_w = self.Interpolation_r2unet(x)
            # input the final vector field and the positions of the points in streamlines
            # apply vec interpolation function to compute the predicted vec in streamlines
            final = torch.cat((I_u,I_v,I_w),1)
            return final
###################################################


