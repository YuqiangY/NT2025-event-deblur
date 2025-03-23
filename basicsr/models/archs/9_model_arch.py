import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange
import numbers
import torchvision
import torchvision.transforms.functional


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out=self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1,kernel_size=(3,3), padding=(1,1), bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)

        return out

class Reslayer(nn.Module):
    def __init__(self, inChannels,kernel_size=3):
        super(Reslayer,self).__init__()

        self.conv0 = nn.Conv2d(inChannels, inChannels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.act0 = nn.LeakyReLU(negative_slope=0.2)


    def forward(self,x):

        return self.act0(self.conv0(x)) + x


class CSD_Block(nn.Module):

    def __init__(self, inChannels,outchannels,n_res=2,down=True):
        super(CSD_Block,self).__init__()

        self.conv0 = nn.Conv2d(inChannels, inChannels, kernel_size=3, stride=1, padding=1)
        self.act0 = nn.LeakyReLU(negative_slope=0.2)

        self.reslayers = [Reslayer(inChannels) for _ in range(n_res)]
        self.convres = nn.Sequential(*self.reslayers)
        self.down = down
        if down:
            self.conv_down = nn.Conv2d(inChannels, outchannels, kernel_size=3, stride=2, padding=1)


    def forward(self,input):

        x = self.act0(self.conv0(input))

        x_res = self.convres(x)

        if self.down:
            x_res = self.conv_down(x_res)


        return x_res
    
class CSD_Block_Resformer(nn.Module):

    def __init__(self, inChannels,outchannels,num_heads=2,n_res=2, down=True,LayerNorm_type = 'WithBias'):
        super(CSD_Block_Resformer,self).__init__()

        self.conv0 = nn.Conv2d(inChannels, inChannels, kernel_size=3, stride=1, padding=1)
        self.act0 = nn.LeakyReLU(negative_slope=0.2)
        # self.reslayers = [Reslayer(inChannels) for _ in range(n_res)]
        self.reslayers = [TransformerBlock(dim=int(inChannels), num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=LayerNorm_type) for _ in range(n_res)]
        self.convres = nn.Sequential(*self.reslayers)
        self.down = down
        if down:
            self.conv_down = nn.Conv2d(inChannels, outchannels, kernel_size=3, stride=2, padding=1)


    def forward(self,input):

        x = self.act0(self.conv0(input))

        x_res = self.convres(x)

        if self.down:
            x_res = self.conv_down(x_res)


        return x_res
    
class UCS_Block_Resformer(nn.Module):

    def __init__(self, inChannels,outchannels,num_heads=2,n_res=2, down=True,LayerNorm_type = 'WithBias'):
        super(UCS_Block_Resformer,self).__init__()

        self.convup = nn.ConvTranspose2d(inChannels, inChannels, kernel_size=4, stride=2, padding=1)
        self.actup = nn.LeakyReLU(negative_slope=0.2)

        self.conv_mix = nn.Conv2d(inChannels+inChannels//2, inChannels, kernel_size=3, stride=1, padding=1)
        self.act_mix = nn.LeakyReLU(negative_slope=0.2)

        self.reslayers = [TransformerBlock(dim=int(inChannels), num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=LayerNorm_type) for _ in range(n_res)]
        # self.reslayers = [Reslayer(inChannels) for _ in range(n_res)]
        self.convres = nn.Sequential(*self.reslayers)

        self.conv_final = nn.Conv2d(inChannels, outchannels, kernel_size=3, stride=1, padding=1)


    def forward(self,input0,input1):

        x = self.actup(self.convup(input0))

        x_mix = self.act_mix(self.conv_mix(torch.cat((x,input1),1)))

        x_res = self.convres(x_mix)

        x_final = self.conv_final(x_res)

        return x_final
    
class UCS_Block(nn.Module):

    def __init__(self, inChannels,outchannels,n_res=2):
        super(UCS_Block,self).__init__()

        self.convup = nn.ConvTranspose2d(inChannels, inChannels, kernel_size=4, stride=2, padding=1)
        self.actup = nn.LeakyReLU(negative_slope=0.2)

        self.conv_mix = nn.Conv2d(inChannels+inChannels//2, inChannels, kernel_size=3, stride=1, padding=1)
        self.act_mix = nn.LeakyReLU(negative_slope=0.2)

        self.reslayers = [Reslayer(inChannels) for _ in range(n_res)]
        self.convres = nn.Sequential(*self.reslayers)

        self.conv_final = nn.Conv2d(inChannels, outchannels, kernel_size=3, stride=1, padding=1)


    def forward(self,input0,input1):

        x = self.actup(self.convup(input0))

        x_mix = self.act_mix(self.conv_mix(torch.cat((x,input1),1)))

        x_res = self.convres(x_mix)

        x_final = self.conv_final(x_res)

        return x_final
    



class ImageHeadBlock(nn.Module):
    def __init__(self, inChannels,outChannles,num_heads=2,n_res=2) -> None:
        super(ImageHeadBlock,self).__init__()
        self.conv0 = nn.Conv2d(inChannels, outChannles, kernel_size=3, stride=1, padding=1)
        self.act0 = nn.LeakyReLU(negative_slope=0.2)
        self.block0 = CSD_Block_Resformer(outChannles, outChannles,num_heads=num_heads,n_res=n_res, down=False)
    
    def forward(self,x):

        s0 = self.act0(self.conv0(x))
        s0 = self.block0(s0)

        return s0
        

class ImageEncoder(nn.Module):
    """Modified version of Unet from SuperSloMo.

    Difference :
    1) there is an option to skip ReLU after the last convolution.
    2) there is a size adapter module that makes sure that input of all sizes
       can be processed correctly. It is necessary because original
       UNet can process only inputs with spatial dimensions divisible by 32.
    """

    def __init__(self, inChannels):
        super(ImageEncoder, self).__init__()
        ######encoder

        self.head = ImageHeadBlock(inChannels,64,2)
        self.down1 = CSD_Block(64, 128, n_res=2)  # 128
        self.down2 = CSD_Block(128, 256, n_res=3)  # 64
        self.down3 = CSD_Block(256, 512, n_res=4)


    def forward(self, input):

        s0 = self.head(input)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = [ s1, s2, s3]
        return s0, x

class Conv3dResLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv3dResLayer, self).__init__()
        # for conv3D event
        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        
        return self.act(self.conv0(input))+input

class Conv3dBlock(nn.Module):
        
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, n_res=2):
        super(Conv3dBlock, self).__init__()
        # for conv3D event
        self.conv0 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.act0 = nn.LeakyReLU(negative_slope=0.2)

        self.res_layers = [Conv3dResLayer(in_channels,in_channels, kernel_size,stride=1) for _ in range(n_res)]

        self.convres = nn.Sequential(*self.res_layers)

        # HW down sample /2

        self.conv_downHW1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(1,2,2), padding=kernel_size//2, bias=True)
        # self.actHW1 = nn.LeakyReLU(negative_slope=0.2)

        # T down_sample 30 to t = 5  stride = 6,  3 and 2

        self.conv_downT1 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=(3,1,1), padding=5//2, bias=True)
        self.act_downT1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv_downT2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(2,1,1), padding=kernel_size//2, bias=True)
        # self.actT1 = nn.LeakyReLU(negative_slope=0.2)


    def forward(self, input):

        x = self.act0(self.conv0(input))

        x_res = self.convres(x)

        x_HW = self.conv_downHW1(x_res)

        x_H  = self.act_downT1(self.conv_downT1(x_HW))

        x_H = self.conv_downT2(x_H)

        return x_HW, x_H

class EventEncoder(nn.Module):

    def __init__(self, inChannels):
        super(EventEncoder, self).__init__()
        # for conv3D event

        # self.inplanes = 32
        ######encoder
        self.conv0 = nn.Conv3d(in_channels=inChannels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.act0 = nn.LeakyReLU(negative_slope=0.2)
        self.down1 = Conv3dBlock(16, 32,n_res=2)  # 128
        self.down2 = Conv3dBlock(32, 64,n_res=3)  # 64
        self.down3 = Conv3dBlock(64, 128,n_res=4)


    def forward(self, input):

        x = self.act0(self.conv0(input))
        d1,t1 = self.down1(x)
        d2,t2 = self.down2(d1)
        d3,t3 = self.down3(d2)

        return [d1,d2,d3],[t1,t2,t3]

class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class Parallel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Parallel_Cross_Attention, self).__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias)

        self.q2 = nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias)
        self.k2 = nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias)
        self.v2 = nn.Conv2d(dim, dim//2, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # dual

        q2 = self.q2(y) # event
        k2 = self.k2(x) # image
        v2 = self.v2(x) # image

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature2
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_cat = torch.cat((out,out2),1)

        out_cat = self.project_out(out_cat)

        return out_cat


class ParallelCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(ParallelCrossAttentionBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn = Parallel_Cross_Attention(dim, num_heads, bias)
        # mlp
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, x1, x2):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert x1.shape == x2.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = x1.shape
        fused = x1 + self.attn(self.norm1(x1), self.norm2(x2)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm3(fused))
        fused = to_4d(fused, h, w)

        return fused


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalFusionResidualLayer(nn.Module):
    def __init__(self, channel_i,channel_e,num_heads=4) -> None:
        super(TemporalFusionResidualLayer,self).__init__()

        self.conv0 = nn.Conv2d(in_channels=channel_i+channel_e, out_channels=channel_i, kernel_size=3, stride=1, padding=1, bias=True)
        self.act0 = nn.LeakyReLU(negative_slope=0.2)

        self.att = ParallelCrossAttentionBlock(channel_i, num_heads=num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')

    def forward(self,img, event):

        x = self.act0(self.conv0(torch.cat((img, event),1)))

        out = self.att(x,img)


        return out

class TemporalFusionResidualBlock(nn.Module):

    def __init__(self,channel_i,channel_e,T=5,num_heads = 4) -> None:
        super(TemporalFusionResidualBlock,self).__init__()
        self.T = T
        self.tef_layers=nn.ModuleList()
        for l_i in range(self.T):
            self.tef_layers.append(TemporalFusionResidualLayer(channel_i,channel_e,num_heads))

    def forward(self,img,event):

        # N_E,C_E,T_E,H_E,W_E = event.shape

        x = img

        for T_i in range(self.T):
            x = self.tef_layers[T_i](x, event[:,:,T_i,:,:])+x

        return x
    
class MultiScaleCrosAttentionFusionBlock(nn.Module):

    def __init__(self, channel_base,channel_ref1,channel_ref2,num_heads=4) -> None:
        super(MultiScaleCrosAttentionFusionBlock,self).__init__()

        self.conv_mix = nn.Conv2d(in_channels=channel_base+channel_ref1+channel_ref2, out_channels=channel_base, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.att = ParallelCrossAttentionBlock(dim=channel_base, num_heads=num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
    def forward(self,base,ref1,ref2):

        ref1_resized = F.interpolate(ref1,(base.shape[2],base.shape[3]),mode='bilinear')
        ref2_resized = F.interpolate(ref2,(base.shape[2],base.shape[3]),mode='bilinear')

        mix_feature = self.act(self.conv_mix((torch.cat((base,ref1_resized,ref2_resized),1))))

        att_feature = self.att(base,mix_feature)+base

        return att_feature 


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Model22(nn.Module):
    # base form model17 voxel 30,T 5


    def __init__(self, inChannels_img=3, inChannels_event=1,outChannels=3):
        super(Model22, self).__init__()
        # self._ends_with_relu = ends_with_relu
        # self.num_ff = args.future_frames
        # self.num_fb = args.past_frames
        # self.num_heads=4
        ######encoder部分
        self.encoder_img=ImageEncoder(inChannels_img)
        self.encoder_event=EventEncoder(inChannels_event)

        self.channel_i = [128,256,512]
        self.channel_e = [32,64,128]
        self.num_heads = [1,2,4]
        self.temp_res_fusion = nn.ModuleList()
        # self.deform_blocks = nn.ModuleList()
        for blk_i in range(len(self.channel_i)):
            self.temp_res_fusion.append(TemporalFusionResidualBlock(self.channel_i[blk_i],self.channel_e[blk_i],num_heads=self.num_heads[blk_i],T=5))

            # if blk_i == len(self.channel_i)-1:
            #     self.deform_blocks.append(DeformFusionBlock(self.channel_i[blk_i],have_offset=False))
            # else:
            #     self.deform_blocks.append(DeformFusionBlock(self.channel_i[blk_i],have_offset=True))
        # torch.Size([2, 64, 256, 256]) torch.Size([2, 128, 128, 128]) torch.Size([2, 256, 64, 64]) torch.Size([2, 512, 32, 32])

        self.mscaf0 = MultiScaleCrosAttentionFusionBlock(channel_base=64,channel_ref1=128,channel_ref2=256,num_heads=1)
        self.mscaf1 = MultiScaleCrosAttentionFusionBlock(channel_base=128,channel_ref1=256,channel_ref2=64,num_heads=2)
        self.mscaf2 = MultiScaleCrosAttentionFusionBlock(channel_base=256,channel_ref1=512,channel_ref2=128,num_heads=4)
        self.mscaf3 = MultiScaleCrosAttentionFusionBlock(channel_base=512,channel_ref1=256,channel_ref2=128,num_heads=4)

        

        self.decoder1 = UCS_Block(512,256)
        self.decoder2 = UCS_Block(256,128)
        self.decoder3 = UCS_Block(128,64)

        self.conv_out1 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # self.conv = nn.Conv2d(64, outChannels, 3, stride=1, padding=1)

        # # stage2
        #head
        self.stage2_head = ImageHeadBlock(inChannels=3,outChannles=64,num_heads=1,n_res=1)
        self.stage2_conv_mix1 = nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1)
        self.stage2_act_mix1 = nn.LeakyReLU(negative_slope=0.2)
        self.stage2_down1 = CSD_Block_Resformer(64, 128, num_heads=2,n_res=2)  # 128
        self.stage2_conv_mix2 = nn.Conv2d(128*3, 128, kernel_size=3, stride=1, padding=1)
        self.stage2_act_mix2 = nn.LeakyReLU(negative_slope=0.2)
        self.stage2_down2 = CSD_Block_Resformer(128, 256, num_heads=2,n_res=2)  # 64
        self.stage2_conv_mix3 = nn.Conv2d(256*3, 256, kernel_size=3, stride=1, padding=1)
        self.stage2_act_mix3 = nn.LeakyReLU(negative_slope=0.2)
        self.stage2_down3 = CSD_Block_Resformer(256, 512, num_heads=4,n_res=2)

        self.stage2_decoder1 = UCS_Block_Resformer(512,256,num_heads=4,n_res=2)
        self.stage2_decoder2 = UCS_Block_Resformer(256,128,num_heads=2,n_res=2)
        self.stage2_decoder3 = UCS_Block_Resformer(128,64,num_heads=1,n_res=1)

        self.conv_out2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # self.encoder_level2 = TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        # self.s2_head = shallow_cell(3)
        # self.s2_down1 = EN_Block(64*3, 128, n_layers=2)  # 128
        # self.s2_down2 = EN_Block(128*3, 256, n_layers=2)  # 64
        # self.s2_down3 = EN_Block(256*3, 512, n_layers=2)

        # self.s2_up1 = DE_Block(512*2, 256)
        # self.s2_up2 = DE_Block(256, 128)
        # self.s2_up3 = DE_Block(128, 64)

        # self.conv_last = nn.Conv2d(64, outChannels, 3, stride=1, padding=1)

    def forward(self, x, event, mask=None):


        input_img = x  # image t

        out_img_features0,out_img_features = self.encoder_img(input_img)

        # torch.Size([2, 128, 128, 128])
        # torch.Size([2, 256, 64, 64])
        # torch.Size([2, 512, 32, 32])


        N_E, T_E, H_E, W_E = event.shape
        event3d = torch.reshape(event,(N_E, 1, T_E, H_E, W_E))
        _,out_event1_Ts= self.encoder_event(event3d)


        # for ts in out_event0_Ts:
        #     print(ts.shape)
        #     torch.Size([2, 64, 3, 128, 128])
        #     torch.Size([2, 128, 3, 64, 64])
        #     torch.Size([2, 256, 3, 32, 32])

        # ca res fused
        fused = []

        for s_i in range(len(out_img_features)):
            fused.append(self.temp_res_fusion[s_i](out_img_features[s_i],out_event1_Ts[s_i]))

        #
        # feature size cross
        # out_img_features0,fused[0],fused[1],fused[2]
        # print(out_img_features0.shape,fused[0].shape,fused[1].shape,fused[2].shape)
        # torch.Size([2, 64, 256, 256]) torch.Size([2, 128, 128, 128]) torch.Size([2, 256, 64, 64]) torch.Size([2, 512, 32, 32])

        # dual att (base  ,ref1,ref2 ->resize dual att)
        ms_fused = []
        ms_fused.append(self.mscaf0(out_img_features0,fused[0],fused[1]))
        ms_fused.append(self.mscaf1(fused[0],fused[1],out_img_features0))
        ms_fused.append(self.mscaf2(fused[1],fused[2],fused[0]))
        ms_fused.append(self.mscaf3(fused[2],fused[1],fused[0]))


        f1 = self.decoder1(ms_fused[3],ms_fused[2])
        f2 = self.decoder2(f1,ms_fused[1])
        f3 = self.decoder3(f2,ms_fused[0])

        out1 = self.conv_out1(f3)+input_img
        # stage2 

        s2_head = self.stage2_head(input_img)
        
        s2_f1_mix = self.stage2_act_mix1(self.stage2_conv_mix1(torch.cat((ms_fused[0],s2_head,f3),1)))
        s2_f1 = self.stage2_down1(s2_f1_mix)

        s2_f2_mix = self.stage2_act_mix2(self.stage2_conv_mix2(torch.cat((ms_fused[1],s2_f1,f2),1)))
        s2_f2 = self.stage2_down2(s2_f2_mix)

        s2_f3_mix = self.stage2_act_mix3(self.stage2_conv_mix3(torch.cat((ms_fused[2],s2_f2,f1),1)))
        s2_f3 = self.stage2_down3(s2_f3_mix)

        stage2_f1 = self.stage2_decoder1(s2_f3,s2_f2)
        stage2_f2 = self.stage2_decoder2(stage2_f1,s2_f1)
        stage2_f3 = self.stage2_decoder3(stage2_f2,s2_head)

        out2 = self.conv_out2(stage2_f3)+input_img


        return [out1, out2]