import torch
import torch.nn.functional as F
from torch import nn

def model_unique_name(net, separator=';'):
    
    if type(net.hidden_classes) == str:
        hidden_name = net.hidden_classes
    else:
        hidden_name= '_'.join([str(i) for i in net.hidden_classes])
    
    parts_list= [net.name, str(net.input_channels), str(net.num_classes), hidden_name, str(net.conv_channels), str(net.conditioned_one_hot_encoding), str(net.norm), str(net.activation), str(net.skipConnections), str(net.finalextraconvsblock), str(net.alpha), str(net.args.lr)]
    return (separator.join(parts_list)).replace('.','_').replace('[','').replace(']','')

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

# o parametro convs define quantas convolucoes intermediárias será incluidas no modelo
class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, norm='batch', activation='relu', convs=1, maxpool=1):

        super(EncoderBlock, self).__init__()
        
        layers = []
        if maxpool==3:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            
        if norm!='none':
            layers.append(NormType(norm,out_channels))
        layers.append(Activation(activation))
        
        for i in range(0,convs):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            if norm!='none':
                layers.append(NormType(norm,out_channels))
            layers.append(Activation(activation))

        if dropout:
            if norm!='none':
                layers.append(nn.Dropout(0.5))
                
        if maxpool==1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif maxpool==2:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)
    
# o parametro convs define quantas convolucoes intermediárias será incluidas no modelo
# o parametro convtranspose define se será incluida uma ultima camada convtranspose2d com fator 2 
class DecoderBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, norm='batch', activation='relu', convs=1, convtranspose=True):

        super(DecoderBlock, self).__init__()
        
        layers = []
        if norm!='none':
            layers.append(nn.Dropout2d(0.5))
        layers.append(nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1))
        if norm!='none':
            layers.append(NormType(norm, middle_channels))
        layers.append(Activation(activation))
        
        for i in range(0,convs):
            if i == (convs-1) and convtranspose==False:
                layers.append(nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1))
                if norm!='none':
                    layers.append(NormType(norm,out_channels))
            else:
                layers.append(nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1))
                if norm!='none':
                    layers.append(NormType(norm,middle_channels))
            layers.append(Activation(activation))
        
        if convtranspose:
            layers.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0))
        
        self.decode = nn.Sequential(*layers)

    def forward(self, x):

        return self.decode(x)

class CoReSeg(nn.Module):

    # gera uma string com um token único para o modelo, usado para gerar o nome do arquivo a ser salvo e para gerar as clunas dos arquivos de log ou resultados
    def unique_name(self, separator=';'):
        return model_unique_name(self,separator)
    
    def __init__(self, input_channels, num_classes, hidden_classes=None, conv_channels=64, conditioned_one_hot_encoding=True, norm='batch', activation='relu', skipConnections='all', finalextraconvsblock=2, alpha=0.95, args=None):
        
        super(CoReSeg, self).__init__()

        self.alpha=alpha
        self.args=args
        self.name='CoReSeg'
        self.conv_channels = conv_channels
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_classes = hidden_classes
        self.conditioned_one_hot_encoding = conditioned_one_hot_encoding
        self.norm = norm
        self.activation = activation
        self.skipConnections = skipConnections
        self.finalextraconvsblock = finalextraconvsblock
        
        self.channels_multiplier=2
        if self.skipConnections=='none':
            self.channels_multiplier=1
        elif self.skipConnections=='all':
            self.channels_multiplier=3     

        
        self.conditioned_input_layer = 1
        if self.conditioned_one_hot_encoding:
            self.conditioned_input_layer = self.num_classes - len(hidden_classes)

    def initialize(self):
        unet = UNet(self.input_channels, self.num_classes, self.hidden_classes)

        #closed_set
        self.enc1, self.enc2, self.enc3, self.enc4 = unet.enc1, unet.enc2, unet.enc3, unet.enc4
        self.center, self.dec4, self.dec3, self.dec2, self.dec1, self.final = unet.center, unet.dec4, unet.dec3, unet.dec2, unet.dec1, unet.final

        self.gamma_enc0, self.gamma_enc1,self.gamma_enc2,self.gamma_enc3,self.gamma_enc4 = self.encoder(self.conditioned_input_layer, self.conv_channels)
        self.beta_enc0, self.beta_enc1,self.beta_enc2,self.beta_enc3,self.beta_enc4 = self.encoder(self.conditioned_input_layer, self.conv_channels)
            
        decoder_input_channels = self.conv_channels*16
        if self.skipConnections=='none':
            decoder_input_channels = self.conv_channels*8

        self.os_center, self.os_dec4, self.os_dec3, self.os_dec2, self.os_dec1, self.os_final = self.decoder(decoder_input_channels, self.input_channels, self.conv_channels)

        initialize_weights(self)

    def encoder(self, input_channels, channels=4): #64
        norm       = self.norm
        activation = self.activation
        l0 = EncoderBlock(input_channels, channels, norm=norm, activation=activation, convs=3, maxpool=0)
        l1 = EncoderBlock(channels, channels, norm=norm, activation=activation)
        l2 = EncoderBlock(channels, channels*2, norm=norm, activation=activation)
        l3 = EncoderBlock(channels*2, channels*4, norm=norm, activation=activation)
        l4 = EncoderBlock(channels*4, channels*8, dropout=True, norm=norm, activation=activation)
        return l0, l1, l2, l3, l4
    
    def decoder(self, input_channels, output_channels, channels=4,): #64
        norm            = self.norm
        activation      = self.activation
        skipConnections = self.skipConnections
        
        endlayerconditioningsize = self.conditioned_input_layer
        
        ch_mult = channels*self.channels_multiplier
            
        dec1 = DecoderBlock(ch_mult+endlayerconditioningsize, channels, channels, norm=norm, activation=activation, convs = 1, convtranspose=False)
        
        final = nn.Conv2d(channels, output_channels, kernel_size=1)
        if self.finalextraconvsblock>0:
            dec0 = DecoderBlock(channels, channels, channels, norm=norm, activation=activation, convtranspose=False, convs=self.finalextraconvsblock-1)
            blocks = [dec0,final]
            final = nn.Sequential(*blocks)
            
        center = DecoderBlock(input_channels, ch_mult*8, channels*8, norm=norm, activation=activation)
        dec4 = DecoderBlock(ch_mult*8, ch_mult*4, channels*4, norm=norm, activation=activation)
        dec3 = DecoderBlock(ch_mult*4, ch_mult*2, channels*2, norm=norm, activation=activation)
        dec2 = DecoderBlock(ch_mult*2, ch_mult, channels, norm=norm, activation=activation)
        
        return center,dec4,dec3,dec2,dec1,final

    def concat_decoder_layers_reconstruction(self, dec, cs, cond, one_hot=None):
        assert self.skipConnections in ['all','none','csenc','osenc']
        
        if (one_hot is not None):
            dec = torch.cat([dec, F.upsample(one_hot, dec.size()[2:], mode='bilinear')],dim=1)  
            
        if self.skipConnections=='all':
            dec=torch.cat([dec, F.upsample(cs, dec.size()[2:], mode='bilinear'), F.upsample(cond, dec.size()[2:], mode='bilinear')],dim=1)
        
        elif self.skipConnections=='csenc':
            dec =  torch.cat([dec, F.upsample(cs, dec.size()[2:], mode='bilinear')],dim=1)
        
        elif self.skipConnections=='osenc':
            dec = torch.cat([dec, F.upsample(cond, dec.size()[2:], mode='bilinear')],dim=1)
        
        #if self.skipConnections=='none':
        return dec
    
    def forward(self, x,  labs=None, feat=False):
        # segmentation encoder
        cs_enc1 = self.enc1(x)
        cs_enc2 = self.enc2(cs_enc1)
        cs_enc3 = self.enc3(cs_enc2)
        cs_enc4 = self.enc4(cs_enc3)

        if labs is None:
            #segmentation decoder
            cs_center = self.center(cs_enc4)
            cs_dec4 = self.dec4(torch.cat([cs_center, F.upsample(cs_enc4, cs_center.size()[2:], mode='bilinear')], 1))
            cs_dec3 = self.dec3(torch.cat([cs_dec4, F.upsample(cs_enc3, cs_dec4.size()[2:], mode='bilinear')], 1))
            cs_dec2 = self.dec2(torch.cat([cs_dec3, F.upsample(cs_enc2, cs_dec3.size()[2:], mode='bilinear')], 1))
            cs_dec1 = self.dec1(torch.cat([cs_dec2, F.upsample(cs_enc1, cs_dec2.size()[2:], mode='bilinear')], 1))
            cs_final = self.final(cs_dec1)

            if feat:
                return (F.upsample(cs_final, x.size()[2:], mode='bilinear'),
                          cs_dec1,
                          F.upsample(cs_dec2, x.size()[2:], mode='bilinear'),
                          F.upsample(cs_dec3, x.size()[2:], mode='bilinear'),
                          F.upsample(cs_dec4, x.size()[2:], mode='bilinear'))
            else:
                return F.upsample(cs_final, x.size()[2:], mode='bilinear'), cs_enc4, cs_enc3, cs_enc2, cs_enc1
        
        if self.conditioned_one_hot_encoding:
            labs_m_onehot  = torch.zeros((labs[0].shape[0],self.conditioned_input_layer,labs[0].shape[1],labs[0].shape[2]))
            labs_nm_onehot = torch.zeros((labs[1].shape[0],self.conditioned_input_layer,labs[1].shape[1],labs[1].shape[2]))
            #print(self.conditioned_input_layer)
            for i in range(0,self.conditioned_input_layer):
                labs_m_onehot[:,i,:,:]=labs[0]==i            
                labs_nm_onehot[:,i,:,:]=labs[1]==i
            
            labs=[labs_m_onehot.float().cuda(),labs_nm_onehot.float().cuda()]

        labs[0] = labs[0].cuda()
        labs[1] = labs[1].cuda()
        
        m_gamma_enc0 = self.gamma_enc0(labs[0].cuda())
        m_gamma_enc1 = self.gamma_enc1(m_gamma_enc0)
        m_gamma_enc2 = self.gamma_enc2(m_gamma_enc1)
        m_gamma_enc3 = self.gamma_enc3(m_gamma_enc2)
        m_gamma_enc4 = self.gamma_enc4(m_gamma_enc3)
        
        m_beta_enc0 = self.beta_enc0(labs[0].cuda())
        m_beta_enc1 = self.beta_enc1(m_beta_enc0)
        m_beta_enc2 = self.beta_enc2(m_beta_enc1)
        m_beta_enc3 = self.beta_enc3(m_beta_enc2)
        m_beta_enc4 = self.beta_enc4(m_beta_enc3)
        
        if self.skipConnections=='none':
            m_center = self.os_center(cs_enc4*m_gamma_enc4+m_beta_enc4)
        else:
            m_center = self.os_center(torch.cat([cs_enc4, cs_enc4*m_gamma_enc4+m_beta_enc4],dim=1))

        m_dec4 = self.os_dec4(self.concat_decoder_layers_reconstruction(m_center, cs_enc4, cs_enc4*m_gamma_enc4+m_beta_enc4))
        m_dec3 = self.os_dec3(self.concat_decoder_layers_reconstruction(m_dec4, cs_enc3, cs_enc3*m_gamma_enc3+m_beta_enc3))
        m_dec2 = self.os_dec2(self.concat_decoder_layers_reconstruction(m_dec3, cs_enc2, cs_enc2*m_gamma_enc2+m_beta_enc2))
        m_dec1 = self.os_dec1(self.concat_decoder_layers_reconstruction(m_dec2, cs_enc1, cs_enc1*m_gamma_enc1+m_beta_enc1, labs[0]))
        
        m_final = self.os_final(m_dec1*m_gamma_enc0+m_beta_enc0)

        #conditioned non-match
        # segmentation encoder
        nm_gamma_enc0 = self.gamma_enc0(labs[1].cuda())
        nm_gamma_enc1 = self.gamma_enc1(nm_gamma_enc0)
        nm_gamma_enc2 = self.gamma_enc2(nm_gamma_enc1)
        nm_gamma_enc3 = self.gamma_enc3(nm_gamma_enc2)
        nm_gamma_enc4 = self.gamma_enc4(nm_gamma_enc3)
        
        nm_beta_enc0 = self.beta_enc0(labs[1].cuda())
        nm_beta_enc1 = self.beta_enc1(nm_beta_enc0)
        nm_beta_enc2 = self.beta_enc2(nm_beta_enc1)
        nm_beta_enc3 = self.beta_enc3(nm_beta_enc2)
        nm_beta_enc4 = self.beta_enc4(nm_beta_enc3)
        
        if self.skipConnections=='none':
            nm_center = self.os_center(cs_enc4*nm_gamma_enc4+nm_beta_enc4)
        else:
            nm_center = self.os_center(torch.cat([cs_enc4*nm_gamma_enc4+nm_beta_enc4, cs_enc4],dim=1))
        nm_dec4 = self.os_dec4(self.concat_decoder_layers_reconstruction(nm_center, cs_enc4, cs_enc4*nm_gamma_enc4+nm_beta_enc4))
        nm_dec3 = self.os_dec3(self.concat_decoder_layers_reconstruction(nm_dec4, cs_enc3, cs_enc3*nm_gamma_enc3+nm_beta_enc3))
        nm_dec2 = self.os_dec2(self.concat_decoder_layers_reconstruction(nm_dec3, cs_enc2, cs_enc2*nm_gamma_enc2+nm_beta_enc2))
        nm_dec1 = self.os_dec1(self.concat_decoder_layers_reconstruction(nm_dec2, cs_enc1, cs_enc1*nm_gamma_enc1+nm_beta_enc1, labs[1]))
        
        nm_final = self.os_final(nm_dec1*nm_gamma_enc0+nm_beta_enc0)

        return F.upsample(m_final, x.size()[2:], mode='bilinear'), F.upsample(nm_final, x.size()[2:], mode='bilinear')

    def segment_forward(self, x):
        # encoder
        cs_enc1 = self.enc1(x)
        cs_enc2 = self.enc2(cs_enc1)
        cs_enc3 = self.enc3(cs_enc2)
        cs_enc4 = self.enc4(cs_enc3)

        #segmentation decoder
        cs_center = self.center(cs_enc4)
        cs_dec4 = self.dec4(torch.cat([cs_center, F.upsample(cs_enc4, cs_center.size()[2:], mode='bilinear')], 1))
        cs_dec3 = self.dec3(torch.cat([cs_dec4, F.upsample(cs_enc3, cs_dec4.size()[2:], mode='bilinear')], 1))
        cs_dec2 = self.dec2(torch.cat([cs_dec3, F.upsample(cs_enc2, cs_dec3.size()[2:], mode='bilinear')], 1))
        cs_dec1 = self.dec1(torch.cat([cs_dec2, F.upsample(cs_enc1, cs_dec2.size()[2:], mode='bilinear')], 1))
        cs_final = self.final(cs_dec1)
        return F.upsample(cs_final, x.size()[2:], mode='bilinear'), cs_enc1, cs_enc2, cs_enc3, cs_enc4, cs_center, cs_dec1, cs_dec2, cs_dec3, cs_dec4
    
    def condition_forward(self, condition, enc1, enc2, enc3, enc4, center, dec1, dec2, dec3, dec4):      
        if self.conditioned_one_hot_encoding:
            #print(condition.shape, self.conditioned_input_layer)
            
            new_condition  = torch.zeros((condition.shape[0],self.conditioned_input_layer,condition.shape[2],condition.shape[3]))
            #print(new_condition.shape)
            
            for i in range(0,self.conditioned_input_layer):
                #print(condition.squeeze().shape,new_condition[:,i,:,:].shape)
                new_condition[:,i,:,:]=condition.squeeze()==i
                
            condition=new_condition.float().cuda()
        
        #conditioned encoder

        rec_gamma_enc0 = self.gamma_enc0(condition)
        rec_gamma_enc1 = self.gamma_enc1(rec_gamma_enc0)
        rec_gamma_enc2 = self.gamma_enc2(rec_gamma_enc1)
        rec_gamma_enc3 = self.gamma_enc3(rec_gamma_enc2)
        rec_gamma_enc4 = self.gamma_enc4(rec_gamma_enc3)
        
        rec_beta_enc0 = self.beta_enc0(condition)
        rec_beta_enc1 = self.beta_enc1(rec_beta_enc0)
        rec_beta_enc2 = self.beta_enc2(rec_beta_enc1)
        rec_beta_enc3 = self.beta_enc3(rec_beta_enc2)
        rec_beta_enc4 = self.beta_enc4(rec_beta_enc3)
        
        if self.skipConnections=='none':
            rec_center = self.os_center(enc4*rec_gamma_enc4+rec_beta_enc4)
        else:
            rec_center = self.os_center(torch.cat([enc4*rec_gamma_enc4+rec_beta_enc4, enc4],dim=1))

        #conditioned decoder
        rec_dec4 = self.os_dec4(self.concat_decoder_layers_reconstruction(rec_center, enc4, enc4*rec_gamma_enc4+rec_beta_enc4))
        rec_dec3 = self.os_dec3(self.concat_decoder_layers_reconstruction(rec_dec4, enc3, enc3*rec_gamma_enc3+rec_beta_enc3))
        rec_dec2 = self.os_dec2(self.concat_decoder_layers_reconstruction(rec_dec3, enc2, enc2*rec_gamma_enc2+rec_beta_enc2))
        rec_dec1 = self.os_dec1(self.concat_decoder_layers_reconstruction(rec_dec2, enc1, enc1*rec_gamma_enc1+rec_beta_enc1, condition))
        rec_final = self.os_final(rec_dec1*rec_gamma_enc0+rec_beta_enc0)
        rec_output = F.upsample(rec_final, condition.size()[2:], mode='bilinear')

        return rec_output
    
class UnetEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, norm='batch', activation='relu', convs=1, maxpool=1):

        super(UnetEncoderBlock, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if norm!='none':
            layers.append(NormType(norm,out_channels))
        layers.append(Activation(activation))
        
        for i in range(0,convs):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            if norm!='none':
                layers.append(NormType(norm,out_channels))
            layers.append(Activation(activation))

        if dropout:
            if norm!='none':
                layers.append(nn.Dropout(0.5))
                
        if maxpool==1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif maxpool==2:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2))

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class UnetDecoderBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, norm='batch', activation='relu', convs=1, convtranspose=True):

        super(UnetDecoderBlock, self).__init__()
        
        layers = []
        if norm!='none':
            layers.append(nn.Dropout2d(0.5))
        layers.append(nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1))
        if norm!='none':
            layers.append(NormType(norm, middle_channels))
        layers.append(Activation(activation))
        
        for i in range(0,convs):
            if i == (convs-1) and convtranspose==False:
                layers.append(nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1))
                if norm!='none':
                    layers.append(NormType(norm,out_channels))
            else:
                layers.append(nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1))
                if norm!='none':
                    layers.append(NormType(norm,middle_channels))
            layers.append(Activation(activation))
        
        if convtranspose:
            layers.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0))
        
        self.decode = nn.Sequential(*layers)

    def forward(self, x):

        return self.decode(x)
    
class UNet(nn.Module):

    def __init__(self, input_channels, num_classes, hidden_classes=None):

        super(UNet, self).__init__()

        self.enc1 = UnetEncoderBlock(input_channels, 64)
        self.enc2 = UnetEncoderBlock(64, 128)
        self.enc3 = UnetEncoderBlock(128, 256)
        self.enc4 = UnetEncoderBlock(256, 512, dropout=True)

        self.center = UnetDecoderBlock(512, 1024, 512)

        self.dec4 = UnetDecoderBlock(1024, 512, 256)
        self.dec3 = UnetDecoderBlock(512, 256, 128)
        self.dec2 = UnetDecoderBlock(256, 128, 64)

        self.dec1 = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        if hidden_classes is None:
            self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(64, num_classes - len(hidden_classes), kernel_size=1)

        initialize_weights(self)

    def forward(self, x, feat=False):

        enc1 = self.enc1(x)
#         print('enc1', enc1.size())
        enc2 = self.enc2(enc1)
#         print('enc2', enc2.size())
        enc3 = self.enc3(enc2)
#         print('enc3', enc3.size())
        enc4 = self.enc4(enc3)
#         print('enc4', enc4.size())

        center = self.center(enc4)
#         print('center', center.size())

        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
#         print('dec4', dec4.size())
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
#         print('dec3', dec3.size())
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
#         print('dec2', dec2.size())
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
#         print('dec1', dec1.size())

        final = self.final(dec1)
#         print('final', final.size())

        if feat:
            return (F.upsample(final, x.size()[2:], mode='bilinear'),
                    dec1,
                    F.upsample(dec2, x.size()[2:], mode='bilinear'),
                    F.upsample(dec3, x.size()[2:], mode='bilinear'),
                    F.upsample(dec4, x.size()[2:], mode='bilinear'))
        else:
            return F.upsample(final, x.size()[2:], mode='bilinear')
        
def NormType(normtype, channels):
    if normtype=='batch':
        return nn.BatchNorm2d(channels)
    return nn.InstanceNorm2d(channels)

def Activation(activation):
    if activation=='prerelu':
        return nn.PReLU()
    elif activation=='leakyrelu':
        return nn.LeakyReLU()
    return nn.ReLU(inplace=True)