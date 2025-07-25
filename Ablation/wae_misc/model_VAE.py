
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

import os
import sys

from wae_misc.attention_2d import SpatialTransformer_modified_for_wae as SpatialTransformer
from wae_misc.unet2d import Unet
unk = 22
pad = 23
start = 0
eos = 21

class Maskedwords(nn.Module):
    def __init__(self, p_word_dropout):
        super(Maskedwords, self).__init__()
        self.p = p_word_dropout

    def forward(self, x):
        data = x.clone().detach()
        mask = torch.from_numpy(np.random.binomial(1, self.p, size=tuple(data.size())).astype('uint8')).to(x.device).bool()
        data[mask] = unk
        return data

class MyEncoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, z_dim, bidir, n_layers, dropout):
        super(MyEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.bidir = bidir
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, bidirectional=bidir, dropout=dropout, batch_first=True)
        
        
        if bidir:
            self.mu_layer = nn.Linear(2 * hid_dim, z_dim)
            self.logvar_layer = nn.Linear(2 * hid_dim, z_dim)
        else:
            self.mu_layer = nn.Linear(hid_dim, z_dim)
            self.logvar_layer = nn.Linear(hid_dim, z_dim)

    def forward(self, x):
        _, hidden = self.gru(x, None)
        if self.bidir:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = hidden.view(-1, hidden.size()[-1])
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

class MyDecoder(nn.Module):
    def __init__(self, embedding_layer, emb_dim, hid_dim, output_dim, bidir, n_layers, masked_p,context_dim):
        super(MyDecoder, self).__init__()
        
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.bidir = bidir
        self.n_layers = n_layers
        self.masked_p = masked_p
        model_dim = 64
        img_dim = 128
        self.num_timesteps = 1000
        self.embedding = embedding_layer
        self.word_dropout = Maskedwords(self.masked_p)


        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, bidirectional=bidir, batch_first=True)
        self.gru1 = nn.GRU(input_size=hid_dim, hidden_size=img_dim, num_layers=n_layers, bidirectional=bidir, batch_first=True)
        self.gru2 = nn.GRU(input_size=img_dim, hidden_size=hid_dim, num_layers=n_layers, bidirectional=bidir, batch_first=True)
        #self.cross_attn = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8,  batch_first=True)
        self.cross_attn = SpatialTransformer(in_channels = 1, n_heads = 8, d_head = 2,
                  context_dim=context_dim)

        self.model_unet = Unet(
        dim=model_dim,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        context_dim=256  # constraint dimension
        )

        self.fc = nn.Sequential(nn.Dropout(self.masked_p), nn.Linear(hid_dim, output_dim))

    def forward(self, x, z, c,context = None):
        device = x.device
        batch_size, seq_lens, _ = x.size()
        z_vector = torch.cat([z, c], dim=1)
        #inputs = self.word_dropout(x)
        inputs = x
        
        z_vector_expand = z_vector.unsqueeze(1).expand(-1, seq_lens, -1)
        inputs = torch.cat([inputs, z_vector_expand], dim=2)

        #context is (batch, 1, 7 = features)
        
        #context = context.expand(-1, seq_lens, -1)

        #perform permutation for cross attention
        #inputs = inputs.permute(0,2,1)
        #context = context.permute(0,2,1)

        
        #context = context.unsqueeze(1)

        #exit()
        #inputs = inputs.permute(0,2,1)

        #print('forward shapes:',inputs.shape, z_vector.unsqueeze(0).shape)
        output,  h= self.gru(inputs, z_vector.unsqueeze(0))

        unet_enter = True
        if unet_enter:
            output,_ = self.gru1(output)
            #print("output shape:",output.shape,h.shape)
            #b,c,h,w = output.shape
            #t = torch.randint(0, self.num_timesteps, (b,)).long()
            t = torch.zeros(batch_size, dtype=torch.long, device=device)
            output = output.unsqueeze(1)
            model_output = self.model_unet(output, t, None,context = context)
            output = model_output
            output = output.squeeze(1)

            output,h = self.gru2(output,h)

        else:
        #below is the cross attention portion of the decoder
            output = output.unsqueeze(1)
            output= self.cross_attn(output,context)
            output = output.squeeze(1)
                
        y = self.fc(output)
        return y

    def samples(self, start_token, z, c, h,context):
        inputs = start_token
        seq_lens = inputs.shape[1]
        batch_size, _, _ = inputs.size()
        device = start_token.device
        
        z = z.expand(-1,seq_lens, -1)
        c = c.expand(-1,seq_lens, -1)
        #print('decoder sample shape:',inputs.shape,z.shape,c.shape,h.shape,context.shape)
        inputs = torch.cat([inputs, z, c],2)
        

        
        output, h = self.gru(inputs, h)


        unet_enter = True
        if unet_enter:
            output,_ = self.gru1(output)
            #print("output shape:",output.shape,h.shape)
            #b,c,h,w = output.shape
            #t = torch.randint(0, self.num_timesteps, (b,)).long()
            t = torch.zeros(batch_size, dtype=torch.long, device=device)
            output = output.unsqueeze(1)
            model_output = self.model_unet(output, t, None,context = context)
            output = model_output
            output = output.squeeze(1)

            output,h = self.gru2(output,h)

        else:
        #below is the cross attention portion of the decoder
            output = output.unsqueeze(1)
            output= self.cross_attn(output,context)
            output = output.squeeze(1)


        logits = self.fc(output.squeeze(1))
        return logits, h



class RNN_VAE(nn.Module):
    def __init__(self, vocab_size, max_seq_len,context_dim, device, z_dim, c_dim, emb_dim, Encoder_args, Decoder_args):
        super(RNN_VAE, self).__init__()
        self.MAX_SEQ_LEN = max_seq_len
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.device = torch.device('cuda')
        self.emb_dim = emb_dim

        self.encoder = MyEncoder(emb_dim=self.emb_dim, hid_dim=Encoder_args['h_dim'], z_dim=z_dim, 
                                 bidir=Encoder_args['biGRU'], n_layers=Encoder_args['layers'], dropout=Encoder_args['p_dropout'])
        
        self.decoder = MyDecoder(embedding_layer=self.emb_dim, emb_dim=self.emb_dim + z_dim + c_dim, hid_dim=z_dim + c_dim, 
                                output_dim=self.vocab_size, bidir=False, n_layers=1, masked_p=Decoder_args['p_word_dropout'],context_dim=context_dim)

    def sample_z(self, mu, logvar):
        eps = torch.randn(mu.size(0), self.z_dim).to(self.device)
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, mbsize):
        return torch.randn(mbsize, self.z_dim).to(self.device)

    def sample_c_prior(self, mbsize):
        return torch.from_numpy(np.random.multinomial(1, [0.5, 0.5], mbsize).astype('float32')).to(self.device)

    def forward(self, sequences,context  =None):
        inputs = sequences
        mbsize = sequences.size(0)
        mu, logvar = self.encoder(inputs)
        z = self.sample_z(mu, logvar)
        c = self.sample_c_prior(mbsize)
        dec_logits = self.decoder(sequences, z, c, context = context)
        return (mu, logvar), (z, c), dec_logits

    def sample(self, mbsize, embedding=None, z=None, c=None,context = None):
        self.eval()
        if z is None:
            z = self.sample_z_prior(mbsize)
            c = self.sample_c_prior(mbsize)
        

        h = torch.cat([z, c], dim=1).unsqueeze(0)
        z = z.unsqueeze(1)
        c = c.unsqueeze(1)
        #expand embeddings to accomodate batch size
        #embedding = embedding.expand(mbsize,-1,-1)

        logits, h = self.decoder.samples(embedding, z, c, h,context)
        self.train()

        #seq = []
        #for i in range(64):
        #    logits, h = self.decoder.samples(embedding, z, c, h)
        #    seq.append(logits)

        return logits
    
    
    def sample_deprecated(self, mbsize,embedding = None, z=None, c=None):
        #mbsize is batch  
        if z is None:
            z = self.sample_z_prior(mbsize)  # Sample Z from the prior distribution      
            c = self.sample_c_prior(mbsize)  # onehots
        else:
            z = z
            c = c
        h = torch.cat([z, c], dim=1).unsqueeze(0)
        
        #print(h.shape,z.shape,c.shape)
        self.eval()

        #bypass for full sampling
        logits, h = self.decoder.samples(embedding, z, c, h)
        self.train()
        return logits


        seqs = []
        finished = torch.zeros(mbsize, dtype=torch.bool).to(self.device)
        prefix = torch.LongTensor(mbsize).to(self.device).fill_(start)
        seqs.append(prefix)



        for i in range(self.MAX_SEQ_LEN):
            logits, h = self.decoder.samples(prefix, z, c, h)
            prefix = torch.distributions.Categorical(logits=logits).sample()
            prefix.masked_fill_(finished, pad)
            finished[prefix == eos] = True  # new EOS reached, mask out in the future.
            seqs.append(prefix)
            
            if finished.sum() == mbsize:
                break
        seqs = torch.stack(seqs, dim=1)
        self.train()
        
        return seqs


