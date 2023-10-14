import torch
import torch.nn as nn
import transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MAE(nn.Module):
    def __init__(self,
                 encoder,
                 decoder_dim, decoder_depth, decoder_num_heads=8, decoder_dim_per_head=64, decoder_dropout=0.,
                 mask_ratio=0.75
                ):
        super().__init__()
        assert  0.< mask_ratio <1., f'mask ration must be between 0 and 1, got {mask_ratio}'

        # encoder(VIT)
        self.encoder = encoder
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w

        num_patches_plus_cls, encoder_dim = encoder.pos_embed.shape[-2:]

        # 每个patch的像素数目
        num_pixels_per_patch = encoder.patch_embed.weight.shape[-1]

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if not encoder_dim==decoder_dim else nn.Identity()

        self.mask_ratio = mask_ratio

        #mask token 按照论文来，是一个共享的可学习向量
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))

        # decoder(multi layer transformer)
        self.decoder = transformer.Transformer(
            dim=decoder_dim, mlp_dim=decoder_dim*4,
            depth=decoder_depth, num_heads=decoder_num_heads,dim_per_head= decoder_dim_per_head,dropout=decoder_dropout
        )
        # decoder position embedding
        self.decoder_pos_embed = nn.Embedding(num_patches_plus_cls-1, decoder_dim)

        # project back to patch pixels
        self.pixel_head = nn.Linear(decoder_dim, num_pixels_per_patch)

    def forward(self,x):
        b,c,h,w = x.shape
        # patch partition
        assert not (h%self.patch_h) and not (w%self.patch_w),f'image size{(h, w)} must be divisible by patch size:{(self.patch_h, self.patch_w)}'
        num_patches = (h//self.patch_h)*(w//self.patch_w)

        patches = x.reshape(
            b,c,
            h//self.patch_h, self.patch_h,
            w//self.patch_w, self.patch_w
        ).permute(0,2,4,3,5,1).reshape(b,num_patches,-1)

        num_mask = int(num_patches * self.mask_ratio)

        # shuffle
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        mask_indices, unmask_indices = shuffle_indices[:,:num_mask], shuffle_indices[:,num_mask:]

        # get mask/unmask patches
        batch_indices = torch.arange(b).unsqueeze(-1)
        # (b, num_patches, patch_size**2 * c)
        mask_patches, unmask_patches = patches[batch_indices, mask_indices], patches[batch_indices, unmask_indices]

        # encode 只encode unmask的部分,只使用其中的embedding和transformer
        unmask_tokens = self.encoder.patch_embed(unmask_patches)
        unmask_tokens += self.encoder.pos_embed.repeat(b,1,1)[batch_indices, unmask_indices + 1] # VIT的encoder pos embedding里带了cls，所以要+1
        encoded_tokens = self.encoder.transformer(unmask_tokens)

        # decode
        # (b, num_unmask, encoder_dim)->(b, num_unmask, decoder_dim)
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

        # (decoder_dim) -> (b,num_mask,decoder_dim)
        mask_tokens = self.mask_embed[None,None,:].repeat(b,num_mask,1)
        mask_tokens += self.decoder_pos_embed(mask_indices)

        # (b, num_patches, decoder_dim)
        cat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        dec_input_tokens = torch.zeros_like(cat_tokens)
        dec_input_tokens[batch_indices, shuffle_indices] = cat_tokens

        # decode
        dec_out_tokens = self.decoder(dec_input_tokens)

        # output of masked tokens
        #(b, num_patches, dec_dim) ->(b, num_mask, dec_dim)
        dec_out_masked_tokens = dec_out_tokens[batch_indices, mask_indices]

        # to patch pixels
        dec_out_masked_pixels = self.pixel_head(dec_out_masked_tokens)
    
        #(b, num_mask, pixels_per_patch=patch_size**2 * c)
        return mask_patches, dec_out_masked_pixels
    
    def predict(self, x):
        # use pre-trained model to predict
        device = x.device

        b,c,h,w = x.shape

        # patch partition
        assert (h % self.patch_h == 0) and (w % self.patch_w == 0), f'image size{(h, w)} must be divisible by patch size:{(self.patch_h, self.patch_w)}'
        num_patches = (h//self.patch_h) * (w//self.patch_w)

        # (b,num_patches, patch_size**2 *c)
        patches = x.reshape(
            b,c,
            h//self.patch_h, self.patch_h,
            w//self.patch_w, self.patch_w
        ).permute(0,2,4,3,5,1).reshape(b,num_patches,-1)

        num_mask = int(num_patches * self.mask_ratio)
        
        #shuffle
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        mask_indices, unmask_indices = shuffle_indices[:,:num_mask], shuffle_indices[:,num_mask:]

        # slice the patches
        batch_indices = torch.arange(b, device=device).unsqueeze(-1)
        mask_patches, unmask_patches = patches[batch_indices,mask_indices], patches[batch_indices, unmask_indices]

        # encode unmask patches
        unmask_tokens = self.encoder.patch_embed(unmask_patches) 
        unmask_tokens += self.encoder.pos_embed.repeat(b,1,1)[batch_indices, unmask_indices + 1]
        encoded_tokens = self.encoder.transformer(unmask_tokens)

        ''' decode all patches '''
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)
        
        # mask token
        mask_tokens = self.mask_embed[None,None,:].repeat(b, num_mask ,1)
        mask_tokens += self.decoder_pos_embed(mask_indices)

        # concat and un-shuffle
        dec_input_tokens = torch.zeros_like(patches,device=device)
        dec_input_tokens[batch_indices, mask_indices] = mask_tokens
        dec_input_tokens[batch_indices, unmask_indices] = enc_to_dec_tokens

        # mask pixel prediction
        dec_out_tokens = self.decoder(dec_input_tokens)
        dec_mask_tokens = dec_out_tokens[batch_indices, mask_indices]

        dec_out_mask_pixels = self.pixel_head(dec_mask_tokens)

        '''Reconstruction'''
        recon_patches = patches.detach()
        # un-shuffle
        recon_patches[batch_indices, mask_indices] = dec_out_mask_pixels
        
        # reshape back to image
        # (b,n_patches, patch_size**2 * c) -> (b,c,h,w)
        recon_img = recon_patches.reshape(
            b, h//self.patch_h, w//self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0,5,1,3,2,4).reshape(b,c,h,w)

        # masked image
        mask_patches = torch.randn_like(mask_patches,device=mask_patches.device)
        patches[batch_indices, mask_indices] = mask_patches
        mask_img = patches.reshape(
            b, h//self.patch_h, w//self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0,5,1,3,2,4).reshape(b,c,h,w)

        return recon_img, mask_img
        
