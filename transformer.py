import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_pair(t):
    return t if isinstance(t,tuple) else (t,t)

class PreNorm(nn.Module):
    def __init__(self, dim, net) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net
    
    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = num_heads*dim_per_head
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head==dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x):
        b, l, d = x.shape

        # QKV
        #(b,l,dim_all_heads*3)
        qkv = self.to_qkv(x)
        #(3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b,l,3,self.num_heads,-1).permute(2,0,3,1,4).contiguous()
        # 3*(b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        # compute attention
        #(b,num_heads,l,l)
        atten = self.attend(
            torch.matmul(q,k.transpose(-1,-2))*self.scale
        )

        # put attention on value & resize
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(atten,v)

        #(b,l,dim_all_heads)
        z = z.permute(0,2,1,3).reshape(b,l,-1)

        # project out
        #(b, l, dim)
        return self.out(z)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self,x):
        return self.net(x)
    
class Transformer(nn.Module):
    def __init__(self,
                  dim, mlp_dim, 
                  depth=6, num_heads=8, 
                  dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,SelfAttention(dim=dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim,FFN(dim=dim, hidden_dim= mlp_dim))
            ]))
    def forward(self,x):
        for atten, ffn in self.layers:
            x = x + atten(x)
            x = x + ffn(x)       
        return x
    
class VIT(nn.Module):
    def __init__(self,
                 image_size, patch_size,
                 num_classes=1000, dim=1024,
                 depth=6, num_heads=8, dim_per_head=64,
                 mlp_dim=2048, pool='cls', channels=3,dropout=0.,embed_dropout=0.):
        super().__init__()

        img_h,img_w = to_pair(image_size)
        self.patch_h, self.patch_w = to_pair(patch_size)
        assert (img_h % self.patch_h == 0 and img_w % self.patch_w == 0),\
            f'Image dimensions ({img_h},{img_w}) must be divisible by the patch dimensions ({self.patch_h, self.patch_w})).'
        num_patches = (img_h//self.patch_h) * (img_w//self.patch_w)
        
        assert pool in {'cls', 'mean'}, f'pool type must be either cls or mean, got {pool}.'

        patch_dim = self.patch_h * self.patch_w * channels
        self.patch_embed = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1,1,dim,device=device))

        # 1(cls) + num_patches
        self.pos_embed = nn.Parameter(torch.randn(1,num_patches+1, dim, device=device))
        self.dropout = nn.Dropout(p=embed_dropout)

        self.transformer = Transformer(
            dim=dim,mlp_dim=mlp_dim,
            depth=depth, num_heads=num_heads, dim_per_head=dim_per_head,
            dropout=dropout
        )
        self.pool = pool

        self.mlp_head = PreNorm(dim, nn.Linear(dim,num_classes))
    
    def forward(self, x):
        b,c,img_h,img_w = x.shape
        assert not(img_h % self.patch_h == 0 and img_w % self.patch_w == 0),\
            f'Image dimensions ({img_h},{img_w} must be divisible by the patch dimensions ({self.patch_h, self.patch_w})).'
        
        # patch partition
        num_patches = (img_h//self.patch_h) * (img_w//self.patch_w)
        patches = x.reshape(
            b,c,
            img_h//self.patch_h, self.patch_h,
            img_w//self.patch_w, self.patch_w,
        ).permute(0,2,4,3,5,1).reshape(b, num_patches, -1)

        #(b,num_patches,dim)
        tokens = self.patch_embed(patches)

        # (b, num_patches+1, dim)
        tokens = torch.cat([self.cls_token.repeat(b,1,1),tokens],dim=1)
        tokens += self.pos_embed[:,:(num_patches+1)] # auto extended
        tokens = self.dropout(tokens)

        # transformer encoding
        enc_tokens = self.transformer(tokens)

        # pooling
        pooled = enc_tokens[:,0] if self.pool=='cls' else enc_tokens.mean(dim=1)

        # Classification
        logits = self.mlp_head(pooled)

        return logits



