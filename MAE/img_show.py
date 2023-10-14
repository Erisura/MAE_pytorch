import torch
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('./models/MAE.pth')
model.to(device)

img_raw_1 = Image.open('./imgs/1.jpeg')
h, w = img_raw_1.height, img_raw_1.width
print(f'img h x w = {h} x {w}')

imgs_size, patch_size = (224,224),(16,16)
img = img_raw_1.resize(imgs_size)
rh, rw = img.height, img.width
print(f'resized img h x w = {rh} x {rw}')
img.save('./imgs/resized_img_1.jpg')

img_ts = ToTensor()(img).unsqueeze(0).to(device)
masked_img_ts, recon_img_ts = model.predict(img_ts)
masked_img_ts, recon_img_ts = masked_img_ts.squeeze(0), recon_img_ts.squeeze(0)

masked_img = ToPILImage()(masked_img_ts)
masked_img.save('./imgs/masked_img.jpg')

recon_img = ToPILImage()(recon_img_ts)
recon_img.save('./imgs/recon_img.jpg')