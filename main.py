import torch
import torch.nn as nn
import mae
import transformer
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

def cifar100_dataset(batch_size=64, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    cifar100_train = CIFAR100(
        root='./cifar100',
        train=True,
        transform=transform_train,
        download=True
    )
    train_loader = DataLoader(
        dataset=cifar100_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    cifar100_test = CIFAR100(
        root='./cifar100',
        train=False,
        transform=transform_train,
        download=True
    )
    test_loader = DataLoader(
        dataset=cifar100_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

def main():
    """train"""
    cifar100_train_loader, cifar100_test_loader = cifar100_dataset(batch_size=64, num_workers=4)
    encoder = transformer.VIT(
        image_size=32,
        patch_size=16,
        dropout=0.1,
        embed_dropout=0.1
    )
    model = mae.MAE(
        encoder=encoder,
        decoder_dim=768,
        decoder_depth=6,
        decoder_dropout=0.1,
        mask_ratio=0.75
    )
    
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

    training_epoch = 1000

    model.train()
    max_loss = 1e5
    for epoch in range(training_epoch):
        running_loss = 0
        for i,(inputs,_) in enumerate(cifar100_train_loader):

            inputs = inputs.to(device)

            optimizer.zero_grad()

            mask_img, out_mask_img = model(inputs)
            loss = criterion(mask_img, out_mask_img)
            loss.backward()

            optimizer.step()

            running_loss+=loss.item()

            if (i+1)%100 == 0:
                print(f'epoch: {epoch}, batch: {i+1}/{len(cifar100_train_loader)}, loss:{loss.item()}')
        
        print('###############################################')
        print(f'epoch: {epoch}, running loss: {running_loss}')
        print('###############################################')
        
        if (epoch+1)%5 == 0:
            model.eval()
            print('#################model testing##################')
            with torch.no_grad():
                loss_total = 0
                for i,(inputs,_) in enumerate(cifar100_test_loader):
                    
                    inputs = inputs.to(device)
                    
                    mask_img, out_mask_img = model(inputs)
                    loss = criterion(mask_img, out_mask_img)

                    loss_total += loss.item()
                print(f'total loss: {loss_total}')
                
                # save the best model
                if loss_total < max_loss:
                    torch.save(model,'./models/MAE.pth')
                    print(f'best model saved at epoch:{epoch}')
                    max_loss = loss_total


if __name__ == '__main__':
    main()