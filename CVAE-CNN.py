import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image



class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, response=True, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        if annotations_file and os.path.exists(annotations_file):
            df = pd.read_csv(annotations_file)
            self.filenames = df.iloc[:, 0].tolist()
            start_col = 2 if response else 1
            self.conditions = torch.tensor(df.iloc[:, start_col:].values, dtype=torch.float32)
        else:
            self.filenames = sorted(os.listdir(img_dir))
            self.conditions = torch.zeros((len(self.filenames), 1), dtype=torch.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert('RGB')       

        if self.transform:
            img = self.transform(img)              

        x = img.clone()                             
        y = img

        c = self.conditions[idx]                
        return x, y, c

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.layers = nn.Sequential(*list(vgg[:16])).eval()
        for p in self.layers.parameters():
            p.requires_grad = False
        self.resize = resize

    def forward(self, x, y):
        if self.resize:
            x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224,224), mode='bilinear', align_corners=False)
        return F.mse_loss(self.layers(x), self.layers(y))


class CVAE(nn.Module):
    def __init__(self, in_channels, cond_dim, latent_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,  32, 4, 2, 1)  
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,           64, 4, 2, 1)  
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,          128, 4, 2, 1)  
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,         256, 4, 2, 1)  
        self.bn4   = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256,         512, 4, 2, 1)  
        self.bn5   = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512,         512, 4, 2, 1)  
        self.bn6   = nn.BatchNorm2d(512)

        flat_dim = 512 * 8 * 8

        self.fc_mu     = nn.Linear(flat_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim + cond_dim, latent_dim)


        self.fc_dec   = nn.Linear(latent_dim + cond_dim, flat_dim)

        self.deconv1 = nn.ConvTranspose2d(512, 512, 4, 2, 1) 
        self.dbn1    = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1) 
        self.dbn2    = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  
        self.dbn3    = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128,  64, 4, 2, 1) 
        self.dbn4    = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64,   32, 4, 2, 1) 
        self.dbn5    = nn.BatchNorm2d(32)
        self.deconv6 = nn.ConvTranspose2d(32, in_channels, 4, 2, 1) 

    def encode(self, x, c):
        h = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), 0.2)
        h = F.leaky_relu(self.bn5(self.conv5(h)), 0.2)
        h = F.leaky_relu(self.bn6(self.conv6(h)), 0.2)  
        h = h.view(h.size(0), -1)                       
        h = torch.cat([h, c], dim=1)                    
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        log_var = torch.clamp(log_var, -10, 10)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * 0.1
        return mu + eps * std

    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = self.fc_dec(h)
        h = h.view(-1, 512, 8, 8)                       
        h = F.leaky_relu(self.dbn1(self.deconv1(h)),0.2)
        h = F.leaky_relu(self.dbn2(self.deconv2(h)),0.2)
        h = F.leaky_relu(self.dbn3(self.deconv3(h)),0.2)
        h = F.leaky_relu(self.dbn4(self.deconv4(h)),0.2)
        h = F.leaky_relu(self.dbn5(self.deconv5(h)),0.2)
        h = self.deconv6(h)                             
        return h  

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z, c)
        return recon_logits, mu, logvar

def sobel_edges(x):
    # x: [B, C, H, W]
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    edge_x = F.conv2d(x, sobel_x.repeat(x.size(1),1,1,1), padding=1, groups=x.size(1))
    edge_y = F.conv2d(x, sobel_y.repeat(x.size(1),1,1,1), padding=1, groups=x.size(1))
    return torch.sqrt(edge_x**2 + edge_y**2)

def loss_function(recon_logits, x, mu, logvar, beta=1.0,
                       w_mse=1.0, w_ssim=0.5, w_perc=0.1, w_edge=0.1,perceptual_model=None):
    recon = torch.sigmoid(recon_logits)
    mse_loss = F.mse_loss(recon, x)
    ssim_loss = 1 - ssim(recon, x, data_range=1.0, size_average=True)
    if perceptual_model is not None:
        perc_loss = perceptual_model(recon, x)
    else:
        perc_loss = torch.zeros((), device=x.device)
    edge_loss = F.l1_loss(sobel_edges(recon), sobel_edges(x))
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = w_mse*mse_loss + w_ssim*ssim_loss + w_perc*perc_loss + w_edge*edge_loss + beta*kld
    compare_loss = w_mse*mse_loss + w_ssim*ssim_loss + w_perc*perc_loss + w_edge*edge_loss + kld/20000
    return loss,mse_loss,ssim_loss,perc_loss,edge_loss,kld,compare_loss

# def loss_function(recon_logits, x, mu, logvar,beta=1.0):
#     recon_loss = F.binary_cross_entropy_with_logits(recon_logits, x, reduction='mean')
#     kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + beta * kld,recon_loss,kld


def train(model, dataloader, optim,best_model_total_path, final_model_total_path, epochs=50, perceptual_model=None):
    
    model.train()
    min_total_loss = float('inf')
    best_model_total_path = best_model_total_path
    final_model_total_path = final_model_total_path
    for epoch in range(epochs):
        beta = min(1/20000, max(0,(epoch-epochs * 0.5) / (epochs * 0.1))/20000)
        total_loss = 0.0
        total_loss_mse = 0.0
        total_loss_ssim = 0.0
        total_loss_perc = 0.0
        total_loss_edge = 0.0
        total_loss_kld = 0.0
        total_loss_compare = 0.0
        for x, y, c in dataloader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            optim.zero_grad()
            recon_logits, mu, logvar = model(x, c)
            # loss,recon_loss,kld = loss_function(recon_logits, y, mu, logvar, beta=beta)
            loss,mse_loss,ssim_loss,perc_loss,edge_loss,kld,compare_loss = loss_function(recon_logits, y, mu, logvar, beta=beta, w_mse=1.0, w_ssim=0.3, w_perc=0.2, w_edge=0.2,perceptual_model = perceptual_model)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_loss_mse += mse_loss.item()
            total_loss_ssim += ssim_loss.item()
            total_loss_perc += perc_loss.item()
            total_loss_edge += edge_loss.item()
            total_loss_kld += kld.item()
            total_loss_compare += compare_loss.item()
        avg_total_loss = total_loss / len(dataloader)
        avg_total_loss_mse = total_loss_mse / len(dataloader)
        avg_total_loss_ssim = total_loss_ssim / len(dataloader)
        avg_total_loss_perc = total_loss_perc / len(dataloader)
        avg_total_loss_edge = total_loss_edge / len(dataloader)
        avg_total_loss_kld = total_loss_kld / len(dataloader)
        avg_total_loss_compare = total_loss_compare / len(dataloader)
        # print(f'epoch = {epoch + 1}, total_loss = {avg_total_loss:.4f}')
        print(f"epoch {epoch+1}: beta={beta:.4f}, mse={avg_total_loss_mse:.4f}, ssim={avg_total_loss_ssim:.4f}, perc={avg_total_loss_perc:.4f}, edge={avg_total_loss_edge:.4f}, kld={avg_total_loss_kld:.4f}, total={avg_total_loss:.4f}")


        if avg_total_loss_compare < min_total_loss:
            min_total_loss = avg_total_loss_compare
            torch.save(model.state_dict(), best_model_total_path)
            print('Saved')

    torch.save(model.state_dict(), final_model_total_path)
        

# def evaluate(model, dataloader):
#     model.eval()
#     recons = []
#     with torch.no_grad():
#         for x, y, c in dataloader:
#             x, y, c = x.to(device), y.to(device), c.to(device)
#             logits, _, _ = model(x, c)
#             recon = torch.sigmoid(logits)
#             recons.append(recon.cpu())
#     return torch.cat(recons, dim=0)

def evaluate(model, dataloader, dataset_filenames, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    start = 0

    with torch.no_grad():
        for x, y, c in dataloader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            b = c.shape[0]
            logits, _, _ = model(x, c)
            recons = torch.sigmoid(logits).cpu()

            batch_fnames = dataset_filenames[start:start + b]

            visualize_and_save_recons(recons, batch_fnames, output_dir,suffix= 'recon')

            start += b

# def evaluate_from_condition(model, test_dl, latent_dim=256):
#     model.eval()
#     generated_preds = []
#     # actuals = []

#     with torch.no_grad():
#         for _, _, c in test_dl:  
#             c = c.to(device)
#             num_samples = c.shape[0]

#             z = torch.randn(num_samples, latent_dim).to(device)

#             logits = model.decode(z, c)
#             recon  = torch.sigmoid(logits)

#             generated_preds.append(recon.cpu())
#             # actuals.append(y.cpu())

#     return torch.cat(generated_preds, dim=0)
#     # return torch.cat(generated_preds, dim=0), torch.cat(actuals, dim=0)
def evaluate_from_condition(model, test_dl, dataset_filenames, output_dir,latent_dim=256):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    start = 0

    with torch.no_grad():
        for _, _, c in test_dl:
            b = c.shape[0]
            c = c.to(device)

            z = torch.randn(b, latent_dim).to(device)

            logits = model.decode(z, c)
            recons = torch.sigmoid(logits).cpu()  

            batch_fnames = dataset_filenames[start:start + b]

            visualize_and_save_recons(recons, batch_fnames, output_dir,suffix= 'gen')

            start += b



def visualize_and_save_recons(recons, dataset_filenames, output_dir, suffix):
    os.makedirs(output_dir, exist_ok=True)

    for img_tensor, fname in zip(recons, dataset_filenames):
        base = os.path.splitext(fname)[0]
        save_path = os.path.join(output_dir, f"{base}_{suffix}.png")
        save_image(img_tensor, save_path)
    print(f"Saved {len(recons)} images under {output_dir}")



if __name__ == '__main__':
    # dataindex = 21015
    # trainsize = 10000
    
    # img_dir = 'D:\\datasets\\'+ str(dataindex) +'_0_0'
    # gen_dir = 'E:\\ukb_eyeimage\\gen\\'+ str(dataindex) + '_' + str(trainsize)
    # recon_dir = 'E:\\ukb_eyeimage\\recon\\'+ str(dataindex) + '_' + str(trainsize)
    
    # train_condition_dir = 'E:\\ukb_eyeimage\\condition\\train_' + str(dataindex) + '_' + str(trainsize) + '.csv'
    # test_condition_dir = 'E:\\ukb_eyeimage\\condition\\test_' + str(dataindex) + '.csv'
    # best_model_total_path = 'E:\\ukb_eyeimage\\Model\\cvae_model_best_' + str(dataindex) + '_' + str(trainsize) + '.pth'
    # final_model_total_path = 'E:\\ukb_eyeimage\\Model\\cvae_model_final_' + str(dataindex) + '_' + str(trainsize) + '.pth'

    # transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),            # -> [C,512,512], float32, [0,1]
    #     # transforms.Normalize([0.5]*3, [0.5]*3),
    # ])

    # train_dataset = CustomImageDataset(
    #     img_dir= img_dir,
    #     annotations_file=train_condition_dir,
    #     response=True, 
    #     transform=transform
    # )

    # test_dataset = CustomImageDataset(
    #     img_dir= img_dir,
    #     annotations_file=test_condition_dir,
    #     response=True,  
    #     transform=transform
    # )

    # train_dl = DataLoader(
    #     train_dataset,
    #     batch_size=64,
    #     shuffle=True,    
    #     num_workers=4
    # )

    # test_dl = DataLoader(
    #     test_dataset,
    #     batch_size=64,
    #     shuffle=False,    
    #     num_workers=4
    # )


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # in_channels = 3
    # cond_dim    = train_dataset.conditions.shape[1]


    # model = CVAE(in_channels, cond_dim).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # perceptual_model = VGGPerceptualLoss().to(device).eval()
    # for p in perceptual_model.parameters():
    #     p.requires_grad = False
    # print(1)
    # # train(model, train_dl, optimizer,best_model_total_path,final_model_total_path,
    # #   epochs=50, perceptual_model=perceptual_model)
    # train(model, train_dl, optimizer,best_model_total_path,final_model_total_path)

    # # model.load_state_dict(torch.load(best_model_total_path, map_location=device))
    # # print(1)
    # # gens = evaluate_from_condition(model, test_dl, test_dataset.filenames, gen_dir)

    # # recons = evaluate(model, train_dl, train_dataset.filenames, recon_dir)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda:", torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())
    