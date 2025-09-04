import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split  
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.io
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.optim as optim


from utils import (
    smooth_output_gaussian,
    minmax_scale,
    inverse_minmax_scale,
    wavelength_to_rgb,
    calculate_metrics,
    save_to_csv,
    inverse_transform_by_row,
    plot_results,
    plot_line_comparison_all
)


class VAE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # encoder
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        
        self.fc1 = nn.Linear(1024, 256)  
        self.fc2 = nn.Linear(1024, 256)  

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
        )

    def encode(self, x):
        x = self.encoder_layer(x)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)*0.01
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var



def loss_function(recon_x, x, mu, log_var, beta=1.0):
    L1 = F.l1_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    return L1 + beta * KLD




 

def train(model, train_dl, epochs=500):
    model.train()

    min_total_loss = float('inf')

    best_model_total_path = 'F:\\Ziqi\\Model\\vae_model_best.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            y_pred, mu, log_var = model(x)
            loss = loss_function(y_pred, y, mu, log_var) 
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_total_loss = total_loss / len(train_dl)
        print(f'epoch = {epoch + 1}, total_loss = {avg_total_loss:.4f}')

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_total_path)

    torch.save(model.state_dict(), 'F:\\Ziqi\\Model\\vae_model.pth')


def evaluate(model, test_dl):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            y_pred, _, _ = model(x)

            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    return np.vstack(predictions), np.vstack(actuals)



def load_real_data(aphy_file_path, rrs_file_path, seed=42, test_indices_path="test_indices_500.npy"):
    torch.manual_seed(seed)  # 固定 PyTorch 随机种子
    np.random.seed(seed)     # 固定 NumPy 随机种子

    # 读取数据
    array1 = np.loadtxt(aphy_file_path, delimiter=',', dtype=float)
    array2 = np.loadtxt(rrs_file_path, delimiter=',', dtype=float)

    Rrs_real = array2
    a_phy_real = array1

    input_dim = Rrs_real.shape[1]
    output_dim = a_phy_real.shape[1]


    # 归一化 Rrs_real
    scalers_Rrs_real = [MinMaxScaler(feature_range=(1, 10)) for _ in range(Rrs_real.shape[0])]
    Rrs_real_normalized = np.array([scalers_Rrs_real[i].fit_transform(row.reshape(-1, 1)).flatten() for i, row in enumerate(Rrs_real)])

    # 转换为 PyTorch 张量
    Rrs_real_tensor = torch.tensor(Rrs_real_normalized, dtype=torch.float32)
    a_phy_real_tensor = torch.tensor(a_phy_real, dtype=torch.float32)

    # 构建数据集
    dataset_real = TensorDataset(Rrs_real_tensor, a_phy_real_tensor)
    dataset_size = len(dataset_real)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size

    # **固定 test set 索引**
    if os.path.exists(test_indices_path):
        test_indices = np.load(test_indices_path)  # 直接加载已有的 test set 索引
    else:
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)  # 随机打乱索引
        test_indices = indices[train_size:]  # 取 30% 作为 test set
        np.save(test_indices_path, test_indices)  # 保存索引，确保所有代码运行时 test set 一致

    # 计算 train set 索引
    train_indices = np.setdiff1d(np.arange(dataset_size), test_indices)

    # 根据索引划分数据
    train_dataset_real = torch.utils.data.Subset(dataset_real, train_indices)
    test_dataset_real = torch.utils.data.Subset(dataset_real, test_indices)

    # 创建 DataLoader
    train_real_dl = DataLoader(train_dataset_real, batch_size=1024, shuffle=True, num_workers=0)
    test_real_dl = DataLoader(test_dataset_real, batch_size=test_size, shuffle=False, num_workers=0)

    return train_real_dl, test_real_dl, input_dim, output_dim




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_real_dl, test_real_dl, input_dim, output_dim  = load_real_data('F:\\Ziqi\\Data\\500\\500.csv','F:\\Ziqi\\Data\\500\\500.csv')

    save_dir = "F:\\Ziqi\\plots\\VAE_ziqi_500"
    os.makedirs(save_dir, exist_ok=True)

    model = VAE(input_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)


    train(model, train_real_dl, epochs=3000)

    model.load_state_dict(torch.load('F:\\Ziqi\\Model\\vae_model_best.pth', map_location=device))

    predictions, actuals = evaluate(model, test_real_dl)

    plot_line_comparison_all(predictions, actuals, save_dir, mode='test')
    plot_results(predictions, actuals, save_dir, mode='test')


