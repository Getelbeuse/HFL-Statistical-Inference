import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
# from sklearn.preprocessing import MinMaxScaler
import os
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.io
# import pandas as pd

from utils import (
    # smooth_output_gaussian,
    # minmax_scale,
    # inverse_minmax_scale,
    # wavelength_to_rgb,
    # calculate_metrics,
    # save_to_csv,
    # inverse_transform_by_row,
    plot_results,
    plot_line_comparison_all
)


# ------------------ Conditional VAE ------------------ #
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, output_dim):
        super().__init__()
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.fc_mu = nn.Linear(1024, 256)
        self.fc_logvar = nn.Linear(1024, 256)

        self.decoder = nn.Sequential(
            nn.Linear(256 + cond_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
        )

    def encode(self, x, c):
        xc = torch.cat([x, c], dim=1)
        h = self.encoder_layer(xc)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * 0.01
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)
        return x_recon, mu, log_var


# ------------------ Loss Function ------------------ #
def loss_function(recon_x, x, mu, log_var, beta=1.0):
    L1 = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return L1 + beta * KLD


# ------------------ Training ------------------ #
def train(model, train_dl,trainsize, epochs=2000):
    model.train()
    min_total_loss = float('inf')
    best_model_total_path = 'E:\\CVAE\\Ziqi\\Model\\cvae_model_best_' + str(trainsize) + '.pth'
    final_model_total_path = 'E:\\CVAE\\Ziqi\\Model\\cvae_model_' + str(trainsize) + '.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y, c in train_dl:
            x, y, c = x.to(device), y.to(device), c.to(device)
            y_pred, mu, log_var = model(x, c)
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

    torch.save(model.state_dict(), final_model_total_path)


# ------------------ Evaluation ------------------ #
def evaluate(model, test_dl):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y, c in test_dl:
            x, y, c = x.to(device), y.to(device), c.to(device)
            y_pred, _, _ = model(x, c)
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    return np.vstack(predictions), np.vstack(actuals)


def evaluate_from_condition(model, test_dl, latent_dim=256):

    model.eval()
    generated_preds, actuals = [], []

    with torch.no_grad():
        for _, y, c in test_dl:  
            y, c = y.to(device), c.to(device)
            num_samples = c.shape[0]

            z = torch.randn(num_samples, latent_dim).to(device)

            y_pred = model.decode(z, c)

            generated_preds.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    return np.vstack(generated_preds), np.vstack(actuals)



# ------------------ Data Loading ------------------ #
# def load_real_data_with_condition(
#     data_file_path, condition_file_path,
#     seed=42, test_indices_path="test_indices_2000.npy"
# ):
#     torch.manual_seed(seed)
#     np.random.seed(seed)

#     data = np.loadtxt(data_file_path, delimiter=',', dtype=float)
#     cond = np.loadtxt(condition_file_path, delimiter=',', dtype=float)

#     assert len(data) == len(data) == len(cond), "Mismatch in sample counts"

#     input_dim = data.shape[1]
#     cond_dim = cond.shape[1]
#     output_dim = data.shape[1]

#     # scalers = [MinMaxScaler(feature_range=(1, 10)) for _ in range(data.shape[0])]
#     # data_normalized = np.array([
#     #     scalers[i].fit_transform(row.reshape(-1, 1)).flatten()
#     #     for i, row in enumerate(data)
#     # ])

#     x_tensor = torch.tensor(data, dtype=torch.float32)
#     y_tensor = torch.tensor(data, dtype=torch.float32)
#     c_tensor = torch.tensor(cond, dtype=torch.float32)

#     dataset = TensorDataset(x_tensor, y_tensor, c_tensor)
#     dataset_size = len(dataset)
#     train_size = int(0.9 * dataset_size)

#     if os.path.exists(test_indices_path):
#         test_indices = np.load(test_indices_path)
#     else:
#         indices = np.arange(dataset_size)
#         np.random.shuffle(indices)
#         test_indices = indices[train_size:]
#         np.save(test_indices_path, test_indices)

#     train_indices = np.setdiff1d(np.arange(dataset_size), test_indices)

#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)

#     train_dl = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
#     test_dl = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

#     return train_dl, test_dl, input_dim, cond_dim, output_dim

def load_real_data_with_condition(data_file_path, condition_file_path,shuffle = True, seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.loadtxt(data_file_path, delimiter=',', dtype=float)
    cond = np.loadtxt(condition_file_path, delimiter=',', dtype=float)

    assert len(data) == len(data) == len(cond), "Mismatch in sample counts"

    input_dim = data.shape[1]
    cond_dim = cond.shape[1]
    output_dim = data.shape[1]

    x_tensor = torch.tensor(data, dtype=torch.float32)
    y_tensor = torch.tensor(data, dtype=torch.float32)
    c_tensor = torch.tensor(cond, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor, c_tensor)

    dl = DataLoader(dataset, batch_size=1024, shuffle=shuffle, num_workers=0)

    return dl, input_dim, cond_dim, output_dim

# ------------------ Main ------------------ #
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainsize = 500

    data_file_path = 'E:\\CVAE\\Ziqi\\Data\\' + str(trainsize) + 'x.csv'
    condition_file_path = 'E:\\CVAE\\Ziqi\\Data\\' + str(trainsize) + 'c.csv'
    test_data_file_path = 'E:\\CVAE\\Ziqi\\Data\\tx.csv'
    test_condition_file_path = 'E:\\CVAE\\Ziqi\\Data\\tc.csv'
    best_model_total_path = 'E:\\CVAE\\Ziqi\\Model\\cvae_model_best_' + str(trainsize) + '.pth'
    save_plot_dir_recon = 'E:\\CVAE\\Ziqi\\plots\\CVAE_ziqi_recon_' + str(trainsize)
    save_plot_dir_gen = 'E:\\CVAE\\Ziqi\\plots\\CVAE_ziqi_gen_' + str(trainsize)
    save_pred_dir_recon = 'E:\\CVAE\\Ziqi\\Pred\\recon_pred_' + str(trainsize) + '.csv'
    save_pred_dir_gen = 'E:\\CVAE\\Ziqi\\Pred\\gene_pred_' + str(trainsize) + '.csv'

    train_dl, input_dim, cond_dim, output_dim = load_real_data_with_condition(data_file_path,condition_file_path,shuffle = True)
    test_dl, _, _, _ = load_real_data_with_condition(test_data_file_path,test_condition_file_path,shuffle = False)

    model = CVAE(input_dim, cond_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # train(model, train_dl,trainsize, epochs=2000)

    os.makedirs(save_plot_dir_recon, exist_ok=True)
    os.makedirs(save_plot_dir_gen, exist_ok=True)

    model.load_state_dict(torch.load(best_model_total_path, map_location=device))

    K = 100
    
    predictions, actuals = evaluate(model, train_dl)

    plot_line_comparison_all(predictions[:K], actuals[:K], save_plot_dir_recon, mode='test')
    # plot_results(predictions[:K], actuals[:K], save_plot_dir_recon, mode='test')

    diff = predictions - actuals
    mae  = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    np.savetxt(save_pred_dir_recon, predictions, delimiter=",", 
           header=",".join([f"pred_{i}" for i in range(predictions.shape[1])]),
           comments="")
    
    pred_cond, actual_cond = evaluate_from_condition(model, test_dl, latent_dim=256)

    plot_line_comparison_all(pred_cond[:K], actual_cond[:K], save_plot_dir_gen, mode='test')
    # plot_results(pred_cond[:K], actual_cond[:K], save_plot_dir_gen, mode='test')

    diff2 = pred_cond - actual_cond

    mae2  = np.mean(np.abs(diff2))
    rmse2 = np.sqrt(np.mean(diff2**2))

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae2:.4f}")
    print(f"RMSE: {rmse2:.4f}")

    mean_abs = np.mean(np.abs(actuals))  
    print(f"mean_abs: {mean_abs:.4f}")

    np.savetxt(save_pred_dir_gen, pred_cond, delimiter=",", 
           header=",".join([f"pred_{i}" for i in range(pred_cond.shape[1])]),
           comments="")