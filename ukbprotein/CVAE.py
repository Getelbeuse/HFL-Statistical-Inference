import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os

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
    best_model_total_path = 'E:\\ukbprotein\\Model\\cvae_model_best_' + str(trainsize) + '_' + month + '.pth'
    final_model_total_path = 'E:\\ukbprotein\\Model\\cvae_model_' + str(trainsize) + '_' + month + '.pth'

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y, c in train_dl:
            x, y, c = x.to(device), y.to(device), c.to(device)
            y_pred, mu, log_var = model(x, c)
            beta = 1 * min(1.0, epoch / (epochs*0.5))
            loss = loss_function(y_pred, y, mu, log_var, beta=beta)
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
    print(device)
    trainsize = 10000
    month = 'nov'

    data_file_path = 'E:\\ukbprotein\\Data\\' + str(trainsize) + 'x_' + month + '.csv'
    condition_file_path = 'E:\\ukbprotein\\Data\\' + str(trainsize) + 'c_' + month + '.csv'
    test_data_file_path = 'E:\\ukbprotein\\Data\\tx_' + month + '.csv'
    test_condition_file_path = 'E:\\ukbprotein\\Data\\tc_' + month + '.csv'
    best_model_total_path = 'E:\\ukbprotein\\Model\\cvae_model_best_' + str(trainsize) + '_' + month + '.pth'
    save_pred_dir_recon = 'E:\\ukbprotein\\Pred\\recon_pred_' + str(trainsize) + '_' + month + '.csv'
    save_pred_dir_gen = 'E:\\ukbprotein\\Pred\\gene_pred_' + str(trainsize) + '_' + month + '.csv'

    train_dl, input_dim, cond_dim, output_dim = load_real_data_with_condition(data_file_path,condition_file_path,shuffle = True)
    test_dl, _, _, _ = load_real_data_with_condition(test_data_file_path,test_condition_file_path,shuffle = False)

    model = CVAE(input_dim, cond_dim, output_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    print(1)
    # train(model, train_dl,trainsize, epochs=2000)

    model.load_state_dict(torch.load(best_model_total_path, map_location=device))
    print(1)
    predictions, actuals = evaluate(model, train_dl)

    np.savetxt(save_pred_dir_recon, predictions, delimiter=",", 
           header=",".join([f"pred_{i}" for i in range(predictions.shape[1])]),
           comments="")
    print(1)
    pred_cond, actual_cond = evaluate_from_condition(model, test_dl, latent_dim=256)

    np.savetxt(save_pred_dir_gen, pred_cond, delimiter=",", 
           header=",".join([f"pred_{i}" for i in range(pred_cond.shape[1])]),
           comments="")
    print(1)