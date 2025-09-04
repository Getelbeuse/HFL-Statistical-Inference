# pip install torch torchvision pillow
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 32),              # -> 32×512×512
            ConvBlock(32, 32),
            nn.MaxPool2d(2),               # -> 32×256×256
        )
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64, dropout=0.05),
            nn.MaxPool2d(2),               # -> 64×128×128
        )
        self.stage3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128,128, dropout=0.10),
            nn.MaxPool2d(2),               # -> 128×64×64
        )
        self.stage4 = nn.Sequential(
            ConvBlock(128,256),
            ConvBlock(256,256, dropout=0.10),
            nn.MaxPool2d(2),               # -> 256×32×32
        )
        self.stage5 = nn.Sequential(
            ConvBlock(256,384),
            ConvBlock(384,384, dropout=0.15),
            nn.MaxPool2d(2),               # -> 384×16×16
        )
        self.head = nn.Sequential(
            ConvBlock(384, 512),
            nn.AdaptiveAvgPool2d(1),       # -> 512×1×1
            nn.Flatten(),                  # -> 512
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)    # -> 13
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.head(x)
        return x  # logits (B×13)

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train(model, loader, optim, criterion, device,best_model_total_path, final_model_total_path, epochs=50):
    model.train()
    min_total_loss = float('inf')
    best_model_total_path = best_model_total_path
    final_model_total_path = final_model_total_path
    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy(logits, labels) * bs
        avg_total_loss = total_loss / len(loader)
        avg_total_acc = total_acc / len(loader)
        print(f"epoch {epoch+1}: acc={avg_total_acc:.4f}, loss={avg_total_loss:.4f}")


        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_model_total_path)
            print('Saved')

    torch.save(model.state_dict(), final_model_total_path)




def main():
    dataindex = 21015
    trainsize = 10000
    
    train_dir = f"/scratch/gilbreth/dong427/dataset/ukb{dataindex}"
    best_model_total_path = '/home/dong427/Project/CVAE/Model/cnn_model_best_' + str(dataindex) + '_' + str(trainsize) + '.pth'
    final_model_total_path = '/home/dong427/Project/CVAE/Model/cnn_model_final_' + str(dataindex) + '_' + str(trainsize) + '.pth'
    output_csv_path = '/home/dong427/Project/CVAE/predict/prediction_' + str(dataindex) + '_' + str(trainsize) + '.csv'

    NUM_CLASSES = 13 
    IMG_SIZE = 512


    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


    train(model, train_loader, optimizer, criterion, device,best_model_total_path, final_model_total_path, epochs=50)
    # model.load_state_dict(torch.load(best_model_total_path, map_location=device))
    model.eval()
    all_paths, all_true, all_pred, all_prob = [], [], [], []
    softmax = nn.Softmax(dim=1)

    @torch.no_grad()
    def run_inference():
        for imgs, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs  = softmax(logits)                    # B x 13
            conf, preds = probs.max(dim=1)              # Top-1 prob & label

            for i, (path, true_label) in enumerate(train_ds.samples[len(all_paths):len(all_paths)+imgs.size(0)]):
                all_paths.append(path)
                all_true.append(int(true_label))
                all_pred.append(int(preds[i].item()))
                all_prob.append(float(conf[i].item()))

    run_inference()

    def parse_id_from_name(p):
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        return stem.split("_")[0]

    ids = [parse_id_from_name(p) for p in all_paths]

    pd.DataFrame({
        "id": ids,
        "true_label": all_true,      
        "pred_label": all_pred,      
        "pred_prob": all_prob        
    }).to_csv(output_csv_path, index=False)

    print(f"Done!")

if __name__ == "__main__":
    main()
