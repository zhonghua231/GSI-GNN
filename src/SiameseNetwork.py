import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def load_embeddings(embeddings_file):
    df_embeddings = pd.read_csv(embeddings_file)
    embeddings = {
        row['node_name']: torch.tensor(
            [float(x) for x in row['node_emb'].strip('[]').split(',')], dtype=torch.float
        )
        for _, row in df_embeddings.iterrows()
    }

    return embeddings

#embeddings = load_embeddings(r'C:\Users\Windows 10\Desktop\my_model\test_model\SFR-Embedding-Mistral\src\output.csv')
#embeddings = load_embeddings('../Dataset/final_emb_with_attention.csv')
embeddings = load_embeddings('../Dataset/University Course/final_node_embeddings.csv')
#embeddings = load_embeddings('../Dataset/(id+name+lm+node).csv')

class PrerequisiteDataset(Dataset):
    def __init__(self, csv_file, embeddings):
        self.dataframe = pd.read_csv(csv_file)
        self.embeddings = embeddings

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        concept_a_emb = self.embeddings[row['A']]
        concept_b_emb = self.embeddings[row['B']]
        label = row['result']
        return concept_a_emb, concept_b_emb, torch.tensor(label, dtype=torch.float)

#batch_size超参设置
def create_dataloader(csv_file, embeddings, batch_size=32):
    dataset = PrerequisiteDataset(csv_file, embeddings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)

train_loader = create_dataloader('../Dataset/University Course/train_data.csv', embeddings)
val_loader = create_dataloader('../Dataset/University Course/val_data.csv', embeddings)
test_loader = create_dataloader('../Dataset/University Course/test_data.csv', embeddings)

class SiameseNetwork(nn.Module):
    def __init__(self, representation_size):
        super(SiameseNetwork, self).__init__()
        self.ffn = nn.Linear(representation_size, representation_size)
        self.classifier = nn.Linear(representation_size * 4, 1)

    def forward(self, e_i, e_j):
        e_i = F.relu(self.ffn(e_i))
        e_j = F.relu(self.ffn(e_j))
        combined = torch.cat([e_i, e_j, e_i - e_j, e_i * e_j], dim=1)
        p = torch.sigmoid(self.classifier(combined))

        return p

def evaluate(model, data_loader):
    model.eval()
    predictions, actuals = [], []
    scores = []
    with torch.no_grad():
        for concept_a_emb, concept_b_emb, label in data_loader:
            outputs = model(concept_a_emb, concept_b_emb).squeeze()
      #      print(f"outputs: {outputs}")  # 输出每次的预测结果
            predicted = (outputs > 0.5).float()
            predictions.append(predicted)
            actuals.append(label)
            scores.append(outputs)

        # 确保 predictions、actuals 和 scores 都有数据
    if not predictions:
        print("Error: No predictions generated.")
        return None, None, None, None, None
    predictions = torch.cat(predictions).cpu().numpy()
    actuals = torch.cat(actuals).cpu().numpy()
    scores = torch.cat(scores).cpu().numpy()

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='binary')
    recall = recall_score(actuals, predictions, average='binary')
    f1 = f1_score(actuals, predictions, average='binary')
    auc = roc_auc_score(actuals, scores)

    return accuracy, precision, recall, f1, auc

representation_size = next(iter(embeddings.values())).size(0)
model = SiameseNetwork(representation_size)
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)#learning rate 超参设置

best_val_F1 = 0
best_model_state = None


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    losses = []
    for concept_a_emb, concept_b_emb, label in train_loader:
        optimizer.zero_grad()
        outputs = model(concept_a_emb, concept_b_emb)
        # 确保 outputs 和 label 的尺寸一致
        if outputs.numel() == 1:  # 如果 outputs 是一个标量
            outputs = outputs.unsqueeze(0)  # 转换为形状为 [1] 的张量
        if label.numel() == 1:  # 如果 label 是一个标量
            label = label.unsqueeze(0)  # 转换为形状为 [1] 的张量
        loss = criterion(outputs.squeeze(), label)
   #     print(f"loss: {loss.item()}")  # 打印每个批次的损失值
        if losses:
            avg_loss = np.mean(losses)
        else:
            avg_loss = None
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_loss = np.mean(losses)

    val_metrics = evaluate(model, val_loader)
#    print(f"Validation DataLoader has {len(val_loader.dataset)} samples.")

    print(
        f'Epoch {epoch + 1}, Loss: {avg_loss:.6f}, '
        f'Validation Metrics: ACC={val_metrics[0]:.4f}, Pre={val_metrics[1]:.4f}, Recall={val_metrics[2]:.4f}, F1={val_metrics[3]:.4f}, AUC={val_metrics[4]:.4f}')

    if val_metrics[3] > best_val_F1:
        best_val_F1 = val_metrics[3]
        best_model_state = model.state_dict()
        torch.save(best_model_state, '../Datasetbest_model.pth')
        print(f'Saved new best model with F1: {best_val_F1:.4f}')

model.load_state_dict(torch.load('../Datasetbest_model.pth'))
test_metrics = evaluate(model, test_loader)
print(f'Test Metrics with Best Model: ACC={test_metrics[0]:.4f}, Pre={test_metrics[1]:.4f}, Recall={test_metrics[2]:.4f}, F1={test_metrics[3]:.4f}, AUC={test_metrics[4]:.4f}')
