import os
import time
from collections import Counter
import logging

import torch
import numpy as np
import pandas as pd
from math import log
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

from mixup_module import mixup_data


class MarkerToGeneLayer(nn.Module):
    def __init__(self, snp_genes_df):
        super(MarkerToGeneLayer, self).__init__()
        if 'Gene' not in snp_genes_df.columns or 'SNP_ID' not in snp_genes_df.columns:
            raise ValueError("snp_genes_df должен содержать столбцы 'Gene' и 'SNP_ID'.")

        self.gene_to_index = {gene: idx for idx, gene in enumerate(snp_genes_df['Gene'].unique())}
        self.snp_to_gene_map = {snp: gene for snp, gene in zip(snp_genes_df['SNP_ID'], snp_genes_df['Gene'])}

    def forward(self, snps):
        gene_indices = [self.gene_to_index.get(self.snp_to_gene_map.get(snp, ""), -1) for snp in snps]

        gene_to_snp_indices = {}
        for snp_idx, gene_idx in enumerate(gene_indices):
            if gene_idx != -1:
                if gene_idx not in gene_to_snp_indices:
                    gene_to_snp_indices[gene_idx] = []
                gene_to_snp_indices[gene_idx].append(snp_idx)
        return gene_indices, gene_to_snp_indices


class GeneticEncodingBlock(nn.Module):
    def __init__(self, num_genes, embed_dim, snp_genes_df):
        super(GeneticEncodingBlock, self).__init__()
        self.marker_to_gene = MarkerToGeneLayer(snp_genes_df)
        self.gene_embedding = nn.Embedding(num_genes, embed_dim)
        self.embed_dim = embed_dim
        self.snp_weights = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.global_gene_to_index = self.marker_to_gene.gene_to_index
        self.index_to_global_gene = {idx: gene for gene, idx in self.global_gene_to_index.items()}
        self.num_genes_total = num_genes

    def forward(self, X, snp_names, gene_positions):
        batch_size, num_snps = X.shape
        device = next(self.parameters()).device

        X = torch.tensor(X.values, dtype=torch.float32, device=device) if isinstance(X, pd.DataFrame) else X.to(device)
        global_gene_indices_flat, gene_to_snp_indices = self.marker_to_gene(snp_names)

        gene_embeds_output = torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)

        if not gene_to_snp_indices:
            return gene_embeds_output

        unique_global_gene_indices_in_batch = sorted(gene_to_snp_indices.keys())

        gene_pos_values_all_genes = torch.tensor(
            [gene_positions.get(self.index_to_global_gene[global_idx], 0) for global_idx in range(self.num_genes_total)],
            dtype=torch.float32, device=device
        ).unsqueeze(-1)

        gene_avg_values_per_gene = torch.zeros(batch_size, self.num_genes_total, device=device)
        for global_gene_idx, snp_indices in gene_to_snp_indices.items():
            snp_values = X[:, snp_indices]
            avg_per_sample = snp_values.mean(dim=1)
            gene_avg_values_per_gene[:, global_gene_idx] = avg_per_sample

        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device) *
                             (-log(10000.0) / self.embed_dim))

        position_tensor_base = gene_pos_values_all_genes * div_term.unsqueeze(0)
        position_tensor = position_tensor_base.unsqueeze(0).repeat(batch_size, 1, 1)
        position_tensor = position_tensor * torch.exp(gene_avg_values_per_gene.unsqueeze(-1))

        pe = torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)
        pe[:, :, 0::2] = torch.sin(position_tensor)
        pe[:, :, 1::2] = torch.cos(position_tensor)

        snp_embeds = X.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        snp_weights_expanded = self.snp_weights.expand(batch_size, num_snps, self.embed_dim)
        weighted_snp_embeds = snp_embeds * snp_weights_expanded

        max_snp_count = max(len(indices) for indices in gene_to_snp_indices.values()) if gene_to_snp_indices else 1
        padded_snp_embeds = torch.zeros(batch_size, self.num_genes_total, max_snp_count, self.embed_dim, device=device)
        mask = torch.zeros(batch_size, self.num_genes_total, max_snp_count, device=device)

        for global_gene_idx, snp_indices in gene_to_snp_indices.items():
            snp_indices_tensor = torch.tensor(snp_indices, device=device)
            snp_offsets_tensor = torch.arange(len(snp_indices), device=device)
            padded_snp_embeds[:, global_gene_idx, snp_offsets_tensor, :] = weighted_snp_embeds[:, snp_indices_tensor, :]
            mask[:, global_gene_idx, snp_offsets_tensor] = 1

        padded_snp_embeds = padded_snp_embeds.permute(0, 3, 2, 1)

        gene_embeds_conv = F.conv1d(
            padded_snp_embeds.reshape(-1, self.embed_dim, max_snp_count),
            weight=torch.ones(self.embed_dim, 1, max_snp_count, device=device),
            groups=self.embed_dim
        ).view(batch_size, self.embed_dim, self.num_genes_total).permute(0, 2, 1)

        mask_summed = (mask.sum(dim=2) > 0).float().unsqueeze(-1)
        gene_embeds_conv *= mask_summed

        all_global_gene_indices = torch.arange(self.num_genes_total, dtype=torch.long, device=device)
        selected_gene_embeddings = self.gene_embedding(all_global_gene_indices)
        gene_embeds_output = gene_embeds_conv + selected_gene_embeddings.unsqueeze(0)
        gene_embeds_output += pe

        return gene_embeds_output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            activation='gelu', batch_first=True, dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        if x.size(0) > 0 and x.size(1) > 0:
            x = self.transformer(x)
        else:
            return torch.empty(x.size(0), 0, x.size(2), device=x.device)
        return x


class DualRegularizedPredictionBlock(nn.Module):
    """Двойной блок с разной регуляризацией"""
    def __init__(self, embed_dim, num_genes, dropout_shared=0.2, dropout_ptht=0.4):
        super(DualRegularizedPredictionBlock, self).__init__()
        self.initial_num_genes = num_genes
        
        # Общие слои
        self.shared_fc1 = nn.Linear(embed_dim * self.initial_num_genes, 256)
        self.shared_dropout = nn.Dropout(dropout_shared)
        
        # TSW голова (меньше регуляризации)
        self.fc_tsw = nn.Linear(256, 1)
        
        # Ptht голова (больше регуляризации + дополнительные слои)
        self.fc_ptht1 = nn.Linear(256, 128)
        self.ptht_dropout = nn.Dropout(dropout_ptht)
        self.fc_ptht2 = nn.Linear(128, 1)
        
    def forward(self, x):
        if x.size(1) == 0:
            return torch.zeros(x.size(0), 2, device=x.device)

        x_flat = x.view(x.size(0), -1)
        
        # Общая часть
        shared = F.relu(self.shared_fc1(x_flat))
        shared = self.shared_dropout(shared)
        
        # TSW путь (простой)
        tsw = self.fc_tsw(shared)
        
        # Ptht путь (с большей регуляризацией)
        ptht = F.relu(self.fc_ptht1(shared))
        ptht = self.ptht_dropout(ptht)  # Больше dropout!
        ptht = self.fc_ptht2(ptht)
        
        return torch.cat([tsw, ptht], dim=1)


class PhenotypePredictionModel(nn.Module):
    def __init__(self, num_snps, num_genes_total, embed_dim, num_heads, ff_dim,
                 num_layers, snp_genes_df, dropout_rate=0.2, dropout_ptht=0.4):
        super(PhenotypePredictionModel, self).__init__()
        self.genetic_encoding = GeneticEncodingBlock(num_genes_total, embed_dim, snp_genes_df)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, num_layers, dropout_rate)
        self.prediction = DualRegularizedPredictionBlock(embed_dim, num_genes_total, 
                                                         dropout_shared=dropout_rate, 
                                                         dropout_ptht=dropout_ptht)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim * num_genes_total, 512),
            nn.ReLU(),
            nn.Linear(512, num_snps)
        )

    def forward(self, X, snp_names, snp_positions, gene_positions):
        X_encoded = self.genetic_encoding(X, snp_names, gene_positions)
        X_transformed = self.transformer(X_encoded)
        outputs = self.prediction(X_transformed)

        flat = X_transformed.reshape(X_transformed.size(0), -1)
        recon = self.reconstruction_head(flat)

        return outputs, recon


class SNPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def freq_encode_snp(data: pd.DataFrame) -> pd.DataFrame:
    encoded = data.copy()
    for column in encoded.columns:
        valid_values = encoded[column][encoded[column] != -1]
        if not valid_values.empty:
            c = Counter(valid_values)
            encoded[column] = encoded[column].apply(
                lambda x: c.get(x, 0) / len(valid_values) if x != -1 else -1
            ).round(3)
        else:
            encoded[column] = -1
    return encoded


def reconstruction_loss(reconstructed, original):
    return F.mse_loss(reconstructed, original)


def select_important_snps_for_ptht(X_train, y_train, X_val, X_test, k=2000):
    """Выбираем наиболее важные SNP для Ptht"""
    selector = SelectKBest(score_func=f_regression, k=min(k, X_train.shape[1]))
    
    X_train_selected = selector.fit_transform(X_train, y_train['Ptht'])
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Получаем имена выбранных SNP
    selected_indices = selector.get_support(indices=True)
    selected_snp_names = X_train.columns[selected_indices].tolist()
    
    logging.info(f"Выбрано {len(selected_snp_names)} SNP для Ptht из {X_train.shape[1]}")
    
    return (pd.DataFrame(X_train_selected, columns=selected_snp_names),
            pd.DataFrame(X_val_selected, columns=selected_snp_names),
            pd.DataFrame(X_test_selected, columns=selected_snp_names),
            selected_snp_names)


def adaptive_focal_loss(preds, targets, gamma=2.0, alpha=0.25):
    """Фокальная loss для борьбы с переобучением"""
    # Вычисляем обычный MSE
    mse_loss = F.mse_loss(preds, targets, reduction='none')
    
    # Вычисляем weights для сложных примеров
    errors = torch.abs(preds - targets)
    weights = (1.0 + errors) ** gamma
    
    # Балансировка через alpha
    weights = alpha * weights
    
    # Взвешенная loss
    loss = torch.mean(weights * mse_loss)
    
    return loss


def hybrid_loss(preds, targets, epoch, max_epochs):
    """Гибридная loss: SmoothL1 для TSW, Focal для Ptht"""
    # TSW: SmoothL1
    loss_tsw = F.smooth_l1_loss(preds[:, 0], targets[:, 0], beta=10.0)
    
    # Ptht: Focal loss (больше внимания сложным примерам)
    loss_ptht = adaptive_focal_loss(preds[:, 1], targets[:, 1], gamma=2.0, alpha=0.3)
    
    # Динамические веса (больше веса Ptht в начале)
    weight_ptht = max(1.5, 2.0 * (1 - epoch / max_epochs))
    weight_tsw = 1.0
    
    return weight_tsw * loss_tsw + weight_ptht * loss_ptht


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log_v6.txt", mode='w'),
        logging.StreamHandler()
    ]
)


if __name__ == "__main__":

    logging.info("=== ЭКСПЕРИМЕНТ v6 (ФОКУС НА Ptht) ===")

    # =======================
    # ЗАГРУЗКА ДАННЫХ
    # =======================

    snp_df = pd.read_csv('dataset/total_df_for_aio_chickpea_28042016_synchro.csv')
    snp_pheno_df = pd.read_csv('dataset/pheno_2016_VIRVFVIR_421_408_synchro.csv')

    snp_2017_df = pd.read_csv('dataset/snp_2017_421.csv')
    pheno_2017_df = pd.read_csv('dataset/Pheno_2017_421.csv')

    snp_genes_df = pd.read_csv('dataset/snp_genes_connections.tsv', sep='\t')
    gene_positions = pd.read_csv(
        'dataset/gene_positions.csv'
    ).set_index('Gene')['Position'].to_dict()

    # =======================
    # ПРИВЕДЕНИЕ snp_genes_df
    # =======================

    snp_genes_df = snp_genes_df.rename(columns={
        'snp_col_name': 'SNP_ID'
    })[['SNP_ID', 'Gene']]

    logging.info(
        f"snp_genes_df подготовлен: SNP={snp_genes_df.shape[0]}, Genes={snp_genes_df['Gene'].nunique()}"
    )

    # =======================
    # ФЕНОТИПЫ
    # =======================

    combined_2016 = pd.DataFrame({
        "Ptht": snp_pheno_df[['Ptht 1', 'Ptht 2', 'Ptht 3', 'Ptht 4', 'Ptht 5']].mean(axis=1),
        "TSW": snp_pheno_df["TSW"],
        "SNP_ID": snp_pheno_df["SNP ID"].astype(str)
    })

    combined_2017 = pd.DataFrame({
        "Ptht": pheno_2017_df[['Ptht 1', 'Ptht 2', 'Ptht 3', 'Ptht 4', 'Ptht 5']].mean(axis=1),
        "TSW": pheno_2017_df["TSW"],
        "SNP_ID": pheno_2017_df["SNP ID"].astype(str)
    })

    # =======================
    # ФИЛЬТРАЦИЯ SNP
    # =======================

    snp_cols = snp_genes_df['SNP_ID'].astype(str).tolist()

    snp_df = snp_df.filter(items=["SNP ID"] + snp_cols)
    snp_2017_df = snp_2017_df.filter(items=["SNP ID"] + snp_cols)

    snp_df = snp_df.merge(
        combined_2016, left_on="SNP ID", right_on="SNP_ID", how="inner"
    )
    snp_2017_df = snp_2017_df.merge(
        combined_2017, left_on="SNP ID", right_on="SNP_ID", how="inner"
    )

    snp_df.drop(columns=["SNP ID", "SNP_ID"], inplace=True)
    snp_2017_df.drop(columns=["SNP ID", "SNP_ID"], inplace=True)

    full_df = pd.concat([snp_df, snp_2017_df], axis=0).reset_index(drop=True)

    logging.info(f"Объединённый датасет нута: {full_df.shape}")

    # =======================
    # X / y
    # =======================

    phenotype_cols = ["TSW", "Ptht"]

    X = full_df.drop(columns=phenotype_cols)
    y = full_df[phenotype_cols]

    # =======================
    # FEATURE SELECTION ДЛЯ Ptht
    # =======================

    # Кодирование SNP
    X_encoded = freq_encode_snp(X.copy())

    # Разделение данных
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.5, random_state=42
    )

    # ВЫБОР ВАЖНЫХ SNP ДЛЯ Ptht
    X_train_selected, X_val_selected, X_test_selected, selected_snp_names = select_important_snps_for_ptht(
        X_train, y_train, X_val, X_test, k=3000  # 3000 вместо 5783
    )
    
    # Обновляем snp_names
    snp_names = selected_snp_names
    
    # Обновляем snp_genes_df для выбранных SNP
    snp_genes_df = snp_genes_df[snp_genes_df['SNP_ID'].isin(selected_snp_names)].reset_index(drop=True)
    logging.info(f"SNP в snp_genes_df после фильтрации: {snp_genes_df.shape[0]}")

    # Аугментация
    alpha = 0.0625
    num_augmented_samples = 150

    X_train_selected, y_train = mixup_data(
        X_train_selected,
        y_train,
        lam=alpha,
        aug_count=num_augmented_samples / X_train_selected.shape[0]
    )

    # =======================
    # DATASET / DATALOADER
    # =======================

    train_dataset = SNPDataset(X_train_selected, y_train)
    val_dataset = SNPDataset(X_val_selected, y_val)
    test_dataset = SNPDataset(X_test_selected, y_test)

    batch_size = 32
    num_workers = 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # =======================
    # МОДЕЛЬ
    # =======================

    num_snps = X_train_selected.shape[1]
    num_genes_total = snp_genes_df['Gene'].nunique()

    # Параметры
    embed_dim = 32
    num_heads = 1
    ff_dim = 64
    num_layers = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Используемое устройство: {device}")

    dropout_rate = 0.2
    dropout_ptht = 0.5  # Высокий dropout ТОЛЬКО для Ptht

    model = PhenotypePredictionModel(
        num_snps,
        num_genes_total,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        snp_genes_df,
        dropout_rate=dropout_rate,
        dropout_ptht=dropout_ptht
    ).to(device)

    # =======================
    # OPTIM / LOSS
    # =======================

    learning_rate = 0.001
    weight_decay = 0.001

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # =======================
    # ОБУЧЕНИЕ
    # =======================

    num_epochs = 100
    best_val_corr_ptht = 0.0
    best_val_corr_tsw = 0.0
    patience_ptht = 8  # Отдельный patience для Ptht
    patience_tsw = 12  # Больше patience для TSW
    patience_counter_ptht = 0
    patience_counter_tsw = 0
    best_model_path = "best_model_chickpea_v6.pth"

    alpha_recon = 0.005

    logging.info("=== НАЧАЛО ОБУЧЕНИЯ v6 ===")
    logging.info(f"Параметры: lr={learning_rate}, wd={weight_decay}")
    logging.info(f"Dropout Ptht: {dropout_ptht} (очень высокий!)")
    logging.info(f"Feature selection: {num_snps} SNP (из 5783)")
    logging.info(f"Гибридная loss: SmoothL1 для TSW, Focal для Ptht")
    logging.info(f"Отдельный early stopping для Ptht и TSW")

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        total_train_loss = 0
        train_preds, train_targets = [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs, recon = model(batch_X, snp_names, {}, gene_positions)

            # ГИБРИДНАЯ LOSS
            loss_pred = hybrid_loss(outputs, batch_y, epoch, num_epochs)
            loss_recon = reconstruction_loss(recon, batch_X)
            loss = loss_pred + alpha_recon * loss_recon

            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_train_loss += loss.item()
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(batch_y.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)

        # ОЦЕНКА НА ВАЛИДАЦИИ
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                outputs_val, _ = model(batch_X_val, snp_names, {}, gene_positions)
                val_preds.append(outputs_val.cpu().numpy())
                val_targets.append(batch_y_val.cpu().numpy())

        val_preds_np = np.concatenate(val_preds, axis=0)
        val_targets_np = np.concatenate(val_targets, axis=0)
        
        val_corr_tsw = np.corrcoef(val_targets_np[:, 0], val_preds_np[:, 0])[0, 1]
        val_corr_ptht = np.corrcoef(val_targets_np[:, 1], val_preds_np[:, 1])[0, 1]
        
        scheduler.step()

        # Вывод каждые 5 эпох
        if (epoch + 1) % 5 == 0 or epoch == 0:
            train_preds_np = np.concatenate(train_preds, axis=0)
            train_targets_np = np.concatenate(train_targets, axis=0)
            train_corr_tsw = np.corrcoef(train_targets_np[:, 0], train_preds_np[:, 0])[0, 1]
            train_corr_ptht = np.corrcoef(train_targets_np[:, 1], train_preds_np[:, 1])[0, 1]
            
            logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
            logging.info(f"  Train: TSW={train_corr_tsw:.4f}, Ptht={train_corr_ptht:.4f}")
            logging.info(f"  Val:   TSW={val_corr_tsw:.4f}, Ptht={val_corr_ptht:.4f}")

        # ОТДЕЛЬНЫЙ EARLY STOPPING ДЛЯ Ptht И TSW
        save_model = False
        
        if val_corr_ptht > best_val_corr_ptht:
            best_val_corr_ptht = val_corr_ptht
            patience_counter_ptht = 0
            save_model = True
            reason = f"лучшая корреляция Ptht: {val_corr_ptht:.4f}"
        else:
            patience_counter_ptht += 1
            
        if val_corr_tsw > best_val_corr_tsw:
            best_val_corr_tsw = val_corr_tsw
            patience_counter_tsw = 0
            if not save_model:  # Сохраняем если улучшился TSW, но не Ptht
                save_model = True
                reason = f"лучшая корреляция TSW: {val_corr_tsw:.4f}"
        else:
            patience_counter_tsw += 1
        
        if save_model:
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Epoch {epoch+1}: СОХРАНЕНО! ({reason})")
        
        # Проверяем early stopping
        if patience_counter_ptht >= patience_ptht and patience_counter_tsw >= patience_tsw:
            logging.info(f"Ранняя остановка на эпохе {epoch+1} (Ptht: {patience_counter_ptht}, TSW: {patience_counter_tsw})")
            break

    logging.info("=== ОБУЧЕНИЕ v6 ЗАВЕРШЕНО ===")
    
    # ФИНАЛЬНАЯ ОЦЕНКА
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
            outputs_test, _ = model(batch_X_test, snp_names, {}, gene_positions)
            test_preds.append(outputs_test.cpu().numpy())
            test_targets.append(batch_y_test.cpu().numpy())
    
    test_preds_np = np.concatenate(test_preds, axis=0)
    test_targets_np = np.concatenate(test_targets, axis=0)
    
    test_corr_tsw = np.corrcoef(test_targets_np[:, 0], test_preds_np[:, 0])[0, 1]
    test_corr_ptht = np.corrcoef(test_targets_np[:, 1], test_preds_np[:, 1])[0, 1]
    
    test_mse_tsw = np.mean((test_targets_np[:, 0] - test_preds_np[:, 0]) ** 2)
    test_mse_ptht = np.mean((test_targets_np[:, 1] - test_preds_np[:, 1]) ** 2)
    
    logging.info("=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ v6 ===")
    logging.info(f"Test TSW:  Corr={test_corr_tsw:.4f}, MSE={test_mse_tsw:.2f}, RMSE={np.sqrt(test_mse_tsw):.2f}")
    logging.info(f"Test Ptht: Corr={test_corr_ptht:.4f}, MSE={test_mse_ptht:.2f}, RMSE={np.sqrt(test_mse_ptht):.2f}")
    logging.info(f"Лучшая валидация: TSW={best_val_corr_tsw:.4f}, Ptht={best_val_corr_ptht:.4f}")