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

# from mixup_module import mixup_data
# from analysis_module import compute_corr, print_correlation_statistics, print_data_statistics


import pandas as pd
import random
import numpy as np

from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def compute_corr(y_true, y_pred, phenotype_names):
    """
    Рассчитывает корреляцию Пирсона для каждой фенотипической переменной
    для одного набора данных (например, train, val, или test).
    Args:
        y_true (np.ndarray или pd.DataFrame): Фактические значения целевых переменных.
        y_pred (np.ndarray): Предсказанные значения целевых переменных.
        phenotype_names (list): Список названий фенотипических признаков.
    Returns:
        dict: Словарь, где ключ - название признака, значение - корреляция Пирсона.
    """
    correlations = {}
    # Убедимся, что y_true является массивом NumPy для единообразия индексации
    # Если y_true - DataFrame, извлекаем его значения
    y_true_np = y_true.values if isinstance(y_true, pd.DataFrame) else y_true
    
    for i, name in enumerate(phenotype_names):
        try:
            # pearsonr ожидает одномерные массивы. Выбираем i-й столбец.
            corr, _ = pearsonr(y_true_np[:, i], y_pred[:, i])
            correlations[name] = corr
        except Exception as e:
            # Если, например, данные однородны (std=0), pearsonr может вызвать ошибку.
            # В таком случае записываем NaN и выводим предупреждение.
            correlations[name] = float('nan')
            print(f"Предупреждение: Не удалось рассчитать корреляцию для признака '{name}': {e}")
    return correlations

def print_correlation_statistics(correlations_dict, set_name):
    """
    Форматирует и выводит статистики корреляций для данного набора данных.
    Args:
        correlations_dict (dict): Словарь корреляций, возвращенный compute_corr.
        set_name (str): Название набора данных (например, 'train', 'val', 'test').
    Returns:
        str: Отформатированная строка со статистиками.
    """
    output = [f"Корреляции для {set_name} набора:"] # Изменил заголовок для ясности
    for name, corr in correlations_dict.items():
        output.append(f"  {name}: {corr:.4f}")
    return "\n".join(output)

# --- Оригинальные функции, которые не вызывали ошибку, но могут быть переименованы для ясности ---

def plot_histograms(y_train, y_test, y_train_pred, y_test_pred):
    """
    Построение гистограмм фактических и предсказанных значений.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(y_train.iloc[:, 0], bins=50, alpha=0.7, label='TSW_train')
    plt.hist(y_train.iloc[:, 1], bins=50, alpha=0.7, label='Ptht 1_train')
    plt.hist(y_test.iloc[:, 0], bins=50, alpha=0.7, label='TSW_test')
    plt.hist(y_test.iloc[:, 1], bins=50, alpha=0.7, label='Ptht 1_test')
    plt.legend()
    plt.title('Гистограмма целевых переменных')

    plt.subplot(1, 2, 2)
    plt.hist(y_train_pred[:, 0], bins=50, alpha=0.7, label='TSW_train_pred')
    plt.hist(y_train_pred[:, 1], bins=50, alpha=0.7, label='Ptht 1_train_pred')
    plt.hist(y_test_pred[:, 0], bins=50, alpha=0.7, label='TSW_test_pred')
    plt.hist(y_test_pred[:, 1], bins=50, alpha=0.7, label='Ptht 1_test_pred')
    plt.legend()
    plt.title('Гистограмма предсказанных значений')

    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, target_names=['TSW', 'Ptht 1']):
    """
    Построение графиков "фактические vs предсказанные" для каждой целевой переменной.
    """
    num_targets = y_train.shape[1]

    for i in range(num_targets):
        plt.figure(figsize=(12, 6))

        # Обучающая выборка
        plt.subplot(1, 2, 1)
        plt.scatter(y_train.iloc[:, i], y_train_pred[:, i], alpha=0.5)
        
        # Определяем диапазон для красной линии
        min_val = min(y_train.iloc[:, i].min(), y_train_pred[:, i].min())
        max_val = max(y_train.iloc[:, i].max(), y_train_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        plt.title(f'Фактические vs предсказанные (train) для {target_names[i]}')
        plt.xlabel('Фактические значения')
        plt.ylabel('Предсказанные значения')

        # Тестовая выборка
        plt.subplot(1, 2, 2)
        plt.scatter(y_test.iloc[:, i], y_test_pred[:, i], alpha=0.5)
        
        # Определяем диапазон для красной линии
        min_val = min(y_test.iloc[:, i].min(), y_test_pred[:, i].min())
        max_val = max(y_test.iloc[:, i].max(), y_test_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        plt.title(f'Фактические vs предсказанные (test) для {target_names[i]}')
        plt.xlabel('Фактические значения')
        plt.ylabel('Предсказанные значения')

        plt.tight_layout()
        plt.show()

def print_data_statistics(y_train, y_test, y_train_pred, y_test_pred):
    """
    Печатает статистики (среднее и стандартное отклонение) для обучающих и тестовых данных.
    Эта функция была переименована из 'print_statistics' для предотвращения конфликта имен
    с функцией, используемой для вывода корреляций в логах.
    """
    print("\nСтатистика для y_train:")
    print(f"Среднее: {np.mean(y_train, axis=0)}, Стандартное отклонение: {np.std(y_train, axis=0)}")

    print("\nСтатистика для y_test:")
    print(f"Среднее: {np.mean(y_test, axis=0)}, Стандартное отклонение: {np.std(y_test, axis=0)}")

    print("\nСтатистика для y_train_pred:")
    print(f"Среднее: {np.mean(y_train_pred, axis=0)}, Стандартное отклонение: {np.std(y_train_pred, axis=0)}")

    print("\nСтатистика для y_test_pred:")
    print(f"Среднее: {np.mean(y_test_pred, axis=0)}, Стандартное отклонение: {np.std(y_test_pred, axis=0)}")
    

def mixup_data(X, y, aug_count=0.2, lam=0.5):
    augmented_data_x = pd.DataFrame(columns=X.columns) # Инициализация с колонками для сохранения типов
    augmented_data_y = pd.DataFrame(columns=y.columns) # Инициализация с колонками

    num_aug = int(aug_count * X.shape[0])
    for _ in range(num_aug):
        # sample(n=2) возвращает DataFrame, index сохраняет исходные индексы
        tmp_x = X.sample(n=2)
        tmp_y = y.loc[tmp_x.index]

        # Линейная комбинация признаков (без округления и приведения к int)
        new_row_x = lam * tmp_x.iloc[0] + (1 - lam) * tmp_x.iloc[1]
        
        # Линейная комбинация целевых переменных
        new_row_y = lam * tmp_y.iloc[0] + (1 - lam) * tmp_y.iloc[1]

        # Использование pd.concat для добавления строк (рекомендуемый способ)
        augmented_data_x = pd.concat([augmented_data_x, pd.DataFrame([new_row_x], columns=X.columns)], ignore_index=True)
        augmented_data_y = pd.concat([augmented_data_y, pd.DataFrame([new_row_y], columns=y.columns)], ignore_index=True)

    return pd.concat([X, augmented_data_x], ignore_index=True), \
           pd.concat([y, augmented_data_y], ignore_index=True)


def count_parameters(model):
    """
    Возвращает общее количество обучаемых параметров модели,
    а также детализацию по каждому блоку.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Общее количество обучаемых параметров: {total_params:,}")
    print("\n" + "="*60 + "\n")
    
    print("Детализация по блокам:")
    print("-" * 60)
    
    # Счетчики по блокам
    genetic_params = 0
    transformer_params = 0
    prediction_params = 0
    
    # Подсчет параметров для каждого блока
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            if 'genetic_encoding' in name:
                genetic_params += num_params
            elif 'transformer' in name:
                transformer_params += num_params
            elif 'prediction' in name:
                prediction_params += num_params
            
            # Вывод информации по каждому слою
            print(f"{name:50} | {num_params:12,} | {list(param.shape)}")
    
    print("-" * 60)
    print(f"GeneticEncodingBlock:   {genetic_params:12,} параметров")
    print(f"TransformerBlock:       {transformer_params:12,} параметров")
    print(f"PredictionBlock:        {prediction_params:12,} параметров")
    print("=" * 60)
    
    return total_params


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


class PredictionBlock(nn.Module):
    def __init__(self, embed_dim, num_classes, num_genes, dropout_rate = 0.111616):
        super(PredictionBlock, self).__init__()
        self.initial_num_genes = num_genes
        self.fc1 = nn.Linear(embed_dim * self.initial_num_genes, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        if x.size(1) == 0:
            return torch.zeros(x.size(0), self.fc2.out_features, device=x.device)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PhenotypePredictionModel(nn.Module):
    def __init__(self, num_snps, num_genes_total, embed_dim, num_heads, ff_dim,
                 num_layers, num_classes, snp_genes_df, dropout_rate=0.2):
        super(PhenotypePredictionModel, self).__init__()
        self.genetic_encoding = GeneticEncodingBlock(num_genes_total, embed_dim, snp_genes_df)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, num_layers, dropout_rate)
        self.prediction = PredictionBlock(embed_dim, num_classes, num_genes_total, dropout_rate)

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


def multitask_loss(preds, targets, log_sigma1, log_sigma2):
    loss1 = F.mse_loss(preds[:, 0], targets[:, 0])
    loss2 = F.mse_loss(preds[:, 1], targets[:, 1])
    loss = (1 / (2 * torch.exp(log_sigma1) ** 2)) * loss1 + \
           (1 / (2 * torch.exp(log_sigma2) ** 2)) * loss2 + \
           log_sigma1 + log_sigma2
    return loss


def reconstruction_loss(reconstructed, original):
    return F.mse_loss(reconstructed, original)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    PROCESSED_DATA_DIR = r'C:\GitLabRepositories\phenotype_predictor\rice'

    FILTERED_GENOTYPES_FILE = os.path.join(PROCESSED_DATA_DIR, 'df_numeric_genotypes_filtered.csv')
    CLEANED_PHENO_FILE = os.path.join(PROCESSED_DATA_DIR, 'df_phenotypes_cleaned.csv')
    SNP_COORDS_FILE = os.path.join(PROCESSED_DATA_DIR, 'df_snp_coords_for_model.csv')
    GENE_POSITIONS_FILE = os.path.join(PROCESSED_DATA_DIR, 'df_gene_positions.csv')

    logging.info("--- Загрузка подготовленных данных ---")
    try:
        df_numeric_genotypes_filtered = pd.read_csv(FILTERED_GENOTYPES_FILE)
        df_phenotypes_cleaned = pd.read_csv(CLEANED_PHENO_FILE)
        df_snp_coords_for_model = pd.read_csv(SNP_COORDS_FILE)
        df_gene_positions_from_gff = pd.read_csv(GENE_POSITIONS_FILE)

        logging.info("Данные успешно загружены.")
    except FileNotFoundError as e:
        logging.error(f"Ошибка: Файл не найден. Убедитесь, что '{e.filename}' существует по указанному пути.")
        exit()
    except Exception as e:
        logging.error(f"Произошла ошибка при загрузке данных: {e}")
        exit()

    logging.info("--- Подготовка snp_genes_df и gene_positions ---")

    df_snp_coords_for_model.columns = df_snp_coords_for_model.columns.str.strip()
    df_gene_positions_from_gff.columns = df_gene_positions_from_gff.columns.str.strip()

    df_snp_coords_for_model['Chromosome'] = df_snp_coords_for_model['Chromosome'].astype(str).str.replace('chr', '', regex=False)
    df_gene_positions_from_gff['CHROM'] = df_gene_positions_from_gff['CHROM'].astype(str).str.replace('chr', '', regex=False)

    df_snp_coords_for_model.rename(columns={'Chromosome': 'CHROM'}, inplace=True)

    merged_snp_gene_candidates = pd.merge(
        df_snp_coords_for_model,
        df_gene_positions_from_gff,
        on='CHROM',
        how='inner'
    )

    snp_genes_df = merged_snp_gene_candidates[
        (merged_snp_gene_candidates['SNP_POS'] >= merged_snp_gene_candidates['START']) &
        (merged_snp_gene_candidates['SNP_POS'] <= merged_snp_gene_candidates['END'])
    ].copy()

    snp_genes_df = snp_genes_df[['SNP_ID', 'GENE_ID', 'CHROM', 'SNP_POS']].copy()
    snp_genes_df.rename(columns={'GENE_ID': 'Gene'}, inplace=True)

    logging.info(f"Сформирован snp_genes_df. Количество SNP, сопоставленных с генами: {len(snp_genes_df)}")

    gene_positions = df_gene_positions_from_gff.set_index('GENE_ID')['START'].to_dict()

    df_combined_data = pd.merge(
        df_numeric_genotypes_filtered,
        df_phenotypes_cleaned,
        on='Accession ID',
        how='inner'
    )

    logging.info(f"Объединенные генотипы и фенотипы. Размерность: {df_combined_data.shape}")

    phenotype_cols = ['Plant height (cm)', 'Tiller number']
    available_pheno_cols = [col for col in phenotype_cols if col in df_combined_data.columns]
    phenotype_cols = available_pheno_cols

    all_other_cols = df_phenotypes_cleaned.columns.drop('Accession ID').tolist()
    X = df_combined_data.drop(columns=['Accession ID'] + all_other_cols)
    y = df_combined_data[phenotype_cols]

    snp_names = X.columns.tolist()
    snp_positions = df_snp_coords_for_model.set_index('SNP_ID')['SNP_POS'].to_dict()

    X_encoded = freq_encode_snp(X.copy())

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.5, random_state=42
    )

    alpha = 0.0625
    num_augmented_samples = 100
    X_train, y_train = mixup_data(
        X_train.copy(),
        y_train.copy(),
        lam=alpha,
        aug_count=num_augmented_samples / X_train.shape[0]
    )

    train_dataset = SNPDataset(X_train, y_train)
    val_dataset = SNPDataset(X_val, y_val)
    test_dataset = SNPDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_snps = X_encoded.shape[1]
    num_genes_total = len(snp_genes_df['Gene'].unique())
    embed_dim = 64
    num_heads = 2
    ff_dim = 224
    num_layers = 2
    num_classes = y.shape[1]

    device = torch.device("cpu")
    logging.info(f"Используемое устройство: {device}")

    dropout_rate = 0.111616
    model = PhenotypePredictionModel(
        num_snps, num_genes_total, embed_dim, num_heads,
        ff_dim, num_layers, num_classes, snp_genes_df,
        dropout_rate=dropout_rate
    ).to(device)
    total_params = count_parameters(model)
    log_sigma1 = nn.Parameter(torch.zeros(1, device=device))
    log_sigma2 = nn.Parameter(torch.zeros(1, device=device))

    learning_rate = 0.0002962151658830348
    weight_decay = 0.00004191711516695204
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': [log_sigma1, log_sigma2], 'lr': learning_rate}
    ], lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=7, min_lr=1e-6
    )

    num_epochs = 200
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    best_model_path = os.path.join(PROCESSED_DATA_DIR, 'best_rice_report_1_model.pth')

    alpha_start = 0.01
    warmup_epochs = 30


    logging.info(f"--- Начало обучения ({num_epochs} эпох) ---")

    for epoch in range(num_epochs):
        alpha_recon = max(0.0, alpha_start * (1 - epoch / warmup_epochs))
        logging.info(f"Epoch {epoch+1}: alpha_recon={alpha_recon:.6f}")
        
        start_time = time.time()
        model.train()
        total_train_loss = 0
        train_preds, train_targets = [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs, recon = model(batch_X, X_encoded.columns.tolist(), snp_positions, gene_positions)
            loss_pred = multitask_loss(outputs, batch_y, log_sigma1, log_sigma2)
            loss_recon = reconstruction_loss(recon, batch_X)
            loss = loss_pred + alpha_recon * loss_recon

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(batch_y.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        logging.info(f"[Train] loss_pred={loss_pred.item():.4f}, loss_recon={loss_recon.item():.4f}, alpha_recon={alpha_recon}")

        model.eval()
        total_val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                outputs_val, recon_val = model(batch_X_val, X_encoded.columns.tolist(), snp_positions, gene_positions)
                loss_val_pred = multitask_loss(outputs_val, batch_y_val, log_sigma1, log_sigma2)
                loss_val_recon = reconstruction_loss(recon_val, batch_X_val)
                loss_val = loss_val_pred + alpha_recon * loss_val_recon
                total_val_loss += loss_val.item()
                val_preds.append(outputs_val.cpu().numpy())
                val_targets.append(batch_y_val.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        total_test_loss = 0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for batch_X_test, batch_y_test in test_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                outputs_test, recon_test = model(batch_X_test, X_encoded.columns.tolist(), snp_positions, gene_positions)
                loss_test_pred = multitask_loss(outputs_test, batch_y_test, log_sigma1, log_sigma2)
                loss_test_recon = reconstruction_loss(recon_test, batch_X_test)
                loss_test = loss_test_pred + alpha_recon * loss_test_recon
                total_test_loss += loss_test.item()
                test_preds.append(outputs_test.cpu().numpy())
                test_targets.append(batch_y_test.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_preds = np.concatenate(test_preds, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)

        epoch_time = time.time() - start_time

        train_corr = compute_corr(np.concatenate(train_targets, axis=0), np.concatenate(train_preds, axis=0), phenotype_cols)
        val_corr = compute_corr(np.concatenate(val_targets, axis=0), np.concatenate(val_preds, axis=0), phenotype_cols)
        test_corr = compute_corr(test_targets, test_preds, phenotype_cols)

        log_message = (
            f"\nEpoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}\n"
            f"{print_correlation_statistics(train_corr, 'train')}\n"
            f"{print_correlation_statistics(val_corr, 'val')}\n"
            f"{print_correlation_statistics(test_corr, 'test')}\n"
            f"Log Sigma 1 (Plant height): {log_sigma1.item():.4f}, Log Sigma 2 (Tiller number): {log_sigma2.item():.4f}\n"
            f"{'-'*80}"
        )
        logging.info(log_message)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Валидационная потеря улучшилась. Сохранена лучшая модель с потерей: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logging.info(f"Валидационная потеря не улучшилась. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info(f"Ранняя остановка на эпохе {epoch+1}.")
                break

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    logging.info("--- Обучение завершено ---")
    logging.info(f"Лучшая модель загружена из {best_model_path}")

    final_test_loss = 0
    final_test_preds, final_test_targets = [], []
    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
            outputs, recon = model(batch_X_test, X_encoded.columns.tolist(), snp_positions, gene_positions)
            loss = multitask_loss(outputs, batch_y_test, log_sigma1, log_sigma2) + alpha_recon * reconstruction_loss(recon, batch_X_test)
            final_test_loss += loss.item()
            final_test_preds.append(outputs.cpu().numpy())
            final_test_targets.append(batch_y_test.cpu().numpy())

    final_avg_test_loss = final_test_loss / len(test_loader)
    final_test_preds = np.concatenate(final_test_preds, axis=0)
    final_test_targets = np.concatenate(final_test_targets, axis=0)
    final_test_corr = compute_corr(final_test_targets, final_test_preds, phenotype_cols)

    logging.info(f"\n--- Итоговая оценка на тестовом наборе с лучшей моделью ---")
    logging.info(f"Final Test Loss (Best Model): {final_avg_test_loss:.4f}")
    logging.info(f"{print_correlation_statistics(final_test_corr, 'final test (best model)')}")
