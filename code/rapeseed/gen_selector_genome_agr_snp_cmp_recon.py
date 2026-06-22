import time
from collections import Counter
import os
import random
import gc
import pickle
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from math import log
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================
# УСТАНОВКА RANDOM SEED ДЛЯ ВОСПРОИЗВОДИМОСТИ
# ============================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================
# ФУНКЦИИ ДЛЯ ОТБОРА ГЕНОВ
# ============================================

def select_genes_weighted(snp_genes_df, top_n=500, strategy='balanced'):
    """
    Расширенная стратегия отбора генов
    
    strategy:
        - 'balanced': равномерно по квартилям
        - 'weighted': больше генов из квартилей с большим количеством SNP
        - 'top': только гены с наибольшим количеством SNP
        - 'proportional': пропорционально размеру квартиля
    """
    print(f"\n📊 Отбор генов по стратегии: {strategy}")
    print(f"   Целевое количество генов: {top_n}")
    
    # Подсчет SNP на ген
    snp_per_gene = snp_genes_df.groupby('Gene')['SNP_ID'].nunique().sort_values(ascending=False)
    
    # Разбиваем на квартили
    quantiles = np.percentile(snp_per_gene.values, [25, 50, 75])
    
    # Распределяем гены по квартилям
    quantile_genes = {0: [], 1: [], 2: [], 3: []}
    for gene, count in snp_per_gene.items():
        if count <= quantiles[0]:
            quantile_genes[0].append(gene)  # 0-25%
        elif count <= quantiles[1]:
            quantile_genes[1].append(gene)  # 25-50%
        elif count <= quantiles[2]:
            quantile_genes[2].append(gene)  # 50-75%
        else:
            quantile_genes[3].append(gene)  # 75-100%
    
    print(f"\n   - Размеры квартилей:")
    for i in range(4):
        upper = quantiles[i] if i < 3 else 'max'
        print(f"      Квартиль {i+1} (до {upper} SNP): {len(quantile_genes[i])} генов")
    
    if strategy == 'balanced':
        # Равномерно по квартилям
        weights = [0.25, 0.25, 0.25, 0.25]
        
    elif strategy == 'weighted':
        # Больше генов из квартилей с большим количеством SNP
        weights = [0.1, 0.2, 0.3, 0.4]
        
    elif strategy == 'top':
        # Только гены с наибольшим количеством SNP
        selected_genes = list(snp_per_gene.head(top_n).index)
        return selected_genes, {'strategy': 'top', 'selected_count': len(selected_genes)}
        
    elif strategy == 'proportional':
        # Пропорционально размеру квартиля
        total = sum(len(quantile_genes[i]) for i in range(4))
        weights = [len(quantile_genes[i]) / total for i in range(4)]
    
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
    
    # Нормализуем веса
    weights = np.array(weights) / sum(weights)
    n_per_quantile = [int(top_n * w) for w in weights]
    
    # Корректируем, чтобы сумма была равна top_n
    diff = top_n - sum(n_per_quantile)
    for i in range(diff):
        n_per_quantile[i % 4] += 1
    
    print(f"\n   - Целевое распределение:")
    for i in range(4):
        print(f"      Квартиль {i+1}: {n_per_quantile[i]} генов (вес={weights[i]:.2f})")
    
    selected_genes = []
    for i in range(4):
        n_to_take = min(n_per_quantile[i], len(quantile_genes[i]))
        if n_to_take > 0:
            selected = random.sample(quantile_genes[i], n_to_take)
            selected_genes.extend(selected)
            print(f"      Квартиль {i+1}: взято {len(selected)} из {len(quantile_genes[i])} генов")
    
    print(f"\n   📊 Результаты отбора:")
    print(f"      - Отобрано: {len(selected_genes)} генов")
    
    selected_df = snp_genes_df[snp_genes_df['Gene'].isin(selected_genes)]
    print(f"      - SNP в отобранных генах: {selected_df['SNP_ID'].nunique():,}")
    print(f"      - Среднее SNP на отобранный ген: {selected_df.groupby('Gene')['SNP_ID'].nunique().mean():.1f}")
    
    return selected_genes, {'strategy': strategy, 'selected_count': len(selected_genes)}


def select_genes_by_genome_coverage(snp_genes_df, genes_gff_df, top_n=3000):
    """
    Отбор генов с сохранением пропорций распределения по хромосомам.
    
    Параметры:
    ----------
    snp_genes_df : DataFrame с колонками SNP_ID и Gene
    genes_gff_df : DataFrame с информацией о расположении генов (GENE_NAME, CHR)
    top_n : количество генов для отбора
    
    Возвращает:
    -----------
    selected_genes : list отобранных генов
    gene_stats : словарь со статистикой
    """
    print(f"\n📊 Отбор генов по геномному покрытию")
    print(f"   Целевое количество генов: {top_n}")
    
    # Получаем уникальные гены с их хромосомами
    gene_chromosomes = genes_gff_df[['GENE_NAME', 'CHR']].drop_duplicates()
    gene_chromosomes = gene_chromosomes[gene_chromosomes['GENE_NAME'].isin(snp_genes_df['Gene'].unique())]
    
    # Подсчет генов по хромосомам
    chrom_counts = gene_chromosomes['CHR'].value_counts()
    total_genes = len(gene_chromosomes)
    
    print(f"\n   - Распределение генов по хромосомам:")
    for chrom, count in chrom_counts.items():
        print(f"      {chrom}: {count:,} генов ({count/total_genes*100:.1f}%)")
    
    # Вычисляем целевое количество генов на хромосому
    target_per_chrom = {}
    for chrom, count in chrom_counts.items():
        target_per_chrom[chrom] = max(1, int(top_n * count / total_genes))
    
    # Корректируем, чтобы сумма была равна top_n
    diff = top_n - sum(target_per_chrom.values())
    if diff > 0:
        # Добавляем недостающие гены к хромосомам с наибольшим количеством
        sorted_chroms = sorted(chrom_counts.items(), key=lambda x: x[1], reverse=True)
        for i in range(diff):
            target_per_chrom[sorted_chroms[i % len(sorted_chroms)][0]] += 1
    
    print(f"\n   - Целевое распределение по хромосомам:")
    for chrom, target in target_per_chrom.items():
        print(f"      {chrom}: {target} генов")
    
    # Для каждой хромосомы отбираем гены с наибольшим количеством SNP
    selected_genes = []
    genes_per_chrom = gene_chromosomes.groupby('CHR')['GENE_NAME'].apply(list).to_dict()
    
    # Подсчет SNP на ген
    snp_per_gene = snp_genes_df.groupby('Gene')['SNP_ID'].nunique()
    
    for chrom, target in target_per_chrom.items():
        if chrom in genes_per_chrom:
            chrom_genes = genes_per_chrom[chrom]
            # Сортируем гены по количеству SNP
            chrom_genes_sorted = sorted(chrom_genes, key=lambda g: snp_per_gene.get(g, 0), reverse=True)
            selected = chrom_genes_sorted[:target]
            selected_genes.extend(selected)
            print(f"      {chrom}: взято {len(selected)} из {len(chrom_genes)} генов")
    
    print(f"\n   📊 Результаты отбора:")
    print(f"      - Отобрано: {len(selected_genes)} генов")
    
    selected_df = snp_genes_df[snp_genes_df['Gene'].isin(selected_genes)]
    print(f"      - SNP в отобранных генах: {selected_df['SNP_ID'].nunique():,}")
    print(f"      - Среднее SNP на отобранный ген: {selected_df.groupby('Gene')['SNP_ID'].nunique().mean():.1f}")
    
    return selected_genes, {'strategy': 'genome_coverage', 'selected_count': len(selected_genes), 'target_per_chrom': target_per_chrom}


def select_genes_uniform_by_position(snp_genes_df, genes_gff_df, top_n=3000):
    """
    Отбор генов с равномерным покрытием по физическим позициям генома.
    
    Параметры:
    ----------
    snp_genes_df : DataFrame с колонками SNP_ID и Gene
    genes_gff_df : DataFrame с информацией о расположении генов (GENE_NAME, CHR, GENE_START, GENE_END)
    top_n : количество генов для отбора
    
    Возвращает:
    -----------
    selected_genes : list отобранных генов
    gene_stats : словарь со статистикой
    """
    print(f"\n📊 Отбор генов по равномерному покрытию генома")
    print(f"   Целевое количество генов: {top_n}")
    
    # Функция нормализации имени гена (как в основном коде)
    def normalize_gene_name(name):
        if name.startswith('gene-'):
            return name[5:]
        return name
    
    # Создаем копию и нормализуем имена в GFF
    genes_info = genes_gff_df[['GENE_NAME', 'CHR', 'GENE_START', 'GENE_END']].drop_duplicates().copy()
    genes_info['GENE_NAME_NORM'] = genes_info['GENE_NAME'].apply(normalize_gene_name)
    
    # Получаем список генов из SNP данных (с префиксом gene-)
    snp_genes_set = set(snp_genes_df['Gene'].unique())
    
    # Также создаем нормализованную версию для сопоставления
    snp_genes_norm = {normalize_gene_name(g): g for g in snp_genes_set}
    
    # Фильтруем только гены, которые есть в SNP данных (по нормализованному имени)
    genes_info['IN_SNP_DATA'] = genes_info['GENE_NAME_NORM'].isin(snp_genes_norm.keys())
    genes_info = genes_info[genes_info['IN_SNP_DATA']].copy()
    
    # Добавляем исходное имя из SNP данных
    genes_info['GENE_NAME_ORIG'] = genes_info['GENE_NAME_NORM'].map(snp_genes_norm)
    
    print(f"   - Генов в GFF: {len(genes_gff_df['GENE_NAME'].unique()):,}")
    print(f"   - Генов в SNP данных: {len(snp_genes_set):,}")
    print(f"   - Пересечение: {len(genes_info):,} генов")
    
    if len(genes_info) == 0:
        print("   ⚠️  Нет генов для отбора! Используем weighted стратегию")
        return select_genes_weighted(snp_genes_df, top_n=top_n, strategy='weighted')
    
    # Вычисляем среднюю позицию гена
    genes_info['GENE_MID'] = (genes_info['GENE_START'] + genes_info['GENE_END']) / 2
    
    # Вычисляем нормализованную позицию для каждого гена (по всему геному)
    # Сначала вычисляем смещения для хромосом
    chrom_lengths = genes_info.groupby('CHR')['GENE_MID'].max()
    chrom_offsets = {}
    offset = 0
    for chrom in sorted(chrom_lengths.index):
        chrom_offsets[chrom] = offset
        offset += chrom_lengths[chrom] + 1e6  # добавляем отступ
    
    # Вычисляем глобальные позиции
    genes_info['GLOBAL_POS'] = genes_info.apply(
        lambda row: chrom_offsets.get(row['CHR'], 0) + row['GENE_MID'], axis=1
    )
    
    # Нормализуем позиции в диапазон [0, 1]
    max_pos = genes_info['GLOBAL_POS'].max()
    genes_info['NORM_POS'] = genes_info['GLOBAL_POS'] / max_pos
    
    # Сортируем гены по позиции
    genes_sorted = genes_info.sort_values('NORM_POS')
    
    # Выбираем гены равномерно по позициям
    step = len(genes_sorted) / top_n
    selected_indices = [min(int(i * step), len(genes_sorted) - 1) for i in range(top_n)]
    selected_genes_orig = genes_sorted.iloc[selected_indices]['GENE_NAME_ORIG'].tolist()
    
    # Удаляем дубликаты
    selected_genes_orig = list(dict.fromkeys(selected_genes_orig))
    
    # Если получилось меньше, чем нужно, добираем случайными из оставшихся
    if len(selected_genes_orig) < top_n:
        remaining = [g for g in snp_genes_set if g not in selected_genes_orig]
        if remaining:
            additional = random.sample(remaining, min(top_n - len(selected_genes_orig), len(remaining)))
            selected_genes_orig.extend(additional)
    
    # Подсчет SNP на ген
    snp_per_gene = snp_genes_df.groupby('Gene')['SNP_ID'].nunique()
    
    print(f"\n   📊 Результаты отбора:")
    print(f"      - Отобрано: {len(selected_genes_orig)} генов")
    
    # Статистика по хромосомам
    selected_info = genes_info[genes_info['GENE_NAME_ORIG'].isin(selected_genes_orig)]
    chrom_selected = selected_info['CHR'].value_counts()
    print(f"\n   - Распределение отобранных генов по хромосомам:")
    for chrom, count in chrom_selected.items():
        total = len(genes_info[genes_info['CHR'] == chrom])
        print(f"      {chrom}: {count} генов (из {total}, {count/total*100:.1f}%)")
    
    selected_df = snp_genes_df[snp_genes_df['Gene'].isin(selected_genes_orig)]
    print(f"\n      - SNP в отобранных генах: {selected_df['SNP_ID'].nunique():,}")
    print(f"      - Среднее SNP на отобранный ген: {selected_df.groupby('Gene')['SNP_ID'].nunique().mean():.1f}")
    
    return selected_genes_orig, {'strategy': 'uniform_position', 'selected_count': len(selected_genes_orig)}


# def select_genes_uniform_per_chromosome(snp_genes_df, genes_gff_df, top_n=3000):
#     """
#     Отбор генов с равномерным покрытием внутри каждой хромосомы.
    
#     Параметры:
#     ----------
#     snp_genes_df : DataFrame с колонками SNP_ID и Gene
#     genes_gff_df : DataFrame с информацией о расположении генов (GENE_NAME, CHR, GENE_START, GENE_END)
#     top_n : количество генов для отбора
    
#     Возвращает:
#     -----------
#     selected_genes : list отобранных генов
#     gene_stats : словарь со статистикой
#     """
#     print(f"\n📊 Отбор генов по равномерному покрытию хромосом")
#     print(f"   Целевое количество генов: {top_n}")
    
#     # Функция нормализации имени гена
#     def normalize_gene_name(name):
#         if name.startswith('gene-'):
#             return name[5:]
#         return name
    
#     # Создаем копию и нормализуем имена в GFF
#     genes_info = genes_gff_df[['GENE_NAME', 'CHR', 'GENE_START', 'GENE_END']].drop_duplicates().copy()
#     genes_info['GENE_NAME_NORM'] = genes_info['GENE_NAME'].apply(normalize_gene_name)
    
#     # Получаем список генов из SNP данных
#     snp_genes_set = set(snp_genes_df['Gene'].unique())
#     snp_genes_norm = {normalize_gene_name(g): g for g in snp_genes_set}
    
#     # Фильтруем только гены, которые есть в SNP данных
#     genes_info = genes_info[genes_info['GENE_NAME_NORM'].isin(snp_genes_norm.keys())].copy()
#     genes_info['GENE_NAME_ORIG'] = genes_info['GENE_NAME_NORM'].map(snp_genes_norm)
#     genes_info['GENE_MID'] = (genes_info['GENE_START'] + genes_info['GENE_END']) / 2
    
#     print(f"   - Генов в GFF: {len(genes_gff_df['GENE_NAME'].unique()):,}")
#     print(f"   - Генов в SNP данных: {len(snp_genes_set):,}")
#     print(f"   - Пересечение: {len(genes_info):,} генов")
    
#     if len(genes_info) == 0:
#         print("   ⚠️  Нет генов для отбора! Используем weighted стратегию")
#         return select_genes_weighted(snp_genes_df, top_n=top_n, strategy='weighted')
    
#     # Подсчет генов по хромосомам
#     chrom_counts = genes_info['CHR'].value_counts()
#     total_genes = len(genes_info)
    
#     print(f"\n   - Распределение генов по хромосомам:")
#     for chrom, count in chrom_counts.items():
#         print(f"      {chrom}: {count:,} генов ({count/total_genes*100:.1f}%)")
    
#     # Вычисляем целевое количество генов на хромосому (пропорционально длине хромосомы)
#     # Сначала вычисляем длину каждой хромосомы
#     chrom_lengths = genes_info.groupby('CHR')['GENE_END'].max() - genes_info.groupby('CHR')['GENE_START'].min()
#     total_length = chrom_lengths.sum()
    
#     target_per_chrom = {}
#     for chrom in chrom_counts.index:
#         # Пропорционально длине хромосомы
#         chrom_len = chrom_lengths[chrom]
#         target = max(1, int(top_n * chrom_len / total_length))
#         target_per_chrom[chrom] = target
    
#     # Корректируем, чтобы сумма была равна top_n
#     diff = top_n - sum(target_per_chrom.values())
#     if diff > 0:
#         # Добавляем недостающие гены к хромосомам с наибольшей длиной
#         sorted_chroms = sorted(chrom_lengths.items(), key=lambda x: x[1], reverse=True)
#         for i in range(diff):
#             target_per_chrom[sorted_chroms[i % len(sorted_chroms)][0]] += 1
    
#     print(f"\n   - Целевое распределение по хромосомам (по длине):")
#     for chrom, target in target_per_chrom.items():
#         chrom_len = chrom_lengths[chrom] / 1e6
#         print(f"      {chrom}: {target} генов (длина {chrom_len:.1f} Mbp)")
    
#     # Для каждой хромосомы отбираем гены равномерно по позиции
#     selected_genes = []
    
#     for chrom, target in target_per_chrom.items():
#         chrom_genes = genes_info[genes_info['CHR'] == chrom].copy()
#         if len(chrom_genes) == 0:
#             continue
        
#         # Нормализуем позиции внутри хромосомы
#         min_pos = chrom_genes['GENE_MID'].min()
#         max_pos = chrom_genes['GENE_MID'].max()
#         if max_pos > min_pos:
#             chrom_genes['NORM_POS'] = (chrom_genes['GENE_MID'] - min_pos) / (max_pos - min_pos)
#         else:
#             chrom_genes['NORM_POS'] = 0.5
        
#         # Сортируем по позиции
#         chrom_genes_sorted = chrom_genes.sort_values('NORM_POS')
        
#         # Выбираем гены равномерно по позиции
#         step = len(chrom_genes_sorted) / target
#         indices = [min(int(i * step), len(chrom_genes_sorted) - 1) for i in range(target)]
#         selected = chrom_genes_sorted.iloc[indices]['GENE_NAME_ORIG'].tolist()
#         selected_genes.extend(selected)
        
#         print(f"      {chrom}: взято {len(selected)} из {len(chrom_genes)} генов "
#               f"(шаг {step:.1f} позиций)")
    
#     # Удаляем дубликаты и добираем при необходимости
#     selected_genes = list(dict.fromkeys(selected_genes))
    
#     if len(selected_genes) < top_n:
#         remaining = [g for g in snp_genes_set if g not in selected_genes]
#         if remaining:
#             additional = random.sample(remaining, min(top_n - len(selected_genes), len(remaining)))
#             selected_genes.extend(additional)
    
#     # Подсчет SNP на ген
#     snp_per_gene = snp_genes_df.groupby('Gene')['SNP_ID'].nunique()
    
#     print(f"\n   📊 Результаты отбора:")
#     print(f"      - Отобрано: {len(selected_genes)} генов")
    
#     # Статистика по хромосомам
#     selected_info = genes_info[genes_info['GENE_NAME_ORIG'].isin(selected_genes)]
#     chrom_selected = selected_info['CHR'].value_counts()
#     print(f"\n   - Распределение отобранных генов по хромосомам:")
#     for chrom, count in chrom_selected.items():
#         total = len(genes_info[genes_info['CHR'] == chrom])
#         print(f"      {chrom}: {count} генов (из {total}, {count/total*100:.1f}%)")
    
#     selected_df = snp_genes_df[snp_genes_df['Gene'].isin(selected_genes)]
#     print(f"\n      - SNP в отобранных генах: {selected_df['SNP_ID'].nunique():,}")
#     print(f"      - Среднее SNP на отобранный ген: {selected_df.groupby('Gene')['SNP_ID'].nunique().mean():.1f}")
    
#     return selected_genes, {'strategy': 'uniform_per_chromosome', 'selected_count': len(selected_genes)}
def select_genes_uniform_per_chromosome(snp_genes_df, genes_gff_df, top_n=3000, min_snps_per_gene=15):
    """
    Отбор генов с равномерным покрытием внутри каждой хромосомы.
    Добавлена фильтрация по минимальному количеству SNP на ген.
    
    Параметры:
    ----------
    snp_genes_df : DataFrame с колонками SNP_ID и Gene
    genes_gff_df : DataFrame с информацией о расположении генов
    top_n : количество генов для отбора
    min_snps_per_gene : минимальное количество SNP на ген (фильтр снизу)
    """
    print(f"\n📊 Отбор генов по равномерному покрытию хромосом")
    print(f"   Целевое количество генов: {top_n}")
    print(f"   Минимальное количество SNP на ген: {min_snps_per_gene}")
    
    # Функция нормализации имени гена
    def normalize_gene_name(name):
        if name.startswith('gene-'):
            return name[5:]
        return name
    
    # Подсчет SNP на ген
    snp_per_gene = snp_genes_df.groupby('Gene')['SNP_ID'].nunique()
    
    # ⭐ НОВОЕ: Фильтрация генов по минимальному количеству SNP
    genes_above_threshold = snp_per_gene[snp_per_gene >= min_snps_per_gene].index.tolist()
    print(f"\n   - Генов с ≥ {min_snps_per_gene} SNP: {len(genes_above_threshold)} из {len(snp_per_gene)}")
    
    if len(genes_above_threshold) < top_n:
        print(f"   ⚠️  Недостаточно генов с ≥ {min_snps_per_gene} SNP")
        print(f"   Будет использовано {len(genes_above_threshold)} генов")
        top_n = len(genes_above_threshold)
    
    # Создаем копию и нормализуем имена в GFF
    genes_info = genes_gff_df[['GENE_NAME', 'CHR', 'GENE_START', 'GENE_END']].drop_duplicates().copy()
    genes_info['GENE_NAME_NORM'] = genes_info['GENE_NAME'].apply(normalize_gene_name)
    
    # Получаем список генов из SNP данных
    snp_genes_set = set(genes_above_threshold)  # Используем отфильтрованные гены
    
    # Создаем нормализованную версию для сопоставления
    snp_genes_norm = {normalize_gene_name(g): g for g in snp_genes_set}
    
    # Фильтруем только гены, которые есть в SNP данных и имеют достаточно SNP
    genes_info = genes_info[genes_info['GENE_NAME_NORM'].isin(snp_genes_norm.keys())].copy()
    genes_info['GENE_NAME_ORIG'] = genes_info['GENE_NAME_NORM'].map(snp_genes_norm)
    genes_info['GENE_MID'] = (genes_info['GENE_START'] + genes_info['GENE_END']) / 2
    
    print(f"   - Пересечение с GFF: {len(genes_info)} генов")
    
    if len(genes_info) == 0:
        print("   ⚠️  Нет генов для отбора! Используем weighted стратегию")
        return select_genes_weighted(snp_genes_df, top_n=top_n, strategy='weighted')
    
    # Подсчет генов по хромосомам
    chrom_counts = genes_info['CHR'].value_counts()
    total_genes = len(genes_info)
    
    print(f"\n   - Распределение генов по хромосомам (после фильтрации):")
    for chrom, count in chrom_counts.head(10).items():
        print(f"      {chrom}: {count:,} генов ({count/total_genes*100:.1f}%)")
    if len(chrom_counts) > 10:
        print(f"      ... и {len(chrom_counts)-10} других хромосом")
    
    # Вычисляем длину каждой хромосомы
    chrom_lengths = genes_info.groupby('CHR')['GENE_END'].max() - genes_info.groupby('CHR')['GENE_START'].min()
    total_length = chrom_lengths.sum()
    
    # Вычисляем целевое количество генов на хромосому
    target_per_chrom = {}
    for chrom in chrom_counts.index:
        chrom_len = chrom_lengths[chrom]
        target = max(1, int(top_n * chrom_len / total_length))
        target_per_chrom[chrom] = target
    
    # Корректируем, чтобы сумма была равна top_n
    diff = top_n - sum(target_per_chrom.values())
    if diff > 0:
        sorted_chroms = sorted(chrom_lengths.items(), key=lambda x: x[1], reverse=True)
        for i in range(diff):
            target_per_chrom[sorted_chroms[i % len(sorted_chroms)][0]] += 1
    
    print(f"\n   - Целевое распределение по хромосомам (по длине):")
    for chrom, target in list(target_per_chrom.items())[:10]:
        chrom_len = chrom_lengths[chrom] / 1e6
        print(f"      {chrom}: {target} генов (длина {chrom_len:.1f} Mbp)")
    
    # Для каждой хромосомы отбираем гены равномерно по позиции
    selected_genes = []
    
    for chrom, target in target_per_chrom.items():
        chrom_genes = genes_info[genes_info['CHR'] == chrom].copy()
        if len(chrom_genes) == 0:
            continue
        
        # Нормализуем позиции внутри хромосомы
        min_pos = chrom_genes['GENE_MID'].min()
        max_pos = chrom_genes['GENE_MID'].max()
        if max_pos > min_pos:
            chrom_genes['NORM_POS'] = (chrom_genes['GENE_MID'] - min_pos) / (max_pos - min_pos)
        else:
            chrom_genes['NORM_POS'] = 0.5
        
        # Сортируем по позиции
        chrom_genes_sorted = chrom_genes.sort_values('NORM_POS')
        
        # Выбираем гены равномерно по позиции
        step = len(chrom_genes_sorted) / target
        indices = [min(int(i * step), len(chrom_genes_sorted) - 1) for i in range(target)]
        selected = chrom_genes_sorted.iloc[indices]['GENE_NAME_ORIG'].tolist()
        selected_genes.extend(selected)
        
        print(f"      {chrom}: взято {len(selected)} из {len(chrom_genes)} генов "
              f"(шаг {step:.1f} позиций)")
    
    # Удаляем дубликаты
    selected_genes = list(dict.fromkeys(selected_genes))
    
    # Если получилось меньше, чем нужно, добираем случайными из оставшихся
    if len(selected_genes) < top_n:
        remaining = [g for g in genes_above_threshold if g not in selected_genes]
        if remaining:
            additional = random.sample(remaining, min(top_n - len(selected_genes), len(remaining)))
            selected_genes.extend(additional)
    
    # Статистика по отобранным генам
    snp_per_gene_selected = snp_per_gene[snp_per_gene.index.isin(selected_genes)]
    
    print(f"\n   📊 Результаты отбора:")
    print(f"      - Отобрано: {len(selected_genes)} генов")
    print(f"      - Среднее SNP на отобранный ген: {snp_per_gene_selected.mean():.1f}")
    print(f"      - Мин SNP на отобранный ген: {snp_per_gene_selected.min()}")
    print(f"      - Макс SNP на отобранный ген: {snp_per_gene_selected.max()}")
    
    selected_df = snp_genes_df[snp_genes_df['Gene'].isin(selected_genes)]
    print(f"      - SNP в отобранных генах: {selected_df['SNP_ID'].nunique():,}")
    
    # Распределение по хромосомам
    selected_info = genes_info[genes_info['GENE_NAME_ORIG'].isin(selected_genes)]
    chrom_selected = selected_info['CHR'].value_counts()
    print(f"\n   - Распределение отобранных генов по хромосомам:")
    for chrom, count in chrom_selected.head(10).items():
        total = len(genes_info[genes_info['CHR'] == chrom])
        print(f"      {chrom}: {count} генов (из {total}, {count/total*100:.1f}%)")
    
    return selected_genes, {'strategy': 'uniform_per_chromosome', 
                            'selected_count': len(selected_genes),
                            'min_snps_per_gene': min_snps_per_gene,
                            'mean_snps_per_gene': snp_per_gene_selected.mean(),
                            'target_per_chrom': target_per_chrom}


def filter_snp_genes_by_genes(snp_genes_df, selected_genes):
    """Фильтрация snp_genes_df по отобранным генам"""
    return snp_genes_df[snp_genes_df['Gene'].isin(selected_genes)]


# ============================================
# ОПТИМИЗИРОВАННАЯ АУГМЕНТАЦИЯ
# ============================================

def mixup_data_np(X, y, aug_count=2.0, lam=0.3):
    """
    Аугментация с использованием numpy
    """
    n_samples = X.shape[0]
    n_aug = int(aug_count * n_samples)
    
    idx1 = np.random.choice(n_samples, n_aug, replace=True)
    idx2 = np.random.choice(n_samples, n_aug, replace=True)
    
    lambdas = np.random.beta(lam, lam, n_aug).reshape(-1, 1)
    
    X_aug = (1 - lambdas) * X[idx1] + lambdas * X[idx2]
    y_aug = (1 - lambdas) * y[idx1] + lambdas * y[idx2]
    
    X_combined = np.vstack([X, X_aug])
    y_combined = np.vstack([y, y_aug])
    
    return X_combined, y_combined


# ============================================
# ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ ЗАГРУЗКИ SNP
# ============================================

def load_snp_data_optimized(bed_file, bim_file, fam_file, snp_list, sample_indices, 
                            chunk_size=500, use_float16=False):
    """ОПТИМИЗИРОВАННАЯ загрузка SNP данных чанками"""
    
    import struct
    
    print(f"\n📊 Загрузка SNP данных чанками (chunk_size={chunk_size})")
    
    bim = pd.read_csv(bim_file, sep='\t', header=None,
                      names=['CHR', 'SNP_ID', 'CM', 'POS', 'A1', 'A2'])
    
    snp_set = set(snp_list)
    
    snp_indices_to_load = []
    snp_names_loaded = []
    
    print("   - Поиск SNP в BIM файле...")
    for i, snp in enumerate(bim['SNP_ID']):
        if snp in snp_set:
            snp_indices_to_load.append(i)
            snp_names_loaded.append(snp)
    
    print(f"   - Найдено {len(snp_indices_to_load)} SNP из {len(snp_list)} запрошенных")
    
    num_samples = len(fam)
    num_selected_samples = len(sample_indices)
    dtype = np.float16 if use_float16 else np.float32
    
    sample_byte_indices = [idx // 4 for idx in sample_indices]
    sample_bit_offsets = [(idx % 4) * 2 for idx in sample_indices]
    
    X_chunks = []
    chunk_snp_names = []
    
    bytes_per_snp = (num_samples + 3) // 4
    snp_positions = [3 + snp_idx * bytes_per_snp for snp_idx in snp_indices_to_load]
    
    print(f"   - Загрузка {len(snp_indices_to_load)} SNP чанками по {chunk_size}...")
    
    with open(bed_file, 'rb') as f:
        for chunk_start in tqdm(range(0, len(snp_indices_to_load), chunk_size), 
                                desc="   - Загрузка", unit="chunk"):
            chunk_end = min(chunk_start + chunk_size, len(snp_indices_to_load))
            chunk_snp_positions = snp_positions[chunk_start:chunk_end]
            chunk_snp_names_local = snp_names_loaded[chunk_start:chunk_end]
            
            chunk_data = np.zeros((num_selected_samples, len(chunk_snp_positions)), dtype=dtype)
            
            for j, snp_pos in enumerate(chunk_snp_positions):
                f.seek(snp_pos)
                snp_bytes = f.read(bytes_per_snp)
                
                for i, (byte_idx, bit_offset) in enumerate(zip(sample_byte_indices, sample_bit_offsets)):
                    if byte_idx < len(snp_bytes):
                        byte = snp_bytes[byte_idx]
                        genotype = (byte >> bit_offset) & 3
                        if genotype == 0:
                            chunk_data[i, j] = 0.0
                        elif genotype == 1:
                            chunk_data[i, j] = 1.0
                        elif genotype == 2:
                            chunk_data[i, j] = 2.0
                        else:
                            chunk_data[i, j] = -1.0
                    else:
                        chunk_data[i, j] = -1.0
            
            X_chunks.append(chunk_data)
            chunk_snp_names.append(chunk_snp_names_local)
            
            gc.collect()
    
    return X_chunks, chunk_snp_names


def freq_encode_snp_chunk_optimized(chunk_data):
    """ОПТИМИЗИРОВАННОЕ частотное кодирование для чанка данных"""
    encoded = chunk_data.copy()
    n_samples = chunk_data.shape[0]
    
    for j in range(chunk_data.shape[1]):
        col = chunk_data[:, j]
        valid_mask = col != -1
        valid_values = col[valid_mask]
        
        if len(valid_values) > 0:
            unique, counts = np.unique(valid_values, return_counts=True)
            freq_dict = dict(zip(unique, counts / len(valid_values)))
            
            for val in [0, 1, 2]:
                if val not in freq_dict:
                    freq_dict[val] = 0.0
            
            for i in range(len(col)):
                if valid_mask[i]:
                    encoded[i, j] = freq_dict.get(col[i], 0.0)
                else:
                    encoded[i, j] = 0.0
        else:
            encoded[:, j] = 0.0
    
    return encoded.astype(np.float32)


# ============================================
# КЛАССЫ МОДЕЛИ
# ============================================

class MarkerToGeneLayer(nn.Module):
    def __init__(self, snp_genes_df, gene_list):
        super(MarkerToGeneLayer, self).__init__()
        self.gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}
        self.snp_to_gene_map = {}
        for _, row in snp_genes_df.iterrows():
            if row['Gene'] in self.gene_to_index:
                self.snp_to_gene_map[row['SNP_ID']] = row['Gene']

    def forward(self, snps):
        gene_indices = [self.gene_to_index.get(self.snp_to_gene_map.get(snp, ""), -1) for snp in snps]
        
        gene_to_snp_indices = {}
        for snp_idx, gene_idx in enumerate(gene_indices):
            if gene_idx != -1:
                if gene_idx not in gene_to_snp_indices:
                    gene_to_snp_indices[gene_idx] = []
                gene_to_snp_indices[gene_idx].append(snp_idx)
        
        return gene_indices, gene_to_snp_indices


# class GeneticEncodingBlock(nn.Module):
#     def __init__(self, num_genes, embed_dim, snp_genes_df, gene_list, max_snp_per_gene=20):
#         super(GeneticEncodingBlock, self).__init__()
#         self.marker_to_gene = MarkerToGeneLayer(snp_genes_df, gene_list)
#         self.gene_embedding = nn.Embedding(num_genes, embed_dim)
#         self.embed_dim = embed_dim
#         self.snp_weights = nn.Parameter(torch.ones(1, 1, embed_dim))
        
#         self.num_genes_total = num_genes
#         self.max_snp_per_gene = max_snp_per_gene
        
#         # Batch norm
#         self.bn = nn.BatchNorm1d(embed_dim)
        
#         # Инициализация весов
#         nn.init.xavier_uniform_(self.snp_weights)

#     def forward(self, X, snp_names, gene_positions):
#         batch_size, num_snps = X.shape
#         device = next(self.parameters()).device
        
#         X = X.to(device)
#         _, gene_to_snp_indices = self.marker_to_gene(snp_names)
        
#         if not gene_to_snp_indices:
#             return torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)
        
#         num_genes = len(gene_to_snp_indices)
        
#         # Получаем позиции генов
#         gene_pos_values = torch.tensor(
#             [gene_positions.get(gene, 0) for gene in self.marker_to_gene.gene_to_index.keys()],
#             dtype=torch.float32, device=device
#         ).unsqueeze(-1)  # [num_genes_total, 1]
        
#         # Вычисляем средние значения SNP по генам для позиционного кодирования
#         gene_avg_values = torch.zeros(batch_size, self.num_genes_total, device=device)
#         for gene_idx, snp_indices in gene_to_snp_indices.items():
#             snp_indices_limited = snp_indices[:self.max_snp_per_gene]
#             if snp_indices_limited:
#                 snp_values = X[:, snp_indices_limited]
#                 avg_per_sample = snp_values.mean(dim=1)
#                 gene_avg_values[:, gene_idx] = avg_per_sample
        
#         # Позиционное кодирование
#         div_term = torch.exp(
#             torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device) * 
#             (-log(10000.0) / self.embed_dim)
#         )
        
#         position_tensor = gene_pos_values * div_term.unsqueeze(0) * torch.exp(gene_avg_values.unsqueeze(-1))
        
#         pe = torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)
#         pe[:, :, 0::2] = torch.sin(position_tensor)
#         pe[:, :, 1::2] = torch.cos(position_tensor)
        
#         # ============================================================
#         # НОВАЯ АГРЕГАЦИЯ SNP С ИСПОЛЬЗОВАНИЕМ conv1d
#         # (взято из старого кода)
#         # ============================================================
        
#         # Взвешивание SNP
#         snp_weights_expanded = self.snp_weights.expand(batch_size, num_snps, self.embed_dim)
#         weighted_snp_embeds = X.unsqueeze(-1).repeat(1, 1, self.embed_dim) * snp_weights_expanded
        
#         # Находим максимальное количество SNP на ген
#         max_snp_count = max(len(indices) for indices in gene_to_snp_indices.values())
#         max_snp_count = min(max_snp_count, self.max_snp_per_gene)
        
#         # Создаем плоские массивы индексов для векторизованной обработки
#         gene_indices_flat = []
#         snp_indices_flat = []
#         snp_offsets_flat = []
        
#         for gene_idx, snp_indices in gene_to_snp_indices.items():
#             snp_indices_limited = snp_indices[:self.max_snp_per_gene]
#             for offset, snp_idx in enumerate(snp_indices_limited):
#                 gene_indices_flat.append(gene_idx)
#                 snp_indices_flat.append(snp_idx)
#                 snp_offsets_flat.append(offset)
        
#         if not gene_indices_flat:
#             # Если нет SNP, возвращаем только эмбеддинги генов и PE
#             all_gene_indices = torch.arange(self.num_genes_total, dtype=torch.long, device=device)
#             gene_embeds_output = self.gene_embedding(all_gene_indices).unsqueeze(0) + pe
#             gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
#             gene_embeds_output = self.bn(gene_embeds_output)
#             gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
#             return gene_embeds_output
        
#         # Конвертируем в тензоры
#         gene_indices_flat = torch.tensor(gene_indices_flat, device=device, dtype=torch.long)
#         snp_indices_flat = torch.tensor(snp_indices_flat, device=device, dtype=torch.long)
#         snp_offsets_flat = torch.tensor(snp_offsets_flat, device=device, dtype=torch.long)
        
#         # Создаем паддированные тензоры
#         padded_snp_embeds = torch.zeros(batch_size, self.num_genes_total, max_snp_count, self.embed_dim, device=device)
        
#         # Заполняем данные
#         padded_snp_embeds[:, gene_indices_flat, snp_offsets_flat, :] = weighted_snp_embeds[:, snp_indices_flat, :]
        
#         # Создаем маску для валидных SNP
#         mask = torch.zeros(batch_size, self.num_genes_total, max_snp_count, device=device)
#         mask[:, gene_indices_flat, snp_offsets_flat] = 1
        
#         # Применяем conv1d для агрегации (как в старом коде)
#         # Переставляем размерности: [batch, genes, snps, embed] -> [batch, embed, snps, genes]
#         padded_snp_embeds = padded_snp_embeds.permute(0, 3, 2, 1)  # [batch, embed, snps, genes]
        
#         # Conv1d для агрегации по SNP
#         gene_embeds_conv = F.conv1d(
#             padded_snp_embeds.reshape(-1, self.embed_dim, max_snp_count),
#             weight=torch.ones(self.embed_dim, 1, max_snp_count, device=device),
#             groups=self.embed_dim
#         ).view(batch_size, self.embed_dim, self.num_genes_total).permute(0, 2, 1)  # [batch, genes, embed]
        
#         # Применяем маску (обнуляем гены без SNP)
#         mask_gene = (mask.sum(dim=2) > 0).float().unsqueeze(-1)  # [batch, genes, 1]
#         gene_embeds_conv = gene_embeds_conv * mask_gene
        
#         # Добавляем эмбеддинги генов и позиционное кодирование
#         all_gene_indices = torch.arange(self.num_genes_total, dtype=torch.long, device=device)
#         gene_embeds_output = gene_embeds_conv + self.gene_embedding(all_gene_indices).unsqueeze(0) + pe
        
#         # Batch norm
#         gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
#         gene_embeds_output = self.bn(gene_embeds_output)
#         gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
        
#         return gene_embeds_output


class GeneticEncodingBlock(nn.Module):
    def __init__(self, num_genes, embed_dim, snp_genes_df, gene_list, max_snp_per_gene=20, num_attention_heads=4):
        super(GeneticEncodingBlock, self).__init__()
        self.marker_to_gene = MarkerToGeneLayer(snp_genes_df, gene_list)
        self.gene_embedding = nn.Embedding(num_genes, embed_dim)
        self.embed_dim = embed_dim
        self.num_genes_total = num_genes
        self.max_snp_per_gene = max_snp_per_gene
        
        # Проекция SNP значений
        self.snp_projection = nn.Linear(1, embed_dim)
        
        # Механизм внимания
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            dropout=0.2,  # увеличить с 0.1
            batch_first=True
        )
        
        # ⭐ ВАЖНО: gene_query должен быть параметром модели
        self.gene_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        
        # Инициализация
        nn.init.xavier_uniform_(self.snp_projection.weight)
        nn.init.xavier_uniform_(self.gene_query)

    def forward(self, X, snp_names, gene_positions):
        batch_size, num_snps = X.shape
        device = next(self.parameters()).device
        
        X = X.to(device)
        _, gene_to_snp_indices = self.marker_to_gene(snp_names)
        
        if not gene_to_snp_indices:
            return torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)
        
        # Позиционное кодирование генов
        gene_pos_values = torch.tensor(
            [gene_positions.get(gene, 0) for gene in self.marker_to_gene.gene_to_index.keys()],
            dtype=torch.float32, device=device
        ).unsqueeze(-1)
        
        # Вычисляем средние значения SNP по генам
        gene_avg_values = torch.zeros(batch_size, self.num_genes_total, device=device)
        for gene_idx, snp_indices in gene_to_snp_indices.items():
            snp_indices_limited = snp_indices[:self.max_snp_per_gene]
            if snp_indices_limited:
                snp_values = X[:, snp_indices_limited]
                avg_per_sample = snp_values.mean(dim=1)
                gene_avg_values[:, gene_idx] = avg_per_sample
        
        # Позиционное кодирование
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device) * 
            (-log(10000.0) / self.embed_dim)
        )
        
        position_tensor = gene_pos_values * div_term.unsqueeze(0) * torch.exp(gene_avg_values.unsqueeze(-1))
        
        pe = torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)
        pe[:, :, 0::2] = torch.sin(position_tensor)
        pe[:, :, 1::2] = torch.cos(position_tensor)
        
        # ============================================================
        # ВЕКТОРИЗОВАННАЯ АГРЕГАЦИЯ С ВНИМАНИЕМ
        # ============================================================
        
        # Создаем структуру для хранения SNP по генам с паддингом
        max_snp_count = max(len(indices) for indices in gene_to_snp_indices.values())
        max_snp_count = min(max_snp_count, self.max_snp_per_gene)
        
        gene_indices_flat = []
        snp_indices_flat = []
        snp_offsets_flat = []
        
        for gene_idx, snp_indices in gene_to_snp_indices.items():
            snp_indices_limited = snp_indices[:self.max_snp_per_gene]
            for offset, snp_idx in enumerate(snp_indices_limited):
                gene_indices_flat.append(gene_idx)
                snp_indices_flat.append(snp_idx)
                snp_offsets_flat.append(offset)
        
        if not gene_indices_flat:
            all_gene_indices = torch.arange(self.num_genes_total, dtype=torch.long, device=device)
            gene_embeds_output = self.gene_embedding(all_gene_indices).unsqueeze(0) + pe
            gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
            gene_embeds_output = self.bn(gene_embeds_output)
            gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
            return gene_embeds_output
        
        gene_indices_flat = torch.tensor(gene_indices_flat, device=device, dtype=torch.long)
        snp_indices_flat = torch.tensor(snp_indices_flat, device=device, dtype=torch.long)
        snp_offsets_flat = torch.tensor(snp_offsets_flat, device=device, dtype=torch.long)
        
        # Создаем паддированный тензор SNP [batch, genes, max_snps]
        padded_snps = torch.zeros(batch_size, self.num_genes_total, max_snp_count, device=device)
        padded_snps[:, gene_indices_flat, snp_offsets_flat] = X[:, snp_indices_flat]
        
        # Создаем маску для валидных SNP
        mask = torch.zeros(batch_size, self.num_genes_total, max_snp_count, device=device)
        mask[:, gene_indices_flat, snp_offsets_flat] = 1
        
        # Проецируем SNP в embedding space [batch, genes, snps, embed_dim]
        snp_embeds = self.snp_projection(padded_snps.unsqueeze(-1))
        
        # Добавляем позиционное кодирование внутри гена
        pos_encoding = torch.zeros(1, 1, max_snp_count, self.embed_dim, device=device)
        for pos in range(max_snp_count):
            pos_encoding[:, :, pos, 0::2] = torch.sin(torch.tensor(pos / 10000.0, device=device))
            pos_encoding[:, :, pos, 1::2] = torch.cos(torch.tensor(pos / 10000.0, device=device))
        
        snp_embeds = snp_embeds + pos_encoding
        
        # Применяем attention для каждого гена
        # Reshape для attention: [batch * genes, snps, embed_dim]
        batch_genome_size = batch_size * self.num_genes_total
        snp_embeds_flat = snp_embeds.view(batch_genome_size, max_snp_count, self.embed_dim)
        
        # ⭐ Используем self.gene_query (параметр модели)
        gene_query = self.gene_query.expand(batch_genome_size, 1, self.embed_dim)
        
        # Attention
        attended, attention_weights = self.attention(
            query=gene_query,
            key=snp_embeds_flat,
            value=snp_embeds_flat,
            key_padding_mask=(mask.view(batch_genome_size, max_snp_count) == 0)
        )
        # После attention, посмотрите распределение весов
        print(f"Attention weights - min: {attention_weights.min():.4f}, max: {attention_weights.max():.4f}, mean: {attention_weights.mean():.4f}")
        print(f"Entropy: {-(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean():.4f}")
        
        print("\n--- Attention weights for first 5 genes (first batch) ---")
        for gene_idx in range(min(5, self.num_genes_total)):
            # Получаем веса для этого гена (первый батч)
            gene_weights = attention_weights[gene_idx, 0, :]  # [max_snp_count]
            # Считаем, сколько реальных SNP у этого гена
            real_snp_count = mask[0, gene_idx, :].sum().item()
            if real_snp_count > 0:
                print(f"Gene {gene_idx}: real SNPs = {real_snp_count}")
                print(f"  Weights for real SNPs: {gene_weights[:int(real_snp_count)].detach().cpu().numpy()}")
                print(f"  Sum of real weights: {gene_weights[:int(real_snp_count)].sum().item():.4f} (should be ~1.0)")

        # Reshape обратно: [batch, genes, embed_dim]
        gene_embeds_attention = attended.squeeze(1).view(batch_size, self.num_genes_total, self.embed_dim)
        gene_embeds_attention = self.layer_norm(gene_embeds_attention)
        
        # Обнуляем гены без SNP
        mask_gene = (mask.sum(dim=2) > 0).float().unsqueeze(-1)
        gene_embeds_attention = gene_embeds_attention * mask_gene
        
        # Добавляем эмбеддинги генов и позиционное кодирование
        all_gene_indices = torch.arange(self.num_genes_total, dtype=torch.long, device=device)
        gene_embeds_output = gene_embeds_attention + self.gene_embedding(all_gene_indices).unsqueeze(0) + pe
        
        # Batch norm
        gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
        gene_embeds_output = self.bn(gene_embeds_output)
        gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
        
        return gene_embeds_output
    

class GeneticEncodingBlockConv(nn.Module):
    """Агрегация SNP через простое усреднение (без внимания)"""
    def __init__(self, num_genes, embed_dim, snp_genes_df, gene_list, max_snp_per_gene=1000):
        super(GeneticEncodingBlockConv, self).__init__()
        self.marker_to_gene = MarkerToGeneLayer(snp_genes_df, gene_list)
        self.gene_embedding = nn.Embedding(num_genes, embed_dim)
        self.embed_dim = embed_dim
        self.num_genes_total = num_genes
        self.max_snp_per_gene = max_snp_per_gene
        
        # Batch norm
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, X, snp_names, gene_positions):
        batch_size, num_snps = X.shape
        device = next(self.parameters()).device
        
        X = X.to(device)
        _, gene_to_snp_indices = self.marker_to_gene(snp_names)
        
        if not gene_to_snp_indices:
            return torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)
        
        # Получаем позиции генов
        gene_pos_values = torch.tensor(
            [gene_positions.get(gene, 0) for gene in self.marker_to_gene.gene_to_index.keys()],
            dtype=torch.float32, device=device
        ).unsqueeze(-1)
        
        # Вычисляем средние значения SNP по генам для позиционного кодирования
        gene_avg_values = torch.zeros(batch_size, self.num_genes_total, device=device)
        for gene_idx, snp_indices in gene_to_snp_indices.items():
            snp_indices_limited = snp_indices[:self.max_snp_per_gene]
            if snp_indices_limited:
                snp_values = X[:, snp_indices_limited]
                avg_per_sample = snp_values.mean(dim=1)
                gene_avg_values[:, gene_idx] = avg_per_sample
        
        # Позиционное кодирование
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device) * 
            (-log(10000.0) / self.embed_dim)
        )
        
        position_tensor = gene_pos_values * div_term.unsqueeze(0) * torch.exp(gene_avg_values.unsqueeze(-1))
        
        pe = torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)
        pe[:, :, 0::2] = torch.sin(position_tensor)
        pe[:, :, 1::2] = torch.cos(position_tensor)
        
        # ============================================================
        # ПРОСТОЕ УСРЕДНЕНИЕ SNP (БЕЗ ВНИМАНИЯ)
        # ============================================================
        
        # Создаем тензор для усредненных значений SNP по генам
        gene_embeds_simple = torch.zeros(batch_size, self.num_genes_total, self.embed_dim, device=device)
        
        # Простое усреднение значений SNP
        for gene_idx, snp_indices in gene_to_snp_indices.items():
            snp_indices_limited = snp_indices[:self.max_snp_per_gene]
            if snp_indices_limited:
                snp_values = X[:, snp_indices_limited]  # [batch, n_snps]
                # Усредняем значения SNP
                avg_snp_values = snp_values.mean(dim=1, keepdim=True)  # [batch, 1]
                # Проецируем в embedding space через линейный слой
                # Используем обучаемую проекцию
                projected = avg_snp_values.unsqueeze(-1).expand(-1, -1, self.embed_dim)
                gene_embeds_simple[:, gene_idx, :] = projected.squeeze(1)
        
        # Добавляем эмбеддинги генов и позиционное кодирование
        all_gene_indices = torch.arange(self.num_genes_total, dtype=torch.long, device=device)
        gene_embeds_output = gene_embeds_simple + self.gene_embedding(all_gene_indices).unsqueeze(0) + pe
        
        # Batch norm
        gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
        gene_embeds_output = self.bn(gene_embeds_output)
        gene_embeds_output = gene_embeds_output.permute(0, 2, 1)
        
        return gene_embeds_output


class PhenotypePredictionModelConv(nn.Module):
    """Модель с простым усреднением SNP (без внимания)"""
    def __init__(self, num_snps, num_genes_total, embed_dim, num_heads, ff_dim,
                 num_layers, num_classes, snp_genes_df, gene_list, max_snp_per_gene=1000, 
                 dropout_rate=0.2):
        super(PhenotypePredictionModelConv, self).__init__()
        self.genetic_encoding = GeneticEncodingBlockConv(
            num_genes_total, embed_dim, snp_genes_df, gene_list, max_snp_per_gene
        )
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, num_layers, dropout_rate)
        self.prediction = PredictionBlock(embed_dim, num_classes, num_genes_total, dropout_rate)
        
        # Голова реконструкции
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim * num_genes_total, 512),
            nn.ReLU(),
            nn.Linear(512, num_snps)
        )

    def forward(self, X, snp_names, gene_positions):
        X_encoded = self.genetic_encoding(X, snp_names, gene_positions)
        X_transformed = self.transformer(X_encoded)
        outputs = self.prediction(X_transformed)
        
        # Реконструкция исходных SNP
        flat = X_transformed.reshape(X_transformed.size(0), -1)
        recon = self.reconstruction_head(flat)
        
        return outputs, recon


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout_rate=0.2):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim,
            activation='gelu', 
            batch_first=True,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        if x.size(0) > 0 and x.size(1) > 0:
            x = self.transformer(x)
        return x


class PredictionBlock(nn.Module):
    def __init__(self, embed_dim, num_classes, num_genes, dropout_rate=0.3):
        super(PredictionBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim * num_genes, 32)  # уменьшаем с 64 до 32
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(32)
        
        # Инициализация
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        if x.size(1) > 0:
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)
        return x


# class PhenotypePredictionModel(nn.Module):
#     def __init__(self, num_snps, num_genes_total, embed_dim, num_heads, ff_dim,
#                  num_layers, num_classes, snp_genes_df, gene_list, max_snp_per_gene=20, dropout_rate=0.2):
#         super(PhenotypePredictionModel, self).__init__()
#         self.genetic_encoding = GeneticEncodingBlock(num_genes_total, embed_dim, snp_genes_df, gene_list, max_snp_per_gene)
#         self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, num_layers, dropout_rate)
#         self.prediction = PredictionBlock(embed_dim, num_classes, num_genes_total, dropout_rate)

def reconstruction_loss(reconstructed, original):
    """
    Функция потерь для реконструкции SNP данных
    """
    return F.mse_loss(reconstructed, original)


class PhenotypePredictionModel(nn.Module):
    def __init__(self, num_snps, num_genes_total, embed_dim, num_heads, ff_dim,
                 num_layers, num_classes, snp_genes_df, gene_list, max_snp_per_gene=20, 
                 dropout_rate=0.2, num_attention_heads=2):
        super(PhenotypePredictionModel, self).__init__()
        self.genetic_encoding = GeneticEncodingBlock(
            num_genes_total, embed_dim, snp_genes_df, gene_list, 
            max_snp_per_gene, num_attention_heads
        )
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, num_layers, dropout_rate)
        self.prediction = PredictionBlock(embed_dim, num_classes, num_genes_total, dropout_rate)
        
        # Добавляем голову реконструкции (как в первой модели)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim * num_genes_total, 512),
            nn.ReLU(),
            nn.Linear(512, num_snps)
        )

    def forward(self, X, snp_names, gene_positions):
        X_encoded = self.genetic_encoding(X, snp_names, gene_positions)
        X_transformed = self.transformer(X_encoded)
        outputs = self.prediction(X_transformed)
        
        # Реконструкция исходных SNP
        flat = X_transformed.reshape(X_transformed.size(0), -1)
        recon = self.reconstruction_head(flat)
        
        return outputs, recon

class PhenotypePredictionModelNoRecon(nn.Module):
    """Модель с вниманием, без реконструкции"""
    def __init__(self, num_snps, num_genes_total, embed_dim, num_heads, ff_dim,
                 num_layers, num_classes, snp_genes_df, gene_list, max_snp_per_gene=20, 
                 dropout_rate=0.2, num_attention_heads=2):
        super(PhenotypePredictionModelNoRecon, self).__init__()
        self.genetic_encoding = GeneticEncodingBlock(
            num_genes_total, embed_dim, snp_genes_df, gene_list, 
            max_snp_per_gene, num_attention_heads
        )
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, num_layers, dropout_rate)
        self.prediction = PredictionBlock(embed_dim, num_classes, num_genes_total, dropout_rate)

    def forward(self, X, snp_names, gene_positions):
        X_encoded = self.genetic_encoding(X, snp_names, gene_positions)
        X_transformed = self.transformer(X_encoded)
        outputs = self.prediction(X_transformed)
        return outputs  # только предсказания, без реконструкции


class PhenotypePredictionModelConvNoRecon(nn.Module):
    """Модель с усреднением, без реконструкции"""
    def __init__(self, num_snps, num_genes_total, embed_dim, num_heads, ff_dim,
                 num_layers, num_classes, snp_genes_df, gene_list, max_snp_per_gene=1000, 
                 dropout_rate=0.2):
        super(PhenotypePredictionModelConvNoRecon, self).__init__()
        self.genetic_encoding = GeneticEncodingBlockConv(
            num_genes_total, embed_dim, snp_genes_df, gene_list, max_snp_per_gene
        )
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, num_layers, dropout_rate)
        self.prediction = PredictionBlock(embed_dim, num_classes, num_genes_total, dropout_rate)

    def forward(self, X, snp_names, gene_positions):
        X_encoded = self.genetic_encoding(X, snp_names, gene_positions)
        X_transformed = self.transformer(X_encoded)
        outputs = self.prediction(X_transformed)
        return outputs  # только предсказания, без реконструкции

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Общее количество обучаемых параметров: {total_params:,}")
    return total_params


class ChunkedDataset(Dataset):
    def __init__(self, X_chunks, y, chunk_snp_names_list):
        self.X_chunks = X_chunks
        self.y = torch.tensor(y, dtype=torch.float32)
        self.chunk_snp_names_list = chunk_snp_names_list
        self.num_samples = len(y)
        
        for chunk in X_chunks:
            assert chunk.shape[0] == self.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_parts = []
        for chunk in self.X_chunks:
            x_parts.append(torch.tensor(chunk[idx], dtype=torch.float32))
        x = torch.cat(x_parts)
        y = self.y[idx]
        return x, y


def plot_histograms(y_train, y_test, train_preds, test_preds, save_path='plots/histograms.png'):
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(y_train[:, 0], bins=30, alpha=0.5, label='Actual')
    axes[0, 0].hist(train_preds[:, 0], bins=30, alpha=0.5, label='Predicted')
    axes[0, 0].set_title('Train: OilContent')
    axes[0, 0].legend()
    
    axes[0, 1].hist(y_train[:, 1], bins=30, alpha=0.5, label='Actual')
    axes[0, 1].hist(train_preds[:, 1], bins=30, alpha=0.5, label='Predicted')
    axes[0, 1].set_title('Train: ProteinContent')
    axes[0, 1].legend()
    
    axes[1, 0].hist(y_test[:, 0], bins=30, alpha=0.5, label='Actual')
    axes[1, 0].hist(test_preds[:, 0], bins=30, alpha=0.5, label='Predicted')
    axes[1, 0].set_title('Test: OilContent')
    axes[1, 0].legend()
    
    axes[1, 1].hist(y_test[:, 1], bins=30, alpha=0.5, label='Actual')
    axes[1, 1].hist(test_preds[:, 1], bins=30, alpha=0.5, label='Predicted')
    axes[1, 1].set_title('Test: ProteinContent')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """
    Расчет MAPE (Mean Absolute Percentage Error)
    
    Параметры:
    ----------
    y_true : array-like, фактические значения
    y_pred : array-like, предсказанные значения
    epsilon : float, маленькое число для защиты от деления на ноль
    
    Возвращает:
    ----------
    mape : float, средняя абсолютная процентная ошибка (%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Защита от деления на ноль
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return mape


def print_statistics_with_mape(y_train, y_test, train_preds, test_preds, feature_names):
    """
    Вывод статистики с MAE и MAPE
    """
    print("\n📊 СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
    print("-"*80)
    
    for i, name in enumerate(feature_names):
        # Train
        if len(np.unique(train_preds[:, i])) > 1:
            train_corr = pearsonr(y_train[:, i], train_preds[:, i])[0]
        else:
            train_corr = 0.0
        
        train_mae = np.mean(np.abs(y_train[:, i] - train_preds[:, i]))
        train_mape = calculate_mape(y_train[:, i], train_preds[:, i])
        
        # Test
        if len(np.unique(test_preds[:, i])) > 1:
            test_corr = pearsonr(y_test[:, i], test_preds[:, i])[0]
        else:
            test_corr = 0.0
        
        test_mae = np.mean(np.abs(y_test[:, i] - test_preds[:, i]))
        test_mape = calculate_mape(y_test[:, i], test_preds[:, i])
        
        print(f"\n{name}:")
        print(f"   Train: r={train_corr:.4f}, MAE={train_mae:.4f}, MAPE={train_mape:.2f}%")
        print(f"   Test:  r={test_corr:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")

def print_statistics(y_train, y_test, train_preds, test_preds):
    print("\n📊 СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
    print("-"*60)
    
    for i, name in enumerate(['OilContent', 'ProteinContent']):
        if len(np.unique(train_preds[:, i])) > 1:
            train_corr = pearsonr(y_train[:, i], train_preds[:, i])[0]
        else:
            train_corr = 0.0
            
        if len(np.unique(test_preds[:, i])) > 1:
            test_corr = pearsonr(y_test[:, i], test_preds[:, i])[0]
        else:
            test_corr = 0.0
        
        train_mae = np.mean(np.abs(y_train[:, i] - train_preds[:, i]))
        test_mae = np.mean(np.abs(y_test[:, i] - test_preds[:, i]))
        
        print(f"\n{name}:")
        print(f"   Train: r={train_corr:.4f}, MAE={train_mae:.4f}")
        print(f"   Test:  r={test_corr:.4f}, MAE={test_mae:.4f}")


# ============================================
# ОСНОВНОЙ КОД ОБУЧЕНИЯ
# ============================================

if __name__ == "__main__":
    
    print("="*80)
    print("ОБУЧЕНИЕ МОДЕЛИ НА ДАННЫХ РАПСА (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
    print("="*80)
    
    # ============================================
    # 1. ПАРАМЕТРЫ (ДЛЯ ДВУХ ПРИЗНАКОВ)
    # ============================================

    # Выбор целевого признака
    PREDICT_SINGLE = False
    PREDICT_TARGET = 'FloweringTime'  # для одного признака
    PREDICT_TARGETS = ['FloweringTime', 'GlucosinolateContent']  # для двух признаков

    TOP_N_GENES = 2000
    MAX_SNP_PER_GENE = 1000
    CHUNK_SIZE = 500
    MIN_SNPS_PER_GENE = 15
    # GENE_SELECTION_STRATEGY = 'weighted'
    GENE_SELECTION_STRATEGY = 'uniform_per_chromosome'

    # Модель
    embed_dim = 48
    num_heads = 1
    ff_dim = 128
    num_layers = 4
    num_classes = 1 if PREDICT_SINGLE else 2
    batch_size = 8
    num_epochs = 200
    learning_rate = 0.0005
    weight_decay = 0.1
    dropout_rate = 0.3

    USE_RECONSTRUCTION = True
    USE_ATTENTION = False
    USE_AUGMENTATION = False
    AUGMENTATION_RATE = 0.5
    
    # Пути к файлам
    data_dir = 'data'
    processed_dir = 'processed_data'
    
    # ============================================
    # 2. ЗАГРУЗКА МЕТАДАННЫХ
    # ============================================
    print("\n1. ЗАГРУЗКА МЕТАДАННЫХ")
    print("-"*60)
    
    snp_genes_df = pd.read_csv(f'{processed_dir}/snp_genes_gff_model.csv')
    print(f"📊 SNP-TO-GENE: {len(snp_genes_df)} записей, {snp_genes_df['Gene'].nunique()} генов")
    
    pheno = pd.read_csv(f'{data_dir}/GSTP013.pheno', sep='\t')
    print(f"📊 Фенотипы: {pheno.shape}")
    
    # ============================================
    # 3. ФИЛЬТРАЦИЯ ПО ФЕНОТИПАМ
    # ============================================
    print("\n2. ФИЛЬТРАЦИЯ ПО ФЕНОТИПАМ")
    print("-"*60)

    if PREDICT_SINGLE:
        pheno_filtered = pheno[['LINE'] + [PREDICT_TARGET]].dropna()
    else:
        pheno_filtered = pheno[['LINE'] + PREDICT_TARGETS].dropna()

    print(f"📊 Образцов с признаком: {len(pheno_filtered)}")

    sample_ids = set(pheno_filtered['LINE'].astype(str).str.strip())
    
    # ============================================
    # 4. ЗАГРУЗКА FAM
    # ============================================
    print("\n3. ЗАГРУЗКА FAM ФАЙЛА")
    print("-"*60)
    
    fam = pd.read_csv(f'{data_dir}/Darmor-bzh_991samples_snp.fam', 
                      sep='\s+', header=None,
                      names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'])
    
    sample_indices = [i for i, sid in enumerate(fam['IID'].astype(str).str.strip()) 
                     if sid in sample_ids]
    
    print(f"   - Найдено {len(sample_indices)} образцов")
    
    if len(sample_indices) == 0:
        print("❌ ОШИБКА: Не найдено образцов!")
        exit()
    
    # ============================================
    # 5. ЗАГРУЗКА BIM
    # ============================================
    print("\n4. ЗАГРУЗКА BIM ФАЙЛА")
    print("-"*60)
    
    bim = pd.read_csv(f'{data_dir}/Darmor-bzh_991samples_snp.bim', sep='\t', header=None,
                      names=['CHR', 'SNP_ID', 'CM', 'POS', 'A1', 'A2'])
    
    print(f"   - Всего SNP в BIM: {len(bim)}")
    
    # ============================================
    # 6. ОТБОР ГЕНОВ
    # ============================================
    print("\n5. ОТБОР ГЕНОВ")
    print("-"*60)

    # Загружаем GFF для информации о расположении генов
    genes_gff = pd.read_csv(f'{processed_dir}/snp_genes_gff_full.csv')

    # Выбираем стратегию
    if GENE_SELECTION_STRATEGY == 'genome_coverage':
        selected_genes, gene_stats = select_genes_by_genome_coverage(
            snp_genes_df, genes_gff, top_n=TOP_N_GENES
        )
    elif GENE_SELECTION_STRATEGY == 'uniform_position':
        selected_genes, gene_stats = select_genes_uniform_by_position(
            snp_genes_df, genes_gff, top_n=TOP_N_GENES
        )
    elif GENE_SELECTION_STRATEGY == 'uniform_per_chromosome':
        selected_genes, gene_stats = select_genes_uniform_per_chromosome(
            snp_genes_df, genes_gff, top_n=TOP_N_GENES, min_snps_per_gene=MIN_SNPS_PER_GENE
        )
    elif GENE_SELECTION_STRATEGY == 'weighted':
        selected_genes, gene_stats = select_genes_weighted(snp_genes_df, top_n=TOP_N_GENES, strategy='weighted')
    elif GENE_SELECTION_STRATEGY == 'top':
        selected_genes, gene_stats = select_genes_weighted(snp_genes_df, top_n=TOP_N_GENES, strategy='top')
    else:
        selected_genes, gene_stats = select_genes_weighted(snp_genes_df, top_n=TOP_N_GENES, strategy='balanced')
    
    # Фильтруем snp_genes_df (ВАЖНО: создаем snp_genes_filtered!)
    snp_genes_filtered = filter_snp_genes_by_genes(snp_genes_df, selected_genes)
    print(f"\n   - После фильтрации: {len(snp_genes_filtered)} записей")
    
    # Сохраняем список SNP для загрузки (ВАЖНО: создаем snps_to_load!)
    snps_to_load = snp_genes_filtered['SNP_ID'].tolist()
    print(f"   - SNP для загрузки: {len(snps_to_load)}")
    # ============================================
    # 7. РАСЧЕТ ПОЗИЦИЙ ГЕНОВ
    # ============================================
    print("\n6. РАСЧЕТ ПОЗИЦИЙ ГЕНОВ")
    print("-"*60)

    genes_df = pd.read_csv(f'{processed_dir}/snp_genes_gff_full.csv')
    print(f"📊 GFF файл: {len(genes_df)} записей")
    print(f"   - Уникальных генов в GFF: {genes_df['GENE_NAME'].nunique()}")
    print(f"   - Примеры имен генов в GFF: {genes_df['GENE_NAME'].head(5).tolist()}")

    gene_df_unique = genes_df[['GENE_NAME', 'CHR', 'GENE_START', 'GENE_END']].drop_duplicates()
    gene_df_unique['GENE_MID'] = (gene_df_unique['GENE_START'] + gene_df_unique['GENE_END']) / 2
    gene_df_unique = gene_df_unique.sort_values(['CHR', 'GENE_MID'])

    chrom_offsets = {}
    offset = 0
    for chrom in gene_df_unique['CHR'].unique():
        chrom_data = gene_df_unique[gene_df_unique['CHR'] == chrom]
        chrom_offsets[chrom] = offset
        offset += chrom_data['GENE_MID'].max() + 1e6

    total_offset = offset
    gene_positions_all = {
        row['GENE_NAME']: (row['GENE_MID'] + chrom_offsets[row['CHR']]) / total_offset
        for _, row in gene_df_unique.iterrows()
    }

    print(f"   - Всего позиций рассчитано: {len(gene_positions_all)}")
    
    # Маппинг имен (убираем префикс gene-)
    def normalize_gene_name(name):
        if name.startswith('gene-'):
            return name[5:]
        return name

    final_gene_positions = {}
    for gene in selected_genes:
        normalized = normalize_gene_name(gene)
        if normalized in gene_positions_all:
            final_gene_positions[gene] = gene_positions_all[normalized]
        else:
            # Заглушка
            final_gene_positions[gene] = 0.5

    gene_positions = final_gene_positions
    print(f"   - Позиции для {len(gene_positions)} генов")
    
    # ============================================
    # 8. ЗАГРУЗКА SNP ДАННЫХ
    # ============================================
    print("\n7. ЗАГРУЗКА SNP ДАННЫХ")
    print("-"*60)
    
    bed_file = f'{data_dir}/Darmor-bzh_991samples_snp.bed'
    bim_file = f'{data_dir}/Darmor-bzh_991samples_snp.bim'
    fam_file = f'{data_dir}/Darmor-bzh_991samples_snp.fam'
    
    X_chunks, chunk_snp_names_list = load_snp_data_optimized(
        bed_file, bim_file, fam_file, 
        snps_to_load, sample_indices,
        chunk_size=CHUNK_SIZE,
        use_float16=False
    )
    
    # ============================================
    # 9. КОДИРОВАНИЕ SNP
    # ============================================
    print("\n8. ЧАСТОТНОЕ КОДИРОВАНИЕ SNP")
    print("-"*60)
    
    X_encoded_chunks = []
    for i, chunk_data in enumerate(tqdm(X_chunks, desc="   - Кодирование")):
        encoded = freq_encode_snp_chunk_optimized(chunk_data)
        X_encoded_chunks.append(encoded)
        del chunk_data
        gc.collect()
    
    # ============================================
    # 10. ПОДГОТОВКА ЦЕЛЕВЫХ ПЕРЕМЕННЫХ
    # ============================================
    print("\n9. ПОДГОТОВКА ЦЕЛЕВЫХ ПЕРЕМЕННЫХ")
    print("-"*60)

    # Получаем все значения
    y_all_full = pheno_filtered.set_index('LINE').loc[fam['IID'].astype(str).iloc[sample_indices].values]

    if PREDICT_SINGLE:
        # Предсказываем один признак
        y_all = y_all_full[PREDICT_TARGET].values.reshape(-1, 1)
        print(f"   - Предсказываем: {PREDICT_TARGET}")
    else:
        # Предсказываем оба признака
        y_all = y_all_full[PREDICT_TARGETS].values
        print(f"   - Предсказываем: {PREDICT_TARGETS[0]} и {PREDICT_TARGETS[1]}")

    print(f"   - y shape: {y_all.shape}")

    # Стандартизация
    from sklearn.preprocessing import StandardScaler
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_all)

    print(f"   - После стандартизации: {y_scaled.shape}")
    
    # ============================================
    # 11. РАЗДЕЛЕНИЕ ДАННЫХ
    # ============================================
    print("\n10. РАЗДЕЛЕНИЕ ДАННЫХ")
    print("-"*60)
    
    all_indices = np.arange(len(y_scaled))
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED)
    
    print(f"   - Train: {len(train_idx)} образцов")
    print(f"   - Val: {len(val_idx)} образцов")
    print(f"   - Test: {len(test_idx)} образцов")
    
    def get_chunks_for_indices(chunks, indices):
        return [chunk[indices] for chunk in chunks]
    
    X_train_chunks = get_chunks_for_indices(X_encoded_chunks, train_idx)
    X_val_chunks = get_chunks_for_indices(X_encoded_chunks, val_idx)
    X_test_chunks = get_chunks_for_indices(X_encoded_chunks, test_idx)
    
    y_train = y_scaled[train_idx]
    y_val = y_scaled[val_idx]
    y_test = y_scaled[test_idx]
    
    # ============================================
    # 12. АУГМЕНТАЦИЯ (РАБОТАЮЩАЯ ВЕРСИЯ)
    # ============================================
    print("\n11. АУГМЕНТАЦИЯ ДАННЫХ")
    print("-"*60)

    if USE_AUGMENTATION and len(train_idx) > 0:
        # 1. Аугментируем объединенные данные
        print("   - Объединение чанков для аугментации...")
        X_train_combined = np.hstack(X_train_chunks)  # [189, total_snps]
        
        # Применяем аугментацию
        X_aug, y_aug = mixup_data_np(X_train_combined, y_train, aug_count=AUGMENTATION_RATE, lam=0.3)
        
        # 2. Разбиваем аугментированные данные обратно на чанки
        print("   - Разбиение аугментированных данных на чанки...")
        chunk_sizes = [chunk.shape[1] for chunk in X_train_chunks]
        X_aug_chunks = []
        start_idx = 0
        for size in chunk_sizes:
            X_aug_chunks.append(X_aug[:, start_idx:start_idx + size])
            start_idx += size
        
        # 3. Создаем два отдельных Dataset
        print("   - Создание Dataset для оригинальных и аугментированных данных...")
        
        # Оригинальный Dataset
        train_dataset_original = ChunkedDataset(X_train_chunks, y_train, chunk_snp_names_list)
        
        # Аугментированный Dataset (используем те же имена SNP)
        train_dataset_augmented = ChunkedDataset(X_aug_chunks, y_aug, chunk_snp_names_list)
        
        # 4. Объединяем Dataset'ы
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([train_dataset_original, train_dataset_augmented])
        
        y_train_final = np.vstack([y_train, y_aug])
        print(f"   - Всего образцов после аугментации: {len(y_train_final)}")
        print(f"   - Оригинальных: {len(y_train)}")
        print(f"   - Аугментированных: {len(y_aug)}")
    else:
        train_dataset = ChunkedDataset(X_train_chunks, y_train, chunk_snp_names_list)
        print("   - Аугментация пропущена")

    # ============================================
    # 13. СОЗДАНИЕ DATALOADER
    # ============================================
    print("\n12. СОЗДАНИЕ DATALOADER'ов")
    print("-"*60)

    all_snp_names = []
    for snp_names in chunk_snp_names_list:
        all_snp_names.extend(snp_names)
    print(f"   - Всего SNP: {len(all_snp_names)}")

    val_dataset = ChunkedDataset(X_val_chunks, y_val, chunk_snp_names_list)
    test_dataset = ChunkedDataset(X_test_chunks, y_test, chunk_snp_names_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ============================================
    # 14. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
    # ============================================
    print("\n13. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ")
    print("-"*60)

    # Определяем количество SNP и генов для модели
    num_snps = sum(chunk.shape[1] for chunk in X_train_chunks)
    num_genes_total = len(selected_genes)

    print(f"   - Всего SNP: {num_snps}")
    print(f"   - Всего генов: {num_genes_total}")
    print(f"   - Max SNP per gene: {MAX_SNP_PER_GENE}")
    print(f"   - Тип агрегации: {'Attention' if USE_ATTENTION else 'Simple averaging'}")
    print(f"   - Реконструкция: {'Да' if USE_RECONSTRUCTION else 'Нет'}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - Устройство: {device}")

    if device == 'cuda':
        print(f"   - GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()

    # Выбираем модель в зависимости от флагов
    if USE_RECONSTRUCTION:
        if USE_ATTENTION:
            model = PhenotypePredictionModel(
                num_snps=num_snps,
                num_genes_total=num_genes_total,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                snp_genes_df=snp_genes_filtered,
                gene_list=selected_genes,
                max_snp_per_gene=MAX_SNP_PER_GENE,
                dropout_rate=dropout_rate
            ).to(device)
        else:
            model = PhenotypePredictionModelConv(
                num_snps=num_snps,
                num_genes_total=num_genes_total,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                snp_genes_df=snp_genes_filtered,
                gene_list=selected_genes,
                max_snp_per_gene=MAX_SNP_PER_GENE,
                dropout_rate=dropout_rate
            ).to(device)
    else:
        if USE_ATTENTION:
            model = PhenotypePredictionModelNoRecon(
                num_snps=num_snps,
                num_genes_total=num_genes_total,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                snp_genes_df=snp_genes_filtered,
                gene_list=selected_genes,
                max_snp_per_gene=MAX_SNP_PER_GENE,
                dropout_rate=dropout_rate
            ).to(device)
        else:
            model = PhenotypePredictionModelConvNoRecon(
                num_snps=num_snps,
                num_genes_total=num_genes_total,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                snp_genes_df=snp_genes_filtered,
                gene_list=selected_genes,
                max_snp_per_gene=MAX_SNP_PER_GENE,
                dropout_rate=dropout_rate
            ).to(device)

    total_params = count_parameters(model)

    # ============================================
    # 15. ОПТИМИЗАТОР И ФУНКЦИЯ ПОТЕРЬ
    # ============================================
    print("\n14. НАСТРОЙКА ОПТИМИЗАТОРА")
    print("-"*60)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if PREDICT_SINGLE:
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.HuberLoss(delta=1.0)

    # ============================================
    # 16. ЦИКЛ ОБУЧЕНИЯ
    # ============================================
    print("\n15. НАЧАЛО ОБУЧЕНИЯ")
    print("-"*60)

    best_val_loss = float('inf')
    best_val_corr = -1
    patience = 20
    patience_counter = 0
    best_model_path = 'best_rapeseed_model.pth'
    nan_count = 0

    for epoch in range(num_epochs):
        # ========== TRAINING ==========
        model.train()
        total_train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            if USE_RECONSTRUCTION:
                outputs, recon = model(batch_X, all_snp_names, gene_positions)
                loss = criterion(outputs, batch_y)
            else:
                outputs = model(batch_X, all_snp_names, gene_positions)
                loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > 3:
                    print("❌ Too many NaN losses, stopping...")
                    break
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(batch_y.cpu().numpy())
        
        if nan_count > 3:
            break
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        
        # Вычисление корреляций на train
        if len(train_preds) > 0:
            train_preds = np.concatenate(train_preds, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            
            if PREDICT_SINGLE:
                if len(np.unique(train_preds[:, 0])) > 1:
                    train_corr = pearsonr(train_targets[:, 0], train_preds[:, 0])[0]
                else:
                    train_corr = 0.0
                train_corr_oil = train_corr
                train_corr_protein = 0.0
            else:
                if len(np.unique(train_preds[:, 0])) > 1:
                    train_corr_oil = pearsonr(train_targets[:, 0], train_preds[:, 0])[0]
                else:
                    train_corr_oil = 0.0
                if len(np.unique(train_preds[:, 1])) > 1:
                    train_corr_protein = pearsonr(train_targets[:, 1], train_preds[:, 1])[0]
                else:
                    train_corr_protein = 0.0
        else:
            train_corr_oil = train_corr_protein = 0.0
        
        # ========== VALIDATION ==========
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                if USE_RECONSTRUCTION:
                    outputs, recon = model(batch_X, all_snp_names, gene_positions)
                    loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_X, all_snp_names, gene_positions)
                    loss = criterion(outputs, batch_y)
                
                total_val_loss += loss.item()
                
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        # Вычисление корреляций на validation
        if len(val_preds) > 0:
            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            
            if PREDICT_SINGLE:
                if len(np.unique(val_preds[:, 0])) > 1:
                    val_corr = pearsonr(val_targets[:, 0], val_preds[:, 0])[0]
                else:
                    val_corr = 0.0
                avg_val_corr = val_corr
                val_corr_oil = val_corr
                val_corr_protein = 0.0
            else:
                if len(np.unique(val_preds[:, 0])) > 1:
                    val_corr_oil = pearsonr(val_targets[:, 0], val_preds[:, 0])[0]
                else:
                    val_corr_oil = 0.0
                if len(np.unique(val_preds[:, 1])) > 1:
                    val_corr_protein = pearsonr(val_targets[:, 1], val_preds[:, 1])[0]
                else:
                    val_corr_protein = 0.0
                avg_val_corr = (val_corr_oil + val_corr_protein) / 2
        else:
            val_corr_oil = val_corr_protein = 0.0
            avg_val_corr = 0.0
        
        # Обновление learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # ========== LOGGING ==========
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            
            if PREDICT_SINGLE:
                print(f'   Train Loss: {avg_train_loss:.6f}, Train Corr: {train_corr_oil:.4f}')
                print(f'   Val Loss: {avg_val_loss:.6f}, Val Corr: {val_corr_oil:.4f}')
            else:
                print(f'   Train Loss: {avg_train_loss:.6f}, Train Corr: Oil={train_corr_oil:.4f}, Protein={train_corr_protein:.4f}')
                print(f'   Val Loss: {avg_val_loss:.6f}, Val Corr: Oil={val_corr_oil:.4f}, Protein={val_corr_protein:.4f}')
        
        # ========== SAVE BEST MODEL ==========
        if avg_val_corr > best_val_corr:
            best_val_corr = avg_val_corr
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'val_corr_oil': val_corr_oil,
                'val_corr_protein': val_corr_protein,
                'selected_genes': selected_genes,
                'gene_stats': gene_stats
            }, best_model_path)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'   ✅ New best model! Val Corr: {best_val_corr:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✅ Early stopping at epoch {epoch+1}")
                break
        
        # Ранняя остановка при плохом предсказании
        if avg_val_corr < -0.1 and epoch > 20:
            print(f"\n⚠️  Model is predicting poorly, stopping early")
            break

    print("\n✅ TRAINING COMPLETED")
        
    # ============================================
    # 17. ОЦЕНКА НА ТЕСТЕ
    # ============================================
    print("\n16. ОЦЕНКА МОДЕЛИ НА ТЕСТЕ")
    print("-"*80)

    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds = []
    test_targets = []
    test_pred_losses = []  # Добавлено
    test_recon_losses = []  # Добавлено

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            if USE_RECONSTRUCTION:
                outputs, recon = model(batch_X, all_snp_names, gene_positions)
                loss_pred = criterion(outputs, batch_y)
                loss_recon = F.mse_loss(recon, batch_X)
                test_pred_losses.append(loss_pred.item())
                test_recon_losses.append(loss_recon.item())
            else:
                outputs = model(batch_X, all_snp_names, gene_positions)
                loss_pred = criterion(outputs, batch_y)
                test_pred_losses.append(loss_pred.item())
            
            test_preds.append(outputs.cpu().numpy())
            test_targets.append(batch_y.cpu().numpy())

    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    avg_test_pred_loss = np.mean(test_pred_losses) if test_pred_losses else 0.0
    avg_test_recon_loss = np.mean(test_recon_losses) if test_recon_losses else 0.0

    # Обратное масштабирование
    test_preds_original = y_scaler.inverse_transform(test_preds)
    test_targets_original = y_scaler.inverse_transform(test_targets)

    print(f"\n📊 Потери на тестовом наборе:")
    print(f"   - Потеря предсказания: {avg_test_pred_loss:.6f}")
    if USE_RECONSTRUCTION:
        print(f"   - Потеря реконструкции: {avg_test_recon_loss:.6f}")

    # Для двух признаков
    if not PREDICT_SINGLE:
        feature_names = PREDICT_TARGETS
        print("\n📊 СТАТИСТИКА ПРЕДСКАЗАНИЙ (TEST):")
        print("-"*80)
        
        test_results = {}
        for i, name in enumerate(feature_names):
            if len(np.unique(test_preds_original[:, i])) > 1:
                test_corr = pearsonr(test_targets_original[:, i], test_preds_original[:, i])[0]
            else:
                test_corr = 0.0
            
            test_mae = np.mean(np.abs(test_targets_original[:, i] - test_preds_original[:, i]))
            test_mape = calculate_mape(test_targets_original[:, i], test_preds_original[:, i])
            test_rmse = np.sqrt(np.mean((test_targets_original[:, i] - test_preds_original[:, i])**2))
            
            test_results[name] = {
                'corr': test_corr,
                'mae': test_mae,
                'mape': test_mape,
                'rmse': test_rmse
            }
            
            print(f"\n{name}:")
            print(f"   Test:  r={test_corr:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.2f}%")
        
        # Сохраняем результаты
        results = {
            'test_corr_oil': test_results[PREDICT_TARGETS[0]]['corr'],
            'test_corr_protein': test_results[PREDICT_TARGETS[1]]['corr'],
            'test_mae_oil': test_results[PREDICT_TARGETS[0]]['mae'],
            'test_mae_protein': test_results[PREDICT_TARGETS[1]]['mae'],
            'test_rmse_oil': test_results[PREDICT_TARGETS[0]]['rmse'],
            'test_rmse_protein': test_results[PREDICT_TARGETS[1]]['rmse'],
            'test_mape_oil': test_results[PREDICT_TARGETS[0]]['mape'],
            'test_mape_protein': test_results[PREDICT_TARGETS[1]]['mape'],
            'test_pred_loss': avg_test_pred_loss,
            'test_recon_loss': avg_test_recon_loss,
            'best_val_loss': checkpoint['val_loss'],
            'best_val_corr': checkpoint.get('val_corr_oil', best_val_corr),
            'best_val_corr_oil': checkpoint.get('val_corr_oil', 0),
            'best_val_corr_protein': checkpoint.get('val_corr_protein', 0),
            'selected_genes': selected_genes,
            'gene_stats': gene_stats,
            'alpha_recon': checkpoint.get('alpha_recon', 0)
        }
        
    else:
        feature_names = [PREDICT_TARGET]
        print("\n📊 СТАТИСТИКА ПРЕДСКАЗАНИЙ (TEST):")
        print("-"*80)
        
        if len(np.unique(test_preds_original[:, 0])) > 1:
            test_corr = pearsonr(test_targets_original[:, 0], test_preds_original[:, 0])[0]
        else:
            test_corr = 0.0
        
        test_mae = np.mean(np.abs(test_targets_original[:, 0] - test_preds_original[:, 0]))
        test_mape = calculate_mape(test_targets_original[:, 0], test_preds_original[:, 0])
        test_rmse = np.sqrt(np.mean((test_targets_original[:, 0] - test_preds_original[:, 0])**2))
        
        print(f"\n{PREDICT_TARGET}:")
        print(f"   Test:  r={test_corr:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.2f}%")
        
        # Сохраняем результаты
        results = {
            'test_corr': test_corr,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'test_pred_loss': avg_test_pred_loss,
            'test_recon_loss': avg_test_recon_loss,
            'best_val_loss': checkpoint['val_loss'],
            'best_val_corr': checkpoint.get('val_corr_oil', best_val_corr),
            'selected_genes': selected_genes,
            'gene_stats': gene_stats,
            'alpha_recon': checkpoint.get('alpha_recon', 0)
        }

    # Сохраняем результаты в файл
    with open('rapeseed_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\n💾 Результаты сохранены в 'rapeseed_results.pkl'")

    # Построение графиков
    os.makedirs('plots', exist_ok=True)

    if PREDICT_SINGLE:
        # График для одного признака
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График рассеяния
        axes[0].scatter(test_targets_original[:, 0], test_preds_original[:, 0], alpha=0.5, edgecolors='k', linewidth=0.5)
        min_val = min(test_targets_original[:, 0].min(), test_preds_original[:, 0].min())
        max_val = max(test_targets_original[:, 0].max(), test_preds_original[:, 0].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        axes[0].set_xlabel(f'Actual {PREDICT_TARGET}', fontsize=12)
        axes[0].set_ylabel(f'Predicted {PREDICT_TARGET}', fontsize=12)
        axes[0].set_title(f'{PREDICT_TARGET}\n(r={test_corr:.4f}, MAPE={test_mape:.1f}%)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Гистограмма остатков
        residuals = test_targets_original[:, 0] - test_preds_original[:, 0]
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Residuals Distribution\n(mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f})', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/rapeseed_test_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Сохраняем также предсказания для дальнейшего анализа
        predictions_df = pd.DataFrame({
            'Actual': test_targets_original[:, 0],
            'Predicted': test_preds_original[:, 0],
            'Residuals': residuals
        })
        predictions_df.to_csv('plots/test_predictions.csv', index=False)
        
    else:
        # Графики для двух признаков
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Признак 1
        i = 0
        # График рассеяния
        axes[0, 0].scatter(test_targets_original[:, i], test_preds_original[:, i], alpha=0.5, edgecolors='k', linewidth=0.5)
        min_val = min(test_targets_original[:, i].min(), test_preds_original[:, i].min())
        max_val = max(test_targets_original[:, i].max(), test_preds_original[:, i].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        axes[0, 0].set_xlabel(f'Actual {PREDICT_TARGETS[i]}', fontsize=12)
        axes[0, 0].set_ylabel(f'Predicted {PREDICT_TARGETS[i]}', fontsize=12)
        axes[0, 0].set_title(f'{PREDICT_TARGETS[i]}\n(r={test_results[PREDICT_TARGETS[i]]["corr"]:.4f}, MAPE={test_results[PREDICT_TARGETS[i]]["mape"]:.1f}%)', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Гистограмма остатков для признака 1
        residuals_1 = test_targets_original[:, i] - test_preds_original[:, i]
        axes[0, 1].hist(residuals_1, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title(f'Residuals - {PREDICT_TARGETS[i]}\n(mean={np.mean(residuals_1):.4f}, std={np.std(residuals_1):.4f})', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot для признака 1
        from scipy import stats
        stats.probplot(residuals_1, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title(f'Q-Q Plot - {PREDICT_TARGETS[i]}', fontsize=12)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Признак 2
        i = 1
        # График рассеяния
        axes[1, 0].scatter(test_targets_original[:, i], test_preds_original[:, i], alpha=0.5, edgecolors='k', linewidth=0.5)
        min_val = min(test_targets_original[:, i].min(), test_preds_original[:, i].min())
        max_val = max(test_targets_original[:, i].max(), test_preds_original[:, i].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        axes[1, 0].set_xlabel(f'Actual {PREDICT_TARGETS[i]}', fontsize=12)
        axes[1, 0].set_ylabel(f'Predicted {PREDICT_TARGETS[i]}', fontsize=12)
        axes[1, 0].set_title(f'{PREDICT_TARGETS[i]}\n(r={test_results[PREDICT_TARGETS[i]]["corr"]:.4f}, MAPE={test_results[PREDICT_TARGETS[i]]["mape"]:.1f}%)', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Гистограмма остатков для признака 2
        residuals_2 = test_targets_original[:, i] - test_preds_original[:, i]
        axes[1, 1].hist(residuals_2, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residuals', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title(f'Residuals - {PREDICT_TARGETS[i]}\n(mean={np.mean(residuals_2):.4f}, std={np.std(residuals_2):.4f})', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Q-Q plot для признака 2
        stats.probplot(residuals_2, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title(f'Q-Q Plot - {PREDICT_TARGETS[i]}', fontsize=12)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/rapeseed_test_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Сохраняем предсказания для дальнейшего анализа
        predictions_df = pd.DataFrame({
            f'Actual_{PREDICT_TARGETS[0]}': test_targets_original[:, 0],
            f'Predicted_{PREDICT_TARGETS[0]}': test_preds_original[:, 0],
            f'Residuals_{PREDICT_TARGETS[0]}': residuals_1,
            f'Actual_{PREDICT_TARGETS[1]}': test_targets_original[:, 1],
            f'Predicted_{PREDICT_TARGETS[1]}': test_preds_original[:, 1],
            f'Residuals_{PREDICT_TARGETS[1]}': residuals_2
        })
        predictions_df.to_csv('plots/test_predictions.csv', index=False)

    print(f"\n📊 Графики сохранены:")
    print(f"   - plots/rapeseed_test_results.png")
    print(f"   - plots/test_predictions.csv")

    # Дополнительная информация о реконструкции
    print(f"\n🔧 Информация о реконструкции:")
    print(f"   - Коэффициент реконструкции (alpha_recon): {results.get('alpha_recon', 0):.6f}")
    print(f"   - Потеря реконструкции на тесте: {avg_test_recon_loss:.6f}")
    print(f"   - Отношение recon_loss/pred_loss: {avg_test_recon_loss/avg_test_pred_loss:.4f}")

    print("\n" + "="*80)
    print("🎉 ОБУЧЕНИЕ И ОЦЕНКА ЗАВЕРШЕНЫ УСПЕШНО!")
    print("="*80)
