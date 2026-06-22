import pandas as pd
import random
import numpy as np # Добавлено для np.concatenate

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
