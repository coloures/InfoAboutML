```markdown
# Pandas main info

```python
import pandas as pd
import numpy as np

# =============================================
# СОЗДАНИЕ И ЗАГРУЗКА
# =============================================
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
df = pd.read_csv('data.csv', sep=';', encoding='utf-8-sig', low_memory=False)
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# =============================================
# ОСМОТР ДАННЫХ
# =============================================
df.head(8)
df.tail(5)
df.shape                  # (строки, столбцы)
df.info()                 # типы + пропуски + память
df.describe()             # статистика по числам
df.describe(include='object')  # по строкам/категориям
df.columns.tolist()
df.dtypes

# =============================================
# ВЫБОР И ФИЛЬТРАЦИЯ
# =============================================
df['column']
df[['col1', 'col2']]
df.loc[5:15, 'name':'salary']
df.iloc[0:10, [1, 3, 5]]
df.query("age > 25 and city == 'Riga'")

df[df['salary'] > 5000]
df[df['gender'].isin(['M', 'F'])]
df[df['text'].str.contains('error', na=False, case=False)]

# =============================================
# ПРОПУСКИ
# =============================================
df.isna().sum()
df.isnull().sum()
df.dropna()
df.dropna(subset=['important_col'])
df.fillna(0)
df['age'].fillna(df['age'].median(), inplace=True)
df.fillna({'age': 30, 'city': 'Unknown'})

# =============================================
# ПРЕОБРАЗОВАНИЯ
# =============================================
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['cat'] = df['cat'].astype('category')
df['salary_log'] = np.log1p(df['salary'].clip(lower=1))

# =============================================
# ГРУППИРОВКА И АГРЕГАЦИЯ
# =============================================
df.groupby('city')['salary'].agg(['mean', 'median', 'count', 'max'])
df.groupby(['city', 'gender']).size().unstack(fill_value=0)

# =============================================
# СЛИЯНИЕ ТАБЛИЦ
# =============================================
pd.merge(df1, df2, on='id', how='left')
df.merge(df2, left_on='user_id', right_on='id', suffixes=('_x', '_y'))

# =============================================
# ДУБЛИКАТЫ, СОРТИРОВКА
# =============================================
df.duplicated().sum()
df.drop_duplicates(subset=['id'], keep='first', inplace=True)
df.sort_values(['salary', 'age'], ascending=[False, True])