```markdown
# Scikit-learn main info

## Классификация

| Модель                        | Когда использовать                              | Главные гиперпараметры                              |
|-------------------------------|--------------------------------------------------|------------------------------------------------------|
| LogisticRegression            | Линейные данные, интерпретируемость, быстро      | C=0.1–10, penalty='l2'/'elasticnet', class_weight |
| DecisionTreeClassifier        | Простые нелинейные зависимости                   | max_depth=5–15, min_samples_leaf=2–10               |
| RandomForestClassifier        | Универсальный старт, устойчив к выбросам         | n_estimators=100–500, max_depth=8–25, max_features |
| XGBClassifier / LGBM / CatBoost | Почти всегда лучшее качество                   | learning_rate=0.01–0.3, n_estimators=200–3000, max_depth=3–12 |
| SVC                           | Маленькие выборки, высокая размерность           | kernel='rbf', C=0.1–100, gamma='scale'/'auto'       |
| KNeighborsClassifier          | Очень простые задачи, мало данных                | n_neighbors=5–15, weights='distance'                |

## Регрессия
Аналогично классификации → меняем на  
`RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`

## Важные преобразования
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

StandardScaler()       # → μ=0, σ=1 (самый частый)
MinMaxScaler()         # → [0,1] — для нейросетей, SVM
RobustScaler()         # → устойчив к выбросам

ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

# Scikit-learn main info

## Классификация

| Модель | Когда использовать | Главные гиперпараметры | fit/transform/predict |
|--------|-------------------|------------------------|------------------------|
| **LogisticRegression** | Линейные данные, интерпретируемость, быстро | `C=0.1–10`, `penalty='l2'/'elasticnet'`, `class_weight` | `fit(X_train, y_train)` → `predict(X_test)` |
| **DecisionTreeClassifier** | Простые нелинейные зависимости | `max_depth=5–15`, `min_samples_leaf=2–10` | `fit(X_train, y_train)` → `predict_proba(X_test)` |
| **RandomForestClassifier** | Универсальный старт, устойчив к выбросам | `n_estimators=100–500`, `max_depth=8–25`, `max_features` | `fit(X_train, y_train)` → `predict(X_test)` |
| **XGBClassifier** | Почти всегда лучшее качество | `learning_rate=0.01–0.3`, `n_estimators=200–3000`, `max_depth=3–12` | `fit(X_train, y_train, eval_set=[(X_val, y_val)])` |
| **SVC** | Маленькие выборки, высокая размерность | `kernel='rbf'`, `C=0.1–100`, `gamma='scale'/'auto'` | `fit(X_train_scaled, y_train)` → `predict(X_test_scaled)` |
| **KNeighborsClassifier** | Очень простые задачи, мало данных | `n_neighbors=5–15`, `weights='distance'` | `fit(X_train, y_train)` → `predict(X_test)` |

## Регрессия

| Модель | Когда использовать | Главные гиперпараметры | Особенности |
|--------|-------------------|------------------------|-------------|
| **LinearRegression** | Базовый бейзлайн, линейные зависимости | `fit_intercept=True` | Быстро, интерпретируемо, `coef_`, `intercept_` |
| **RandomForestRegressor** | Универсально, устойчив к шуму | `n_estimators=100–500`, `max_depth=8–25`, `max_features='sqrt'` | Не требует масштабирования, `feature_importances_` |
| **XGBRegressor** | Высокое качество, большие данные | `learning_rate=0.01–0.1`, `n_estimators=500–2000`, `max_depth=3–8` | Быстрый, `early_stopping_rounds=50` |
| **CatBoostRegressor** | Категориальные признаки, табличные данные | `iterations=500–2000`, `learning_rate=0.03–0.1`, `depth=4–10` | Автоматически обрабатывает категории, мало переобучается |
| **LGBMRegressor** | Очень большие данные, скорость | `n_estimators=500–2000`, `learning_rate=0.01–0.1`, `max_depth=3–8` | Самый быстрый из бустингов, экономная память |

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Создаем и обучаем модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # ТОЛЬКО на обучающих данных

# 3. Делаем предсказания
y_train_pred = model.predict(X_train)  # предсказания на обучающих
y_test_pred = model.predict(X_test)    # предсказания на тестовых

# 4. Оцениваем качество
from sklearn.metrics import mean_absolute_error, r2_score
print(f"MAE на тесте: {mean_absolute_error(y_test, y_test_pred):.3f}")
print(f"R² на тесте: {r2_score(y_test, y_test_pred):.3f}")

# Важно: НИКОГДА не используйте данные из теста при обучении!
# Это data leakage (утечка данных)