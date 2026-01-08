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