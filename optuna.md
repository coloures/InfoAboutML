```markdown
# Optuna main info

```python
!pip install optuna -q

import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 20),
        "learning_rate":    trial.suggest_float("learning_rate", 0.008, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
    }
    
    model = XGBClassifier(**params, random_state=42, n_jobs=-1)
    
    score = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='f1_macro', n_jobs=-1
    ).mean()
    
    return score

# Запуск оптимизации
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # 30–80 обычно хватает

print("Лучшие параметры:", study.best_params)
print("Лучший score:   ", study.best_value)

# Визуализация (очень полезно на защите)
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_slice(study)