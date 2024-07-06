import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import time

def load_and_preprocess_data():
    data = np.array([
        [0.4, 250, 22, 8, "Normal", "Normal", "Normal", "Normal"],
        [0.6, 300, 30, 15, "Normal", "Normal", "Normal", "Normal"],
        [0.8, 850, 45, 12, "High", "High", "High", "Normal"],
        [0.2, 180, 15, 4, "Low", "Low", "Low", "Low"],
        [0.5, 600, 50, 18, "Normal", "Normal", "Normal", "High"],
        [0.3, 400, 35, 20, "Normal", "Normal", "Normal", "High"],
        [0.7, 1000, 10, 2, "High", "High", "Low", "Low"],
        [0.1, 50, 5, 1, "Low", "Low", "Low", "Low"],
        [0.9, 900, 48, 16, "High", "High", "High", "High"],
        [0.45, 320, 28, 14, "Normal", "Normal", "Normal", "Normal"],
        [0.65, 500, 38, 13, "Normal", "Normal", "Normal", "Normal"],
        [0.85, 850, 42, 11, "High", "High", "High", "Normal"],
        [0.25, 190, 17, 5, "Low", "Low", "Low", "Low"],
        [0.35, 300, 31, 10, "Normal", "Normal", "Normal", "Normal"],
        [0.75, 720, 46, 19, "High", "High", "High", "High"],
        [0.55, 480, 20, 7, "Normal", "Normal", "Low", "Normal"]
    ])
    X = data[:, :4].astype(float)
    y_labels = ["Vitamin A", "Vitamin B12", "Vitamin D", "Vitamin E"]
    y = {label: data[:, i + 4] for i, label in enumerate(y_labels)}
    return X, y, y_labels

def train_model(X, y, param_grid, cv, scaler):
    best_models = {}
    for label, y_vitamin in y.items():
        class_counts = np.bincount([0 if val == 'Low' else 1 if val == 'Normal' else 2 for val in y_vitamin])
        min_samples = min(class_counts)
        if min_samples < 2:
            print(f"Skipping training for {label} due to lack of class diversity.")
            continue
        k_neighbors = min(5, min_samples - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y_vitamin)
        X_resampled = scaler.fit_transform(X_resampled)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_models[label] = grid_search.best_estimator_
        y_pred = best_models[label].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{label} - Accuracy: {accuracy}")
        print(classification_report(y_test, y_pred, zero_division=0))
    return best_models

def make_predictions(best_models, scaler, X):
    new_blood_test_results = np.zeros((1, 4))
    for i in range(X.shape[1]):
        range_min = np.min(X[:, i])
        range_max = np.max(X[:, i])
        random_value = np.random.uniform(range_min, range_max)
        new_blood_test_results[0, i] = random_value
    new_blood_test_results = scaler.transform(new_blood_test_results)
    new_predictions = {label: best_models[label].predict(new_blood_test_results) for label in best_models.keys()}
    return new_predictions

def print_recommendations(new_predictions, recommendations):
    for vitamin, prediction in new_predictions.items():
        status = prediction[0]
        print(f"{vitamin}: {status}. {recommendations[status][vitamin]}")

if __name__ == "__main__":
    start_time = time.time()
    X, y, y_labels = load_and_preprocess_data()
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    cv = StratifiedKFold(n_splits=5)
    scaler = StandardScaler()
    best_models = train_model(X, y, param_grid, cv, scaler)
    new_predictions = make_predictions(best_models, scaler, X)
    recommendations = {
        "Low": {
            "Vitamin A": "Добавьте в рацион больше моркови, сладкого картофеля и шпината.",
            "Vitamin B12": "Рекомендуется увеличить потребление мяса, рыбы, молочных продуктов или принимать добавки витамина B12.",
            "Vitamin D": "Больше времени на солнце, употребляйте рыбий жир, яйца и обогащённые продукты.",
            "Vitamin E": "Добавьте в рацион орехи, семена, растительные масла и зелёные листовые овощи."
        },
        "High": {
            "Vitamin A": "Уменьшите потребление продуктов, богатых витамином A.",
            "Vitamin B12": "Снизьте потребление добавок витамина B12.",
            "Vitamin D": "Ограничьте потребление добавок витамина D и избегайте длительного пребывания на солнце.",
            "Vitamin E": "Снизьте потребление продуктов, богатых витамином E."
        },
        "Normal": {
            "Vitamin A": "Поддерживайте текущий образ жизни и рацион.",
            "Vitamin B12": "Поддерживайте текущий образ жизни и рацион.",
            "Vitamin D": "Поддерживайте текущий образ жизни и рацион.",
            "Vitamin E": "Поддерживайте текущий образ жизни и рацион."
        }
    }
    print_recommendations(new_predictions, recommendations)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Время выполнения кода: {execution_time} секунд")
