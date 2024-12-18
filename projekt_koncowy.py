import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- 1. Wczytanie danych ---
file_path = "data/default_of_credit_card_clients.xls"
data = pd.read_excel(file_path, header=1)
data.rename(columns={'default payment next month': 'Default'}, inplace=True)

print("Podstawowe informacje o danych:")
print(data.info())
print(data.head())

# --- 2. Sprawdzanie i naprawa spójności danych ---

# Sprawdzanie brakujących wartości
print("\nBrakujące wartości w danych:")
print(data.isnull().sum())

# Usunięcie duplikatów
print(f"\nLiczba duplikatów przed usunięciem: {data.duplicated().sum()}")
data = data.drop_duplicates()
print(f"Liczba duplikatów po usunięciu: {data.duplicated().sum()}")

# Korekta błędów w zmiennych kategorycznych
data['EDUCATION'] = data['EDUCATION'].replace({0: 4})  # Zastąpienie 0 jako "inne"
data['MARRIAGE'] = data['MARRIAGE'].replace({0: 3})    # Zastąpienie 0 jako "inne"

# Sprawdzanie rozkładu zmiennej celu
print("\nRozkład zmiennej celu (Default):")
print(data['Default'].value_counts())

# Wizualizacja zmiennej celu
sns.countplot(x='Default', data=data)
plt.title("Rozkład zmiennej celu: Default")
plt.show()

# --- 3. Analiza eksploracyjna (EDA) ---

# Korelacja między zmiennymi
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title("Macierz korelacji między zmiennymi")
plt.show()

# Rozkład płci (SEX) i edukacji (EDUCATION)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x='SEX', data=data, ax=axes[0])
axes[0].set_title("Rozkład płci (SEX)")
axes[0].set_xlabel("1 = Mężczyzna, 2 = Kobieta")

sns.countplot(x='EDUCATION', data=data, ax=axes[1])
axes[1].set_title("Rozkład edukacji (EDUCATION)")
axes[1].set_xlabel("1 = Podyplomowe, 2 = Uniwersytet, 3 = Szkoła średnia, 4 = Inne")

plt.tight_layout()
plt.show()

# --- 4. Podział danych na zbiór treningowy i testowy ---

# Zmienne objaśniające (X) i zmienna celu (y)
X = data.drop(columns=['ID', 'Default'])  # Usunięcie ID i zmiennej celu
y = data['Default']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nRozmiary zbiorów danych:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --- 5. Trenowanie modelu klasyfikacji ---

# Użycie modelu RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Parametry do GridSearchCV
params = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("\nPrzeprowadzanie GridSearchCV...")
grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Najlepszy model
best_model = grid_search.best_estimator_
print("\nNajlepsze parametry modelu:")
print(grid_search.best_params_)

# --- 6. Ocena modelu ---

# Predykcja na zbiorze testowym
y_pred = best_model.predict(X_test)

# Metryki oceny
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Macierz błędów
print("\nMacierz błędów:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Przewidywane')
plt.ylabel('Rzeczywiste')
plt.title("Macierz błędów")
plt.show()

# Dokładność modelu
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy:.2f}")

# --- 7. Zapis modelu ---

joblib.dump(best_model, 'credit_card_default_model.pkl')
print("\nModel zapisano jako 'credit_card_default_model.pkl'")
