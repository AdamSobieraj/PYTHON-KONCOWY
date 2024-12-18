# data manipulation
import numpy as np
import pandas as pd
import datetime as dt
import time
from math import sqrt

from PIL._imaging import display
from tqdm import tqdm
import holidays

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# feature selection
from sklearn.feature_selection import RFECV

# pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# optimization and hyperparameters
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# model evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

# interpretowalność modelu
import shap
# shap.initjs()

# zapisanie modelu
import joblib

# settings
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# Odczyt danych
dataframe = pd.read_csv('data/bikerides_day.csv')
dataframe['Date'] = pd.to_datetime(dataframe['Date'])

print(dataframe)

# Wartości brakującw
missing = dataframe.isnull().mean()*100
print(missing)

# Duplikaty dat ?
duplicates = dataframe.loc[dataframe.duplicated(subset='Date', keep=False), :]
print(duplicates)

# Czy posortowane
sort = (dataframe['Date'] == dataframe['Date'].sort_values()).all()
print(sort)

# Weryfikacja dat
dateVeri = pd.date_range(start=dataframe['Date'].min(), end=dataframe['Date'].max()).difference(dataframe['Date'])
print(dateVeri)

for variable in dataframe.select_dtypes(include=np.number).columns:
    fig = plt.figure(figsize=(16, 2))
    fig.suptitle(variable, fontsize=12)
    plt.subplot(121)
    sns.distplot(dataframe[variable], kde=True, rug=False)
    plt.title('Histogram')
    plt.subplot(122)
    plt.boxplot(dataframe[variable])
    plt.title('Wykres pudełkowy')
    plt.show()

plt.figure(figsize=(16, 8), dpi=100)
plt.plot(dataframe['Date'], dataframe['Volume'],
         color='tab:red', label='Volume for every day')
plt.plot(dataframe['Date'], dataframe['Volume'].rolling(7).mean(),
         color='tab:blue', linewidth=4, label='7-dniowa średnia krocząca')
plt.title('Osoby wypożyczające rowery na pewnej trasie w Oslo 2016-2020')
plt.xlabel('Date')
plt.ylabel('Liczba')
plt.xlim([dataframe['Date'].min(), dataframe['Date'].max()])
plt.ylim([0, dataframe['Volume'].max()*1.025])
plt.legend(loc='upper right')
plt.show()

full_years = dataframe.loc[(dataframe['Date'] >= '2017-01-01') & (dataframe['Date'] < '2020-01-01')].copy()
full_years['year'] = [d.year for d in full_years['Date']]
full_years['month'] = [d.strftime('%b') for d in full_years['Date']]
full_years['weekday'] = [d.strftime('%A') for d in full_years['Date']]

full_years_monthly = full_years.groupby(['year', 'month'])[['Volume']].mean()
full_years_monthly.reset_index(inplace=True)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

full_years_monthly['month'] = pd.Categorical(full_years_monthly['month'], categories=months, ordered=True)
full_years_monthly.sort_values(by=['year', 'month'], inplace=True)
display(full_years_monthly.head(3))
display(full_years_monthly.tail(3))


plt.figure(figsize=(16, 8), dpi=100)
for year in full_years['year'].unique():
    plt.plot(full_years_monthly.loc[full_years_monthly['year']==year, 'month'],
             full_years_monthly.loc[full_years_monthly['year']==year, 'Volume'], label=year)
    print(year)
plt.legend(loc='upper left')
plt.xlim(['Jan', 'Dec'])
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=80)
sns.boxplot(x='year', y='Volume', data=full_years, ax=axes[0]).set(
    xlabel='Rok',
    ylabel='Liczba'
)

axes[0].set_title('Box plot dla lat (trend)')
sns.boxplot(x='month', y='Volume', data=full_years, ax=axes[1]).set(
    xlabel='Miesiąc',
    ylabel='Liczba'
)

axes[1].set_title('Box plot dla miesiący (sezonowość)')
sns.boxplot(x='weekday', y='Volume', data=full_years, ax=axes[2]).set(
    xlabel='Dzień tygodnia',
    ylabel='Liczba'
)

axes[2].set_title('Box plot dla dni tygodnia (sezonowość)')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30)
plt.show()

holidays_Norway = holidays.NO()
def get_holiday(row):
    holidays = holidays_Norway.get(row['Date'])
    if holidays == None:
        return np.NaN

    # Søndag, czyli niedziela jest traktowana jako święto - usuwamy to święto
    elif holidays != 'Søndag':

        # Usunięcie niedzieli nakładającej się z innymi świętami
        if 'Søndag' in holidays:
            holidays = holidays.replace('Søndag', '')
            holidays = holidays.replace(',', '')
            holidays = holidays.lstrip().rstrip()
            return holidays
        else:
            return holidays
    else:
        return np.NaN
full_years['Holiday'] = full_years.apply(get_holiday, axis=1)
print('Unikalne święta:')
print(full_years['Holiday'].value_counts())

for unique_holiday in full_years['Holiday'].dropna().unique():

    # Filtrowanie miesięcy, w których to święto wystąpiło, usunięcie innych świąt w tym samym okresie
    month_holiday = full_years.loc[full_years['Holiday']==unique_holiday, 'month'].unique()
    selected_holiday = full_years.loc[((full_years['month'].isin(month_holiday))&\
                                       (full_years['Holiday'].isin([np.NaN, unique_holiday]))), :].copy()
    selected_holiday['Holiday'].fillna('Brak święta', inplace=True)

    # Wizualizacja wykresu plotowego
    plt.figure(figsize=(8, 1))
    sns.boxplot(x=selected_holiday['Holiday'], y=selected_holiday['Volume'],
                palette="Blues", order=['Brak święta', unique_holiday])
    plt.title(f'Święto: {unique_holiday}, Miesiące: {month_holiday}')
    plt.show()

full_years.loc[~full_years['Holiday'].isnull(), 'Holiday'] = 1
full_years.fillna(0, inplace=True)

plt.figure(figsize=(16, 7))
sns.boxplot(x=full_years['Holiday'], y=full_years['Volume'], palette="Blues")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=80)
pd.plotting.lag_plot(dataframe['Volume'], lag=1, ax=axes[0], alpha=0.3)
pd.plotting.lag_plot(dataframe['Volume'], lag=7, ax=axes[1], alpha=0.3)
pd.plotting.lag_plot(dataframe['Volume'], lag=365, ax=axes[2], alpha=0.3)
plt.show()

sns.pairplot(full_years[['Volume', 'Rain', 'Temp', 'weekday']], hue='weekday', kind='scatter', plot_kws={'alpha':0.25})
plt.show()

class AddMissingDates(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column
        pass
    def fit(self, X, y = None ):
        return self
    def transform(self, X, y = None ):
        X_transformed = X.copy()

        # Ustawienie kolumny z czasem jako indeksu
        X_transformed = X_transformed.set_index(self.date_column)
        X_transformed.index = pd.to_datetime(X_transformed.index)

        # Dodanie brakujących
        new_idx = pd.date_range(X_transformed.index.min(), X_transformed.index.max())
        X_transformed = X_transformed.reindex(new_idx)
        return X_transformed

steps = [
    ('add_missing_dates', AddMissingDates(date_column='Date'))
]
data_preparation_pipeline = Pipeline(steps = steps)
dataframe_prepared = data_preparation_pipeline.fit_transform(dataframe)
print(dataframe_prepared)

class FillMissings(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None ):
        return self

    def transform(self, X, y = None ):
        X_transformed = X.copy()

        # Uzupełnienie brakujących wartości dla liczby wypożyczonych rowerów
        X_transformed.loc[X_transformed['Volume'].isnull(), 'Volume'] = X_transformed['Volume'].shift(7)

        # Uzupełnienie brakujących wartości dla zmiennych pogodowych
        X_transformed.loc[X_transformed['Rain'].isnull(), 'Rain'] = X_transformed['Rain'].shift(1)
        X_transformed.loc[X_transformed['Temp'].isnull(), 'Temp'] = X_transformed['Temp'].shift(1)

        return X_transformed

steps = [
    ('add_missing_dates', AddMissingDates(date_column='Date')),
    ('fill_missing_values', FillMissings())
]
data_preparation_pipeline = Pipeline(steps = steps)
dataframe_prepared = data_preparation_pipeline.fit_transform(dataframe)
print(dataframe_prepared)

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit( self, X, y = None ):
        return self
    def transform( self, X, y = None ):
        def encode_time(X, col, max_val):
            X[col + '_sin'] = np.sin(2 * np.pi * X[col]/max_val)
            X[col + '_cos'] = np.cos(2 * np.pi * X[col]/max_val)
            X.drop(col, axis='columns', inplace=True)
            return X
        def get_holiday(row):
            holidays = holidays_Norway.get(row['Date'])
            if holidays == None:
                return 0

            # Søndag, czyli niedziela jest traktowana jako święto - usuwamy to święto
            elif holidays != 'Søndag':

                # Usunięcie niedzieli nakładającej się z innymi świętami
                if 'Søndag' in holidays:
                    holidays = holidays.replace('Søndag', '')
                    holidays = holidays.replace(',', '')
                    holidays = holidays.lstrip().rstrip()
                    return 1
                else:
                    return 1
            else:
                return 0
        X_transformed = X.copy()

        # Zmienne z przeszłości - możemy stosować wartości maksymalnie sprzed tygodnia
        # Dlatego najświeższą wartością z przeszłości dla wypożyczonych rowerów będzie wartość sprzed tygodnia
        X_transformed['Volume_lag_7W'] = X_transformed['Volume'].shift(7)

        # Opóźnione wartości
        for lag in range(1, 15):
            X_transformed[f'Volume_lag_{lag+7}W'] = X_transformed['Volume_lag_7W'].shift(lag)

        # Kroczące statystyki
        for window in [7, 14]:
            X_transformed[f'Volume_window_{window}_mean'] = X_transformed['Volume_lag_7W'].rolling(window=window).mean()
            X_transformed[f'Volume_window_{window}_std'] = X_transformed['Volume_lag_7W'].rolling(window=window).std()
            X_transformed[f'Volume_window_{window}_min'] = X_transformed['Volume_lag_7W'].rolling(window=window).min()
            X_transformed[f'Volume_window_{window}_max'] = X_transformed['Volume_lag_7W'].rolling(window=window).max()

        # Dzień w roku - sezonowość
        X_transformed['day_of_year'] = X_transformed.index.dayofyear
        X_transformed = encode_time(X_transformed, 'day_of_year', 366)

        # Dzień w tygodniu - sezonowość
        X_transformed['weekday'] = X_transformed.index.weekday
        X_transformed = encode_time(X_transformed, 'weekday', 7)

        # Rok - należy rozróżnić wartości z 2017 roku, wtedy średnio było mniej wypożyczeń
        X_transformed['year'] = X_transformed.index.year

        # Dodanie święta
        X_transformed['Date'] = X_transformed.index
        X_transformed['Holiday'] = X_transformed.apply(get_holiday, axis=1)
        X_transformed.drop(['Date'], axis=1, inplace=True)

        # Usunięcie wierszy z brakującymi danymi
        X_transformed.dropna(inplace=True)
        return X_transformed


steps = [
    ('add_missing_dates', AddMissingDates(date_column='Date')),
    ('fill_missing_values', FillMissings()),
    ('feature_engineering', FeatureEngineeringTransformer())
]

data_preparation_pipeline = Pipeline(steps = steps)
dataframe_prepared = data_preparation_pipeline.fit_transform(dataframe)
print(dataframe_prepared)

X = dataframe_prepared.drop(['Volume'], axis=1)
y = dataframe_prepared['Volume']

first_test_date = dt.datetime.strptime('2020-01-01', '%Y-%m-%d')
X_train, y_train = X[X.index<first_test_date].copy(), y[y.index<first_test_date].copy()
X_test, y_test = X[X.index>=first_test_date].copy(), y[y.index>=first_test_date].copy()

importances = RandomForestRegressor(n_estimators=1000, max_depth=20, n_jobs=-1).fit(X_train, y_train).feature_importances_
features = pd.concat([pd.DataFrame(X_train.columns, columns=['feat']),
                      pd.DataFrame(importances, columns=['importance'])
                     ], axis=1).sort_values(by='importance', ascending=False)
features = features[features['importance']>0.0075]
features.loc[features['feat'].str.contains('Volume_lag_'), 'Busket'] = 'lag'
features.loc[features['feat'].str.contains('mean'), 'Busket'] = 'mean'
features.loc[features['feat'].str.contains('std'), 'Busket'] = 'std'
features.loc[features['feat'].str.contains('min'), 'Busket'] = 'min'
features.loc[features['feat'].str.contains('max'), 'Busket'] = 'max'
display(features)
features = features[(~features.duplicated(subset='Busket', keep='first'))|(features['Busket'].isnull())]
features = list(features['feat'].values)
print(f'Liczba zmiennych: {len(features)}')
print(f'Zmienne: {features}')

X_train = X_train[features]
X_test = X_test[features]

prediction_baseline = y.copy()
prediction_baseline = np.round((prediction_baseline.shift(367) + prediction_baseline.shift(7)) / 2)
prediction_baseline.dropna(inplace=True)
prediction_baseline = prediction_baseline[prediction_baseline.index>=first_test_date]
print(prediction_baseline)

def plot_prediction_vs_true(yhat_list, yhat_names, ytest):
    plt.figure(figsize=(16, 8), dpi=100)
    plt.plot(ytest, color='black', label='True', linewidth=3)
    for yhat, yname in zip(yhat_list, yhat_names):
        plt.plot(yhat, label=f'Prediction - {yname}')
    plt.xlim([ytest.index.min(), ytest.index.max()])
    plt.legend(loc='upper left')
    plt.show()

plot_prediction_vs_true(yhat_list=[prediction_baseline],
                        yhat_names=['Baseline'],
                        ytest=y_test)


def Cost(y_true, y_pred):
    def apply_cost(x):
        if x < 0:
            x = np.abs(x) * 10
        return x

    y_pred = np.round(y_pred)
    cost = y_pred - y_true
    cost = cost.apply(apply_cost)
    return np.sum(cost)

def model_evaluation(yhat, ytest):
    def fit_scatter_plot(yhat, ytest):
        xmin = ytest.min()
        xmax = ytest.max()
        plt.scatter(x = yhat, y = ytest, alpha=0.25)
        x_line = np.linspace(xmin, xmax, 10)
        y_line = x_line
        plt.plot(x_line, y_line, 'r--')
        plt.xlabel('Predykcja')
        plt.ylabel('Wartość Prawdziwa')
        plt.title(f'Wykres predykcji względem wartości prawdziwych - Test set')
    def plot_of_residuals(yhat, ytest):
        errors = yhat - ytest
        plt.scatter(x = ytest, y = errors, alpha=0.25)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel('Wartość Prawdziwa')
        plt.ylabel('Reszta')
        plt.title(f'Wykres reszt - Test set')
    def hist_of_residuals(yhat, ytest):
        errors = yhat - ytest
        plt.hist(errors, bins = 100)
        plt.axvline(errors.mean(), color='k', linestyle='dashed', linewidth=1)
        plt.title(f'Histogram reszt - Test set')
    fig = plt.figure(figsize = (18, 6))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    fit_scatter_plot(yhat, ytest)
    ax = fig.add_subplot(1, 3, 2)
    plot_of_residuals(yhat, ytest)
    ax = fig.add_subplot(1, 3, 3)
    hist_of_residuals(yhat, ytest)
    plt.show()

    print(f'RMSE Test: {sqrt(mean_squared_error(ytest, yhat))}')
    print(f'Pomniejszony zysk: {Cost(ytest, yhat)}')

model_evaluation(yhat=prediction_baseline, ytest=y_test)

cost_scorer = make_scorer(Cost, greater_is_better=False)

tree = DecisionTreeRegressor(random_state=2022)
params = {'max_depth': [2, 3, 5, 7, 10],
          'min_samples_leaf': [2, 3, 5, 7, 10]}
tree_gridsearch = GridSearchCV(tree,
                               params,
                               scoring=cost_scorer,
                               cv=TimeSeriesSplit(n_splits=5).split(X_train),
                               verbose=10,
                               n_jobs=-1)
tree_gridsearch.fit(X_train, y_train)
print('\nNajlepsze hiperparametry:', tree_gridsearch.best_params_)
tree_model = tree_gridsearch.best_estimator_

prediction_tree = tree_model.predict(X_test)
prediction_tree = pd.Series(prediction_tree, index=y_test.index)


plot_prediction_vs_true(yhat_list=[prediction_baseline, prediction_tree],
                        yhat_names=['Baseline', 'Tree'],
                        ytest=y_test)

model_evaluation(yhat=prediction_tree, ytest=y_test)

forest = RandomForestRegressor(n_estimators=1000, random_state=2022)
params = {'max_depth': [2, 3, 5, 10],
          'min_samples_leaf': [3, 5, 10, 15]}
forest_gridsearch = GridSearchCV(forest,
                                 params,
                                 scoring=cost_scorer,
                                 cv=TimeSeriesSplit(n_splits=5).split(X_train),
                                 verbose=10,
                                 n_jobs=-1)
forest_gridsearch.fit(X_train, y_train)
print('\nNajlepsze hiperparametry:', forest_gridsearch.best_params_)
forest_model = forest_gridsearch.best_estimator_

prediction_forest = forest_model.predict(X_test)
prediction_forest = pd.Series(prediction_forest, index=y_test.index)

plot_prediction_vs_true(yhat_list=[prediction_baseline, prediction_tree, prediction_forest],
                        yhat_names=['Baseline', 'Tree', 'Forest'],
                        ytest=y_test)

model_evaluation(yhat=prediction_forest, ytest=y_test)

explainer = shap.TreeExplainer(forest_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

joblib.dump(forest_model, 'forest_model.pkl')