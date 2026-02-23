import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from feature_selector import FeatureSelector


def export_corr_matrix(corr_matrix):
    # Вывод в консоль
    print("Матрица корреляций:")
    print(corr_matrix)

    # Сохранение в CSV-файл
    corr_matrix.to_csv('correlation_matrix.csv')
    print("Матрица сохранена в correlation_matrix.csv")


def visualize_corr_matrix(corr_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')

    # Настройка меток осей
    plt.colorbar(label='Корреляция')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Матрица корреляций')

    # Добавление значений в ячейки
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                     ha='center', va='center', color='black', fontsize=10)

    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def get_max_corr(corr_matrix):
    mask = np.eye(len(corr_matrix), dtype=bool)
    masked_corr = corr_matrix.mask(mask)

    max_corr_value = masked_corr.abs().max().max()
    feature1, feature2 = masked_corr.abs().stack().idxmax()

    print(f"Признаки с наибольшей корреляцией ({max_corr_value:.3f}):")
    print(f"1. {feature1}")
    print(f"2. {feature2}")
    print(f"Значение корреляции: {corr_matrix.loc[feature1, feature2]:.3f}")

    return feature1, feature2


def raise_feature_selector(df):
    selector = FeatureSelector(list(df.columns))
    return selector.run()


def visualize_selected_features_corr(df, feature1, feature2):
    # Scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df[feature1], df[feature2], alpha=0.7, s=60, c='steelblue')
    plt.xlabel(feature1, fontsize=12)
    plt.ylabel(feature2, fontsize=12)
    corr = df[feature1].corr(df[feature2])
    plt.title(f'{feature1} vs {feature2}\nКорреляция: {corr:.3f}', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Линия тренда
    z = np.polyfit(df[feature1], df[feature2], 1)
    p = np.poly1d(z)
    plt.plot(df[feature1], p(df[feature1]), "r--", linewidth=3, alpha=0.8)

    plt.tight_layout()
    plt.savefig('interactive_scatter_arrows.png', dpi=300, bbox_inches='tight')
    plt.show()


def get_model_data(df, feature1, feature2):
    # feature1, feature2 — выбранные признаки (X)
    # Предполагаем, что есть целевая переменная (y) или используем один признак как target
    x = df[[feature1]]  # Признаки
    y = df[feature2]  # Целевая переменная (второй признак)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.2, random_state=42, shuffle=True
    )

    print(f"Размер выборок:")
    print(f"Обучающая: {x_train.shape[0]} записей ({x_train.shape[0] / len(df) * 100:.1f}%)")
    print(f"Тестовая:   {x_test.shape[0]} записей ({x_test.shape[0] / len(df) * 100:.1f}%)")

    # Обучение модели
    model = LinearRegression()
    model.fit(x_train, y_train)

    print(f"\nМодель обучена на 20% данных!")
    print(f"Коэффициенты: {model.coef_}")
    print(f"Пересечение:  {model.intercept_:.3f}")

    return model, x_train, x_test, y_train, y_test


def visualize_model_results(model, feature1, x_train, x_test, y_train, y_test):
    # Предсказания
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Метрики качества
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\nКачество модели:")
    print(f"Обучающая выборка (20%): R²={train_r2:.3f}, RMSE={train_rmse:.3f}")
    print(f"Тестовая выборка (80%):  R²={test_r2:.3f}, RMSE={test_rmse:.3f}")

    # Визуализация предсказаний
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(x_train[feature1], y_train, alpha=0.6, label='Факт', color='blue')
    plt.scatter(x_train[feature1], y_train_pred, alpha=0.6, label='Предсказание', color='red')
    plt.xlabel(feature1)
    plt.ylabel('Значения')
    plt.title('Обучающая выборка (20%)')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(x_test[feature1], y_test, alpha=0.6, label='Факт', color='blue')
    plt.scatter(x_test[feature1], y_test_pred, alpha=0.6, label='Предсказание', color='red')
    plt.xlabel(feature1)
    plt.ylabel('Значения')
    plt.title('Тестовая выборка (80%)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'R² тест = {test_r2:.3f}')

    plt.tight_layout()
    plt.savefig('model_training_20percent.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_slop_and_intercept(feature1, feature2, x_train, y_train):
    x_train_single = x_train[[feature1]]
    model_single = LinearRegression()
    model_single.fit(x_train_single, y_train)

    # Коэффициенты регрессии для обучающей выборки (20%)
    # Коэффициент наклона (β₁)
    slope = model_single.coef_[0]
    # Точка пересечения (β₀)
    intercept = model_single.intercept_

    print("Коэффициенты линейной регрессии")
    print("=" * 50)
    print(f"Уравнение: y = {slope:.4f} × {feature1} + {intercept:.4f}")
    print(f"Коэффициент наклона (β₁): {slope:.4f}")
    print(f"Точка пересечения (β₀):  {intercept:.4f}")
    print(f"R² на обучающей выборке:  {model_single.score(x_train_single, y_train):.4f}")

    # Прямая проверка формулами
    corr_coef = np.corrcoef(x_train[feature1], y_train)[0, 1]
    std_x = x_train[feature1].std()
    std_y = y_train.std()
    manual_slope = corr_coef * (std_y / std_x)
    manual_intercept = y_train.mean() - manual_slope * x_train[feature1].mean()

    print("\nПРОВЕРКА ФОРМУЛАМИ:")
    print(f"Ручной расчет наклона: {manual_slope:.4f}")
    print(f"Ручной расчет пересечения: {manual_intercept:.4f}")

    plt.figure(figsize=(12, 8))

    # Обучающая выборка (20%)
    plt.scatter(
        x_train[feature1],
        y_train,
        alpha=0.7,
        s=100,
        color='limegreen',
        label=f'Обучающая выборка (n={len(x_train)})'
    )

    # Линия регрессии
    x_range = np.linspace(x_train[feature1].min(), x_train[feature1].max(), 100)
    y_line = slope * x_range + intercept
    plt.plot(
        x_range,
        y_line,
        'red',
        linewidth=3,
        label=f'y = {slope:.2f}x + {intercept:.2f}'
    )

    plt.xlabel(f'{feature1}', fontsize=14)
    plt.ylabel(f'{feature2}', fontsize=14)
    plt.title('Линейная регрессия на обучающей выборке (20%)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_coefficients_train.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_mean_errors(model, x_train, y_train):
    # Предсказания на обучающей выборке
    y_train_pred = model.predict(x_train)

    # РАСЧЕТ ОШИБОК
    mse_train = mean_squared_error(y_train, y_train_pred)  # MSE
    rmse_train = np.sqrt(mse_train)  # RMSE
    mae_train = mean_absolute_error(y_train, y_train_pred)  # MAE

    print("ОШИБКИ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ (20%)")
    print("=" * 50)
    print(f"Среднеквадратичная ошибка (MSE):  {mse_train:.4f}")
    print(f"Корень из MSE (RMSE):             {rmse_train:.4f}")
    print(f"Средняя абсолютная ошибка (MAE):   {mae_train:.4f}")
    print(f"Относительная MAE (% от среднего): {(mae_train / y_train.mean() * 100):.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Распределение ошибок
    errors = y_train - y_train_pred
    axes[0, 0].hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title(f'Распределение ошибок\nMSE={mse_train:.2f}, MAE={mae_train:.2f}')
    axes[0, 0].set_xlabel('Ошибка (факт - предсказание)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Фактические vs Предсказанные
    axes[0, 1].scatter(y_train, y_train_pred, alpha=0.6, s=60)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_xlabel('Фактические значения')
    axes[0, 1].set_ylabel('Предсказанные значения')
    axes[0, 1].set_title('Факт vs Предсказание')

    # 3. Остатки vs Предсказания
    axes[1, 0].scatter(y_train_pred, errors, alpha=0.6)
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Предсказанные значения')
    axes[1, 0].set_ylabel('Остатки')
    axes[1, 0].set_title('Остатки (идеально ~ случайный шум)')

    # 4. Q-Q plot остатков
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q plot остатков')

    plt.tight_layout()
    plt.savefig('regression_errors_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_comparison_table_real_and_predicted(model, x_test, y_test):
    # Предсказания на тестовой выборке
    y_test_pred = model.predict(x_test)

    # Создание таблицы сравнения
    results_df = pd.DataFrame({
        'Реальное (Y)': y_test.reset_index(drop=True),
        'Предсказанное': y_test_pred,
        'Ошибка (абс)': np.abs(y_test.values - y_test_pred),
        'Ошибка (%)': np.abs(y_test.values - y_test_pred) / y_test.values * 100
    })

    # Вывод первых 20 наблюдений
    print("РЕАЛЬНЫЕ vs ПРЕДСКАЗАННЫЕ ЗНАЧЕНИЯ (тестовая выборка)")
    print("=" * 80)
    print(results_df.head(20).round(2))


def predict(model, feature1, feature2):
    # Сбор данных от пользователя
    value = input("Введите значение: ").replace(',', '.')
    input_df = [[float(value)]]

    prediction = model.predict(input_df)[0]

    # Вывод результата
    print(f"ПРЕДСКАЗАНИЕ:")
    print(f"Входные данные для {feature1}: {value}")
    print(f"Результат для {feature2}: {prediction:.4f}")

    return prediction


def main():
    df = pd.read_excel('real_estate_valuation_data_set.xlsx')
    corr_matrix = df.corr()

    # Нормализация данных
    scaler = StandardScaler()
    df = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )

    # Экспортировать данные матрицы корреляций
    # export_corr_matrix(corr_matrix)

    # Визуализировать данные матрицы корреляций
    # visualize_corr_matrix(corr_matrix)

    # Получить максимально коррелирующие признаки
    feature1, feature2 = get_max_corr(corr_matrix)

    # Дать пользователю выбрать признаки
    # feature1, feature2 = raise_feature_selector(df)

    # Визуализировать данные выбранных признаков
    # visualize_selected_features_corr(df, feature1, feature2)

    # Получить обученную модель
    model, x_train, x_test, y_train, y_test = get_model_data(df, feature1, feature2)

    # Визуализировать результат работы модели
    # visualize_model_results(model, feature1, x_train, x_test, y_train, y_test)

    # Визуализировать наклон и точку пересечения для обучающей выборки
    visualize_slop_and_intercept(feature1, feature2, x_train, y_train)

    # Визуализировать среднеквадратичную и среднюю абсолютную ошибки
    # visualize_mean_errors(model, x_train, y_train)

    # Вывести результаты реальных значений зависимой переменой по данным выборки и их предсказаний
    # print_comparison_table_real_and_predicted(model, x_test, y_test)

    # Запросить у пользователя значение и сделать предсказание значения зависимой переменной
    predict(model, feature1, feature2)


if __name__ == '__main__':
    main()
