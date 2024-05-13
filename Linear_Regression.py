import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.impute import SimpleImputer

def read_and_combine_json_files(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for station, hourly_data in data.items():
                    for hour, values in hourly_data.items():
                        values['Station'] = station
                        values['Hour'] = hour
                        all_data.append(values)
        except UnicodeDecodeError as e:
            print(f"Error reading {file_name}: {e}")
    return pd.DataFrame(all_data)

def preprocess_data(df):
    df = df[['PM2.5', 'PM10', 'AQI']].replace('NaN', np.nan).astype(float)
    imputer = SimpleImputer(strategy='mean')
    df[['PM2.5', 'PM10', 'AQI']] = imputer.fit_transform(df[['PM2.5', 'PM10', 'AQI']])
    return df


def linear_regression_analysis(df):
    model = ols('AQI ~ Q("PM2.5") + PM10', data=df).fit()
    print(model.summary())

    fig = px.scatter_3d(df, x='PM2.5', y='PM10', z='AQI', title='3D Scatter plot')
    fig.show()

    return model

def calculate_anova(model):
    anova_results = sm.stats.anova_lm(model, typ=2)
    print("ANOVA results:\n", anova_results)

def calculate_pearson(df):
    pearson_pm25_aqi = df['PM2.5'].corr(df['AQI'])
    pearson_pm10_aqi = df['PM10'].corr(df['AQI'])
    print(f"Pearson correlation coefficient (PM2.5 and AQI): {pearson_pm25_aqi}")
    print(f"Pearson correlation coefficient (PM10 and AQI): {pearson_pm10_aqi}")


def main():
    folder_path = 'D:\\统计建模2024\\北京（18年到23年）\\111_json'
    df = read_and_combine_json_files(folder_path)
    df = preprocess_data(df)

    print("Descriptive statistics after imputation:\n", df.describe())
    model = linear_regression_analysis(df)
    calculate_anova(model)
    calculate_pearson(df)

if __name__ == "__main__":
    main()
