import os
import json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch import nn
from transformers import BertModel, BertConfig
def read_and_preprocess_data(directory, selected_sites):
    all_data = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                for site in selected_sites:
                    site_data = data.get(site)
                    if site_data:
                        for hour, hour_data in site_data.items():

                            pm25 = hour_data.get('PM2.5', np.nan)
                            pm10 = hour_data.get('PM10', np.nan)
                            aqi = hour_data.get('AQI', np.nan)

                            date_str = filename.split('_')[-1].split('.')[0]
                            time_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {hour}:00:00"
                            record = {
                                'site': site,
                                'time': time_str,
                                'PM2.5': pm25,
                                'PM10': pm10,
                                'AQI': aqi
                            }
                            all_data.append(record)
    df = pd.DataFrame(all_data)

    correlation_matrix_original = df[['PM2.5', 'PM10', 'AQI']].corr()
    print(correlation_matrix_original)

    imputer = SimpleImputer(strategy='mean')
    df[['PM2.5', 'PM10', 'AQI']] = imputer.fit_transform(df[['PM2.5', 'PM10', 'AQI']])

    return df

selected_sites = [
    '东城东四', '东城天坛', '西城官园', '西城万寿西宫', '朝阳奥体中心',
    '朝阳农展馆', '海淀万柳', '海淀四季青', '丰台小屯', '丰台云岗', '石景山古城'
]

data_directory = 'D:\\统计建模2024\\北京（18年到23年）\\111_json'
df = read_and_preprocess_data(data_directory, selected_sites)
print(df.head())

class AirQualityTransformer(nn.Module):
    def __init__(self):
        super(AirQualityTransformer, self).__init__()
        self.config = BertConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=256
        )
        self.transformer = BertModel(self.config)
        self.regressor = nn.Linear(self.config.hidden_size, 3)  # 预测PM2.5, PM10, AQI

    def forward(self, input_ids, attention_mask=None):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        prediction = self.regressor(transformer_output.pooler_output)
        return prediction

model = AirQualityTransformer()
class AirQualityDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

features = df[['PM2.5', 'PM10', 'AQI']].values
targets = df[['PM2.5', 'PM10', 'AQI']].values

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

train_dataset = AirQualityDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = AirQualityDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class AirQualityModel(nn.Module):
    def __init__(self):
        super(AirQualityModel, self).__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x):
        return self.fc(x)

model = AirQualityModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_values = []
r2_scores = []

num_epochs = 530
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_values.append(avg_loss)

    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_predictions = model(batch_features)
            all_targets.extend(batch_targets.numpy())
            all_predictions.extend(batch_predictions.numpy())
    r2 = r2_score(all_targets, all_predictions)
    r2_scores.append(r2)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, R^2: {r2}')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_values, label='Training Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('D:\\统计建模2024\\图片大全\\loss.png')

plt.subplot(1, 2, 2)
plt.plot(r2_scores, label='R^2 Score', color='orange')
plt.title('R^2 Score During Training')
plt.xlabel('Epoch')
plt.ylabel('R^2 Score')
plt.legend()
plt.grid(True)
plt.savefig('D:\\统计建模2024\\图片大全\\accuracy.png')

plt.tight_layout()
plt.show()

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        batch_predictions = model(batch_features)
        predictions.extend(batch_predictions.numpy())
        actuals.extend(batch_targets.numpy())

mse = mean_squared_error(actuals, predictions)
print(f'Mean Squared Error: {mse}')

torch.save(model.state_dict(), 'air_quality_model.pth')

def read_test_data(directory, selected_sites):
    test_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for site in selected_sites:
                    site_data = data.get(site)
                    if site_data:
                        for hour, hour_data in site_data.items():
                            date_str = filename.split('_')[-1].split('.')[0]
                            time_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {hour}:00:00"
                            record = {
                                'site': site,
                                'time': time_str,
                                'PM2.5': hour_data.get('PM2.5', np.nan),
                                'PM10': hour_data.get('PM10', np.nan),
                                'AQI': hour_data.get('AQI', np.nan)
                            }
                            test_data.append(record)
    df_test = pd.DataFrame(test_data)
    return df_test

test_directory = 'D:\\统计建模2024\\北京（18年到23年）\\111_test'
df_test = read_test_data(test_directory, selected_sites)


model.load_state_dict(torch.load('air_quality_model.pth'))

features_test = scaler.transform(df_test[['PM2.5', 'PM10', 'AQI']].values)
test_dataset = AirQualityDataset(torch.tensor(features_test, dtype=torch.float32), torch.zeros(len(features_test), 3))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
predictions = []
with torch.no_grad():
    for batch_features, _ in test_loader:
        batch_predictions = model(batch_features)
        predictions.extend(batch_predictions.numpy())


df_test['PM2.5_predicted'] = [pred[0] for pred in predictions]
df_test['PM10_predicted'] = [pred[1] for pred in predictions]
df_test['AQI_predicted'] = [pred[2] for pred in predictions]

# 输出预测结果
print(df_test[['site', 'time', 'PM2.5_predicted', 'PM10_predicted', 'AQI_predicted']].head())


df_test['PM2.5'] = df_test['PM2.5'].fillna(df_test['PM2.5'].mean())
df_test['PM2.5_predicted'] = df_test['PM2.5_predicted'].fillna(df_test['PM2.5_predicted'].mean())
df_test['PM10'] = df_test['PM10'].fillna(df_test['PM10'].mean())
df_test['PM10_predicted'] = df_test['PM10_predicted'].fillna(df_test['PM10_predicted'].mean())
df_test['AQI'] = df_test['AQI'].fillna(df_test['AQI'].mean())
df_test['AQI_predicted'] = df_test['AQI_predicted'].fillna(df_test['AQI_predicted'].mean())

mse_pm25 = mean_squared_error(df_test['PM2.5'], df_test['PM2.5_predicted'])
mse_pm10 = mean_squared_error(df_test['PM10'], df_test['PM10_predicted'])
mse_aqi = mean_squared_error(df_test['AQI'], df_test['AQI_predicted'])

print(f'MSE for PM2.5: {mse_pm25}')
print(f'MSE for PM10: {mse_pm10}')
print(f'MSE for AQI: {mse_aqi}')


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


save_path = 'D:\\统计建模2024\\图片大全'
if not os.path.exists(save_path):
    os.makedirs(save_path)

image_number = 1

for site in selected_sites:
    site_data = df_test[df_test['site'] == site]
    site_data = site_data.sort_values(by='time')
    site_data['hour'] = pd.to_datetime(site_data['time']).dt.hour

    plt.figure(figsize=(15, 5))
    plt.figure(figsize=(15, 5))
    plt.scatter(site_data['hour'], site_data['PM2.5'], label='PM2.5 Real', color='blue', marker='o')
    plt.scatter(site_data['hour'], site_data['PM10'], label='PM10 Real', color='green', marker='o')
    plt.scatter(site_data['hour'], site_data['AQI'], label='AQI Real', color='red', marker='o')

    plt.scatter(site_data['hour'], site_data['PM2.5_predicted'], label='PM2.5 Predicted', color='skyblue', linestyle='--',
             marker='x')
    plt.scatter(site_data['hour'], site_data['PM10_predicted'], label='PM10 Predicted', color='lightgreen', linestyle='--',
             marker='x')
    plt.scatter(site_data['hour'], site_data['AQI_predicted'], label='AQI Predicted', color='salmon', linestyle='--',
             marker='x')

    plt.title(f'Air Quality Predictions for {site}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_path, f'Transformer_{image_number:02d}.png'))

    image_number += 1

    plt.close()

