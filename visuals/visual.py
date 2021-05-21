import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing, SimpleExpSmoothing
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
pd.options.mode.chained_assignment = None

# CSV file may be in different location
df = pd.read_csv("nba.csv")
df.head()

#Preprocess data 
df = df.dropna()
# Normalize Data: x - xmin / xmax - xmin
arr = ['team_id', 'w_pct', 'min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a',
        'fg3_pct', 'ftm', 'fta', 'ft_pct', 'a_team_id']
for field in arr:
    scaler = MinMaxScaler() 
    df[field] = scaler.fit_transform(df[[field]])
df = df.replace('W', 1)
df = df.replace('L', 0)
df = df.replace('t', 1)
df = df.replace('f', 0)
df['wl'] = df['wl'].astype('int') 
df['is_home'] = df['is_home'].astype('int') 

newdf = df[["game_id", "game_date", "is_home", "w_pct"]]
for index, row in newdf.iterrows():
  newdf.loc[index, "real_date"] = datetime.strptime(newdf.loc[index, "game_date"],'%Y-%m-%d')

newdf = newdf[newdf['real_date'].dt.year == 2008 ]
newdf = newdf[newdf['real_date'].dt.month == 11 ]

newdfishome = newdf[newdf['is_home'] == 1]
newdfisaway = newdf[newdf['is_home'] == 0]

newdfishome = newdfishome.drop_duplicates(subset="real_date")
newdfisaway = newdfisaway.drop_duplicates(subset="real_date")

newdfishome["game_date"] = pd.to_datetime(newdfishome["game_date"])
newdfisaway["game_date"] = pd.to_datetime(newdfisaway["game_date"])

newdfishome = newdfishome.sort_values(by=['game_date'])
newdfisaway = newdfisaway.sort_values(by=['game_date'])

plt.figure(figsize=(20,5))
plt.title("Win_Percentages Of Teams Playing On The Road Vs Teams Playing At Home")
plt.xlabel=("Date")
plt.ylabel=("Win Percentages")
plt.plot(newdfisaway["game_date"], newdfisaway["w_pct"], label="Away")
plt.plot(newdfishome["game_date"], newdfishome["w_pct"], label="Home")

plt.legend(loc ="lower right")

plt.figure(figsize=(10,5))
plt.title("Win_Percentages Of Teams Playing At Home")
newdfishome = newdfishome.head(10)
plt.plot(newdfishome["game_date"], newdfishome["w_pct"])

plt.figure(figsize=(10,5))
plt.title("Win_Percentages Of Teams Playing On The Road")
newdfisaway = newdfisaway.head(10)
plt.plot(newdfisaway["game_date"], newdfisaway["w_pct"])

# ROC Curve
filterdf = df[["wl", 'team_id', 'w_pct', 'a_team_id', 'is_home']]
  
filterdf.head()
y = filterdf["wl"]
X = filterdf.loc[:, filterdf.columns != 'wl']
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=100)

model = sm.Logit(y_train,X_train).fit()
model.summary()

predictions = model.predict(sm.add_constant(X_test))
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

matrix = confusion_matrix(y_test, predictions)
print(matrix)

classThreshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
FPRarr = []
TPRarr = []

for i in range(0, len(classThreshold)):
  currPredictions = model.predict(sm.add_constant(X_test))
  currPredictions[currPredictions > classThreshold[i]] = 1
  currPredictions[currPredictions <= classThreshold[i]] = 0
  
  matrix = confusion_matrix(y_test, currPredictions)

  TPRarr.append(matrix[1][1] / (matrix[1][1] + matrix[1][0]))
  FPRarr.append(matrix[0][1] / (matrix[0][1] + matrix[0][0]))
  
plt.title("TPR vs FPR")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(FPRarr, TPRarr, color="orange")