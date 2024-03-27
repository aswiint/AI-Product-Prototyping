import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, shapiro
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

import warnings
warnings.filterwarnings("ignore")

# Reading the Dataset
path = "spain_energy_market.csv"
data = pd.read_csv(path, sep=",", parse_dates=["datetime"])
data = data[data["name"] == "Demanda programada PBF total"]
data["date"] = pd.to_datetime(data["datetime"], format="mixed")
data["date"] = data["date"].dt.date
data.set_index("date", inplace=True)
data = data[["value"]]
data = data.asfreq("D")
data = data.rename(columns={"value": "energy"})
data.info()
print("\nDataFrame Head: \n", data[:5])
data.plot(title="Energy Demand")
plt.ylabel("MWh")
plt.show()
print("\nNumber of Records: ", len(pd.date_range(start="2014-01-01", end="2018-12-31")))

data["year"] = data.index.year
data["qtr"] = data.index.quarter
data["mon"] = data.index.month
data["ix"] = range(0, len(data))
data[["movave_7", "movstd_7"]] = data.energy.rolling(7).agg([np.mean, np.std])
data[["movave_30", "movstd_30"]] = data.energy.rolling(30).agg([np.mean, np.std])
data[["movave_90", "movstd_90"]] = data.energy.rolling(90).agg([np.mean, np.std])
data[["movave_365", "movstd_365"]] = data.energy.rolling(365).agg([np.mean, np.std])
data[["energy", "movave_7"]].plot(title="Daily Energy Demand in Spain (MWh)")
plt.ylabel("(MWh)")
plt.show()


# Exploratory Data Analysis (EDA)

# Target Analysis (Normality)
mean = np.mean(data.energy.values)
std = np.std(data.energy.values)
skew = skew(data.energy.values)
ex_kurt = kurtosis(data.energy)
print("Skewness: {} \nKurtosis: {}".format(skew, ex_kurt+3))


def shapiro_test(data, alpha=0.05):
    stat, pval = shapiro(data)
    print("H0: Data was drawn from a Normal Distribution")
    if pval < alpha:
        print("pval {} is lower than significance level: {}, therefore null hypothesis is rejected".format
              (pval, alpha))
    else:
        print("pval {} is higher than significance level: {}, therefore null hypothesis cannot be rejected".format
              (pval, alpha))


shapiro_test(data.energy, alpha=0.05)

sns.distplot(data.energy)
plt.title("Target Analysis")
plt.xticks(rotation=45)
plt.xlabel("(MWh)")
plt.axvline(x=mean, color='r', linestyle='-', label="mu: {0:.2f}%".format(mean))
plt.axvline(x=mean+2*std, color='orange', linestyle='-')
plt.axvline(x=mean-2*std, color='orange', linestyle='-')
plt.show()

# Inserting the rolling quantiles to the monthly returns
data_rolling = data.energy.rolling(window=90)
data['q10'] = data_rolling.quantile(0.1).to_frame("q10")
data['q50'] = data_rolling.quantile(0.5).to_frame("q50")
data['q90'] = data_rolling.quantile(0.9).to_frame("q90")
data[["q10", "q50", "q90"]].plot(title="Volatility Analysis: 90-rolling percentiles")
plt.ylabel("(MWh)")
plt.show()

# Coefficient of Variation
data.groupby("qtr")["energy"].std().divide(data.groupby("qtr")["energy"].mean()).plot(kind="bar")
plt.title("Coefficient of Variation (CV) by qtr")
plt.show()
data.groupby("mon")["energy"].std().divide(data.groupby("mon")["energy"].mean()).plot(kind="bar")
plt.title("Coefficient of Variation (CV) by month")
plt.show()

# Heteroscedasticity Analysis
data[["movstd_30", "movstd_365"]].plot(title="Heteroscedasticity analysis")
plt.ylabel("(MWh)")
plt.show()

# Seasonal Analysis
data[["movave_30", "movave_90"]].plot(title="Seasonal Analysis: Moving Averages")
plt.ylabel("(MWh)")
plt.show()
sns.boxplot(data=data, x="qtr", y="energy")
plt.title("Seasonality analysis: Distribution over quarters")
plt.ylabel("(MWh)")
plt.show()

# Seasonal Patterns
data_mon = data. energy.resample("M").agg(sum).to_frame("energy")
data_mon["ix"] = range(0, len(data_mon))
print("\nSeasonal Patterns in the quarter: \n", data_mon[:5])

# Trend Analysis: Regression
sns.regplot(data=data_mon, x="ix", y="energy")
plt.title("Trend analysis: Regression")
plt.ylabel("(MWh)")
plt.xlabel("")
plt.show()

# Trend Analysis: Annual Box-Plot Distribution
sns.boxplot(data=data["2014":"2017"], x="year", y="energy")
plt.title("Trend Analysis: Annual Box-plot Distribution")
plt.ylabel("(MWh)")
plt.show()


# Feature Engineering

data["target"] = data.energy.add(-mean).div(std)
sns.distplot(data["target"])
plt.show()
features = []
corr_features = []
targets = []
tau = 30

# Forecasting Periods
for t in range(1, tau + 1):
    data["target_t" + str(t)] = data.target.shift(-t)
    targets.append("target_t" + str(t))

for t in range(1, 31):
    data["feat_ar" + str(t)] = data.target.shift(t)
    # data["feat_ar" + str(t) + "_lag1y"] = data.target.shift(350)
    features.append("feat_ar" + str(t))
    # corr_features.append("feat_ar" + str(t))
    # features.append("feat_ar" + str(t) + "_lag1y")

for t in [7, 14, 30]:
    data[["feat_movave" + str(t), "feat_movstd" + str(t), "feat_movmin" + str(t), "feat_movmax" + str(t)]] = (
        data.energy.rolling(t).agg([np.mean, np.std, np.max, np.min]))
    features.append("feat_movave" + str(t))
    # corr_features.append("feat_movave" + str(t))
    features.append("feat_movstd" + str(t))
    features.append("feat_movmin" + str(t))
    features.append("feat_movmax" + str(t))

months = pd.get_dummies(data.mon, prefix="mon", drop_first=True)
months.index = data.index
data = pd.concat([data, months], axis=1)
features = features + months.columns.values.tolist()
corr_features = ["feat_ar1", "feat_ar2", "feat_ar3", "feat_ar4", "feat_ar5", "feat_ar6", "feat_ar7", "feat_movave7",
                 "feat_movave14", "feat_movave30"]

# Calculating Correlation matrix
corr = data[["target_t1"] + corr_features].corr()
top5_mostCorrFeats = corr["target_t1"].apply(abs).sort_values(ascending=False).index.values[:6]

# Correlation Matrix HeatMap
sns.heatmap(corr, annot=True)
plt.title("Pearson Correlation with 1 period target")
plt.yticks(rotation=0)
plt.xticks(rotation=90)  # fix tick label directions
plt.tight_layout()  # fits plot area to the plot, "tightly"
plt.show()

# Correlation Matrix Scatter Plot
sns.pairplot(data=data[top5_mostCorrFeats].dropna(), kind="reg")
plt.title("Most important features Matrix Scatter Plot")
plt.show()


# Model Building

data_feateng = data[features + targets].dropna()
nobs = len(data_feateng)
print("Number of observations: ", nobs)

# Splitting the Data
X_train = data_feateng.loc["2014":"2017"][features]
y_train = data_feateng.loc["2014":"2017"][targets]
X_test = data_feateng.loc["2018"][features]
y_test = data_feateng.loc["2018"][targets]
n, k = X_train.shape
print("Total number of observations: ", nobs)
print("Train: {}{}, \nTest: {}{}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
plt.plot(y_train.index, y_train.target_t1.values, label="train")
plt.plot(y_test.index, y_test.target_t1.values, label="test")
plt.title("Train/Test split")
plt.xticks(rotation=45)
plt.show()


# Baseline Model: Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train["target_t1"])
p_train = reg.predict(X_train)
p_test = reg.predict(X_test)
RMSE_train = np.sqrt(mean_squared_error(y_train["target_t1"], p_train))
RMSE_test = np.sqrt(mean_squared_error(y_test["target_t1"], p_test))
print("Train RMSE: {}\nTest RMSE: {}".format(RMSE_train, RMSE_test))

# Training Random Forest with Time Series Split

splits = TimeSeriesSplit(n_splits=3, max_train_size=365 * 2)
for train_index, val_index in splits.split(X_train):
    print("TRAIN:", len(train_index), "TEST:", len(val_index))
    y_train["target_t1"][train_index].plot()
    y_train["target_t1"][val_index].plot()
    plt.show()
