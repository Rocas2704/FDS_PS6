# Problem Set: Multi-Model Financial Prediction and Risk Analysis
# Student Exercise Manual
# Please complete the code

# -------------------------------
# Import necessary libraries
# -------------------------------
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import tensorflow as tf  # Import TensorFlow for building neural network models
import seaborn as sns  # Import Seaborn for statistical plotting
import matplotlib.pyplot as plt  # Import Matplotlib for creating plots
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting datasets
from sklearn.metrics import mean_squared_error, r2_score  # Import metrics for evaluating models
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet  # Import linear regression models
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.cross_decomposition import PLSRegression  # Import PLSRegression for partial least squares regression
from sklearn.preprocessing import SplineTransformer  # Import SplineTransformer for spline feature transformation
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # Import ensemble regression models

plt.style.use('seaborn-v0_8-whitegrid')  # Set the plotting style to Seaborn whitegrid
plt.rcParams['figure.figsize'] = (12, 8)  # Set default figure size for plots to 12x8 inches

# -------------------------------
# Part 1: Data Generation
# -------------------------------
n_stocks = 10  # Define the number of stocks
n_months = 24  # Define the number of months (time periods)
n_characteristics = 5  # Define the number of stock-specific characteristics
n_macro_factors = 3  # Define the number of macroeconomic factors

np.random.seed(42)  # Set the random seed for reproducibility
stock_characteristics = np.random.rand(n_stocks, n_months, n_characteristics)  # Generate random stock characteristic data
macro_factors = np.random.rand(n_months, n_macro_factors)  # Generate random macroeconomic factor data


# -------------------------------
# Part 2: Train-Test Split
# -------------------------------
# Initialize matrix zi_t
zi_t = np.zeros((n_stocks, n_months, n_characteristics * n_macro_factors))

for t in range(n_months):
    for i in range(n_stocks):
        interaction = np.outer(stock_characteristics[i, t, :], macro_factors[t, :]).flatten()
        zi_t[i, t, :] = interaction

# Generate true parameter theta and r_i_t+1

theta = np.random.rand(n_characteristics * n_macro_factors)

# Initialize return matrices
ri_t_plus_1 = np.zeros((n_stocks, n_months))

for t in range(n_months):
    for i in range(n_stocks):
        ri_t_plus_1[i, t] = zi_t[i, t, :].dot(theta) + np.random.normal(0, 0.1)

# Flatten zi_t: (n_stocks * n_months, n_features)
zi_t_flattened = zi_t.reshape(n_stocks * n_months, -1)

# Flatten ri_t_plus_1: (n_stocks * n_months,)
ri_t_flattened = ri_t_plus_1.flatten()

zi_t_df = pd.DataFrame(
    zi_t_flattened,
    columns=[f"z_{k+1}" for k in range(zi_t_flattened.shape[1])]
)

zi_t_df["Stock_ID"] = np.repeat(range(1, n_stocks + 1), n_months)
zi_t_df["Month"] = np.tile(range(1, n_months + 1), n_stocks)

ri_t_df = pd.DataFrame({
    "Stock_ID": np.repeat(range(1, n_stocks + 1), n_months),
    "Month": np.tile(range(1, n_months + 1), n_stocks),
    "Excess_Return": ri_t_flattened
})

combined_data = pd.concat([zi_t_df, ri_t_df["Excess_Return"]], axis=1)

X = combined_data[[col for col in combined_data.columns if col.startswith("z_")]]
y = combined_data["Excess_Return"]

from sklearn.model_selection import GroupShuffleSplit
def split_by_stock_fraction(df, test_fraction=0.3):
    test_indices = []
    train_indices = []

    for stock_id in df["Stock_ID"].unique():
        stock_df = df[df["Stock_ID"] == stock_id]
        n_test = int(len(stock_df) * test_fraction)
        test_idx = stock_df.sample(n=n_test, random_state=42).index
        train_idx = stock_df.index.difference(test_idx)

        test_indices.extend(test_idx)
        train_indices.extend(train_idx)

    return train_indices, test_indices



train_idx, test_idx = split_by_stock_fraction(combined_data, test_fraction=0.4)

X_train_val = X.iloc[train_idx].reset_index(drop=True)
y_train_val = y.iloc[train_idx].reset_index(drop=True)
meta_train_val = combined_data.loc[train_idx, ["Stock_ID", "Month"]].reset_index(drop=True)

X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)
meta_test = combined_data.loc[test_idx, ["Stock_ID", "Month"]].reset_index(drop=True)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2857, random_state=42
)

# -------------------------------
# Part 3: Model Training Section
# -------------------------------

# --- OLS Regression ---

ols_model = LinearRegression()
ols_model.fit(X_train_val, y_train_val)
y_pred_ols = ols_model.predict(X_test)

# --- Weighted Linear Regression ---

class WeigthtedLineaRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y, weights):
        W = np.diag(weights)
        XtW = X.T@W
        self.theta = np.linalg.inv(XtW @ X) @ XtW @ y

    def predict(self, X):
        return X @ self.theta
w_train = np.random.rand(X_train_val.shape[0])
wls_model = WeigthtedLineaRegression()
wls_model.fit(X_train_val.values,y_train_val.values,w_train.values if isinstance(w_train, pd.Series) else w_train)
y_pred_wls = wls_model.predict(X_test)

# --- Huber Regressor ---

hb_model = HuberRegressor()
hb_model.fit(X_train_val,y_train_val)
y_pred_hb = hb_model.predict(X_test)

# --- ElasticNet Model Tuning ---

best_mse = float('inf')
for alpha in [0.01, 0.1, 1.0, 10.0]:
    for l1_ratio in [0.1, 0.5, 0.9]:
        model = ElasticNet(alpha= alpha, l1_ratio= l1_ratio, fit_intercept=True, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_pred)

        if best_mse > mse_val:
            best_mse = mse_val
            best_alpha = alpha
            best_l1_ratio = l1_ratio

en_model = ElasticNet(alpha= best_alpha, l1_ratio=best_l1_ratio, fit_intercept=True, max_iter=1000, random_state=42)
en_model.fit(X_train_val,y_train_val)
y_pred_en = en_model.predict(X_test)


# --- Principal Component Regression (PCR) ---
best_pcr_mse = float('inf')
best_n_components_pcr = None

for n_components in range(1, min(X_train.shape[1], 20) + 1):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    lr_pcr = LinearRegression()
    lr_pcr.fit(X_train_pca, y_train)
    y_val_pred = lr_pcr.predict(X_val_pca)
    mse = mean_squared_error(y_val, y_val_pred)

    if mse < best_pcr_mse:
        best_pcr_mse = mse
        best_n_components_pcr = n_components

pca = PCA(n_components=best_n_components_pcr)
X_train_val_pca = pca.fit_transform(X_train_val)
X_test_pca = pca.transform(X_test)

lr_pcr = LinearRegression()
lr_pcr.fit(X_train_val_pca, y_train_val)
y_pred_pcr = lr_pcr.predict(X_test_pca)
# --- Partial Least Squares Regression (PLS) ---
best_pls_mse = float('inf')
best_n_components_pls = None

for n_components in range(1, min(X_train.shape[1], 20) + 1):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    y_val_pred = pls.predict(X_val).flatten()
    mse = mean_squared_error(y_val, y_val_pred)

    if mse < best_pls_mse:
        best_pls_mse = mse
        best_n_components_pls = n_components

pls = PLSRegression(n_components=best_n_components_pls)
pls.fit(X_train_val, y_train_val)
y_pred_pls = pls.predict(X_test).flatten()

# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---

spline_transformer = SplineTransformer(
    degree=2, n_knots=5, knots="uniform", include_bias=False
)

X_train_spline = spline_transformer.fit_transform(X_train)
X_val_spline = spline_transformer.transform(X_val)
X_test_spline = spline_transformer.transform(X_test)

l1_ratios = [0.1, 0.5, 0.9]
alphas = np.logspace(-3, 1, 10)

best_model = None
best_val_mse = float("inf")

for l1_ratio in l1_ratios:
    for alpha in alphas:
        model = ElasticNet(
            l1_ratio=l1_ratio,
            alpha=alpha,
            random_state=42,
            max_iter=10000
        )
        model.fit(X_train_spline, y_train)
        val_preds = model.predict(X_val_spline)
        val_mse = mean_squared_error(y_val, val_preds)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model

X_train_val_spline = spline_transformer.fit_transform(X_train_val)
best_model.fit(X_train_val_spline, y_train_val)

y_pred_glr = best_model.predict(X_test_spline)
# --- non-linear models ---

# --- Neural Network Model ---
def create_model(learning_rate=0.01, neurons_layer1=32, neurons_layer2=16, neurons_layer3=8):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(neurons_layer1, activation='relu'),
        tf.keras.layers.Dense(neurons_layer2, activation='relu'),
        tf.keras.layers.Dense(neurons_layer3, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model
best_model = None
best_val_loss = float('inf')

learning_rates = [0.001, 0.01, 0.1]
neurons_layer1_options = [32, 64]
neurons_layer2_options = [16, 32]
neurons_layer3_options = [8, 16]

for lr in learning_rates:
    for n1 in neurons_layer1_options:
        for n2 in neurons_layer2_options:
            for n3 in neurons_layer3_options:
                model = create_model(learning_rate=lr, neurons_layer1=n1, neurons_layer2=n2, neurons_layer3=n3)
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                    ]
                )
                val_loss = min(history.history['val_loss'])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model

y_pred_nn = best_model.predict(X_test).flatten()

# --- Gradient Boosting Regressor ---
brt_best_model = None
brt_best_rmse = float('inf')

for n_estimators in [50, 100, 200]:
    for learning_rate in [0.01, 0.1, 0.2]:
        for max_depth in [3, 5, 7]:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            if val_rmse < brt_best_rmse:
                brt_best_rmse = val_rmse
                brt_best_model = model
y_pred_brt = brt_best_model.predict(X_test)

# --- Random Forest Regressor ---
rf_best_model = None
rf_best_rmse = float('inf')

for n_estimators in [50, 100, 200]:
    for max_depth in [3, 5, 7]:
        for min_samples_split in [2, 5, 10]:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            if val_rmse < rf_best_rmse:
                rf_best_rmse = val_rmse
                rf_best_model = model

y_pred_rf = rf_best_model.predict(X_test)

# -------------------------------
# Part 4: Prediction Wrappers
# -------------------------------

predictions = {
    "OLS": y_pred_ols,
    "Weighted_Linear_Regression": y_pred_wls,
    "Huber": y_pred_hb,
    "ElasticNet": y_pred_en,
    "PCR": y_pred_pcr,
    "PLS": y_pred_pls,
    "Spline_ElasticNet": y_pred_glr,
    "Neural_Network": y_pred_nn,
    "Gradient_Boosting": y_pred_brt,
    "Random_Forest": y_pred_rf
}

results_df = pd.DataFrame(predictions)
results_df["True"] = y_test.reset_index(drop=True)
results_df["Stock_ID"] = meta_test["Stock_ID"].values
results_df["Month"] = meta_test["Month"].values
results_df.to_csv("Predictions.csv", index=False)
# -------------------------------
# Part 5: Full-Sample Time Series Plots - to see the predictions vs. actuals
# -------------------------------
def plot_predictions_by_stock(results_df, model_name):

    fig, axs = plt.subplots(5, 2, figsize=(15, 12))
    fig.suptitle(f"{model_name} Results for Stock Return Prediction", fontsize=16)
    axs = axs.flatten()

    for stock_id in range(1, 11):
        df_stock = results_df[results_df["Stock_ID"] == stock_id].sort_values("Month")

        axs[stock_id - 1].plot(df_stock["Month"], df_stock["True"], marker='o', label='Actual', color='blue')
        axs[stock_id - 1].plot(df_stock["Month"], df_stock[model_name], marker='x', linestyle='--', label='Predicted',
                               color='red')
        axs[stock_id - 1].set_title(f"Stock {stock_id}")
        axs[stock_id - 1].set_xlabel("Month")
        axs[stock_id - 1].set_ylabel("Excess Return")
        axs[stock_id - 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

for model in results_df:
    if model not in ["Stock_ID", "Month", "True"]:
        plot_predictions_by_stock(results_df, model)

# -------------------------------
# Part 6: Out-of-Sample R² Results Table - to evaluate model performance
# -------------------------------
# TODO: Calculate R² according to the formula: 1 - (sum of squared errors / total sum of squares)



# -------------------------------
# Part 7: Diebold-Mariano Test Statistics - to compare model predictions
# -------------------------------


# -------------------------------
# Part 8: Variable Importance Calculations & Heatmaps - to understand feature importance ( to see which features are more important)
# -------------------------------
# TODO: Define a function to compute variable importance based on the drop in R² when a feature is removed
# -------------------------------


# -------------------------------
# Part 9: Auxiliary Functions and Decile Portfolio Analysis - to analyze model performance across deciles - to compare predicted vs actual  sharpe ratios
# -------------------------------

