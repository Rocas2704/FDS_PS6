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
import os
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting datasets
from sklearn.metrics import mean_squared_error, r2_score  # Import metrics for evaluating models
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet  # Import linear regression models
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.cross_decomposition import PLSRegression  # Import PLSRegression for partial least squares regression
from sklearn.preprocessing import SplineTransformer  # Import SplineTransformer for spline feature transformation
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # Import ensemble regression models


plt.style.use('seaborn-v0_8-whitegrid')  # Set the plotting style to Seaborn whitegrid
plt.rcParams['figure.figsize'] = (12, 8)  # Set default figure size for plots to 12x8 inches
os.makedirs("outputs", exist_ok=True)
# -------------------------------
# Part 1: Data Generation
# -------------------------------
n_stocks = 100  # Define the number of stocks
n_months = 24  # Define the number of months (time periods)
n_characteristics = 5  # Define the number of stock-specific characteristics
n_macro_factors = 3  # Define the number of macroeconomic factors

np.random.seed(42)  # Set the random seed for reproducibility
stock_characteristics = np.random.rand(n_stocks, n_months,
                                       n_characteristics)  # Generate random stock characteristic data
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
        ri_t_plus_1[i, t] = zi_t[i, t, :].dot(theta) + np.random.normal(0, 0.2)

# Flatten zi_t: (n_stocks * n_months, n_features)
zi_t_flattened = zi_t.reshape(n_stocks * n_months, -1)

# Flatten ri_t_plus_1: (n_stocks * n_months,)
ri_t_flattened = ri_t_plus_1.flatten()

zi_t_df = pd.DataFrame(
    zi_t_flattened,
    columns=[f"z_{k + 1}" for k in range(zi_t_flattened.shape[1])]
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


def split_by_stock_fraction(df, test_fraction):
    """
        Custom train-test splitting method for panel data (e.g., stock-month panel).

        This function performs a group-aware train-test split. Instead of using
        sklearn's train_test_split—which randomly selects individual rows without
        accounting for group structure—this function ensures that each group (here: each stock)
        contributes a specified fraction of its observations to the test set.

        This is especially useful in panel or time-series cross-sectional data where
        each entity (e.g., stock, firm, user) has multiple observations across time,
        and we want balanced representation in both train and test sets.

        Parameters:
        -----------
        df : pd.DataFrame
            The complete dataset containing a "Stock_ID" column used for grouping.

        test_fraction : float
            The fraction of observations to be assigned to the test set for each group (stock).
            For example, 0.3 assigns 30% of each stock's data to the test set and the rest to train.

        Returns:
        --------
        train_indices : list
            A list of integer indices corresponding to the training set rows.

        test_indices : list
            A list of integer indices corresponding to the test set rows.
        """

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


train_idx, test_idx = split_by_stock_fraction(combined_data, test_fraction=0.3)

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
        XtW = X.T @ W
        self.theta = np.linalg.inv(XtW @ X) @ XtW @ y

    def predict(self, X):
        return X @ self.theta


w_train = np.random.rand(X_train_val.shape[0])
wls_model = WeigthtedLineaRegression()
wls_model.fit(X_train_val.values, y_train_val.values, w_train.values if isinstance(w_train, pd.Series) else w_train)
y_pred_wls = wls_model.predict(X_test)

# --- Huber Regressor ---

hb_model = HuberRegressor()
hb_model.fit(X_train_val, y_train_val)
y_pred_hb = hb_model.predict(X_test)

# --- ElasticNet Model Tuning ---

best_mse = float('inf')
for alpha in [0.01, 0.1, 1.0, 10.0]:
    for l1_ratio in [0.1, 0.5, 0.9]:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_pred)

        if best_mse > mse_val:
            best_mse = mse_val
            best_alpha = alpha
            best_l1_ratio = l1_ratio

en_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=True, max_iter=1000, random_state=42)
en_model.fit(X_train_val, y_train_val)
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

best_spline_model = None
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
            best_spline_model = model

X_train_val_spline = spline_transformer.fit_transform(X_train_val)
best_spline_model.fit(X_train_val_spline, y_train_val)
y_pred_glr = best_spline_model.predict(X_test_spline)


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


best_nn_model = None
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
                    best_nn_model = model

y_pred_nn = best_nn_model.predict(X_test).flatten()

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
results_df.to_csv("outputs/Predictions.csv", index=False)


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
def r2_oos(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    return 1 - numerator / denominator


r2_scores = {}

for model_name in predictions.keys():
    r2_scores[model_name] = r2_oos(y_test.values, predictions[model_name])

r2_df = pd.DataFrame.from_dict(r2_scores, orient='index', columns=['Out-of-Sample R²'])
r2_df = r2_df.sort_values('Out-of-Sample R²', ascending=False)

r2_df.to_csv("outputs/R2_oos_results.csv", index=True)


# -------------------------------
# Part 7: Diebold-Mariano Test Statistics - to compare model predictions
# -------------------------------

def newey_west_se(data, lag=1):
    n = len(data)
    data_mean = np.mean(data)
    gamma = [np.sum((data[i:] - data_mean) * (data[:n - i] - data_mean)) / n for i in range(lag + 1)]
    nw_var = gamma[0] + 2 * np.sum([(1 - i / (lag + 1)) * gamma[i] for i in range(1, lag + 1)])
    return np.sqrt(nw_var / n)


def diebold_mariano(y_true, y_pred1, y_pred2, lag=1):
    e1_sq = (y_true - y_pred1) ** 2
    e2_sq = (y_true - y_pred2) ** 2
    d = e1_sq - e2_sq  # pointwise difference in squared errors
    mean_d = np.mean(d)
    se_d = newey_west_se(d, lag=lag)
    dm_stat = mean_d / se_d
    return dm_stat


model_names = list(predictions.keys())
dm_matrix = pd.DataFrame(index=model_names, columns=model_names)

for m1 in model_names:
    for m2 in model_names:
        if m1 != m2:
            dm_stat = diebold_mariano(y_test.values, predictions[m1], predictions[m2])
            dm_matrix.loc[m1, m2] = dm_stat
        else:
            dm_matrix.loc[m1, m2] = np.nan

dm_matrix.to_csv("outputs/DM_test_results.csv")
dm_matrix = dm_matrix.astype(float)

plt.figure(figsize=(12, 10))

sns.heatmap(
    dm_matrix.astype(float),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",  # azul (negativo) a rojo (positivo)
    center=0,
    linewidths=0.5,
    cbar_kws={'label': 'DM Statistic'}
)

plt.title("Diebold-Mariano Test Statistic Matrix", fontsize=14)
plt.xlabel("Model 2 (Column Model)")
plt.ylabel("Model 1 (Row Model)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/DM_test_heatmap.png", dpi=300)
plt.show()


# -------------------------------
# Part 8: Variable Importance Calculations & Heatmaps - to understand feature importance ( to see which features are more important)
# -------------------------------
def compute_r2_importance(model, X_test, y_test):
    def compute_r2_importance(model, X_test, y_test):
        """
        Compute R²-based feature importance by variable omission.

        This method estimates the importance of each input feature by measuring the drop
        in out-of-sample R² when that feature is zeroed out. For each feature j, the model
        is evaluated on a modified version of X_test where feature j is replaced with zero,
        and the reduction in R² is interpreted as the importance of that feature.

        Parameters
        ----------
        model : object
            A trained regression model with a `.predict()` method.
        X_test : array-like or DataFrame, shape (n_samples, n_features)
            The test feature matrix.
        y_test : array-like, shape (n_samples,)
            The true target values corresponding to `X_test`.

        Returns
        -------
        list of float
            A list of R² drops (importances) for each feature. Higher values indicate greater
            importance. The length of the list equals the number of features in `X_test`.

        Notes
        -----
        - This method assumes features are numerically centered around 0 or that setting
          them to zero has a meaningful "removal" effect.
        - If features are strongly correlated, this method may overestimate their importance.
        """
        r2_full = r2_score(y_test, model.predict(X_test))
        importances = []

        for j in range(X_test.shape[1]):
            X_modified = X_test.copy()
            if isinstance(X_modified, pd.DataFrame):
                X_modified.iloc[:, j] = 0
            else:
                X_modified[:, j] = 0
            r2_j = r2_score(y_test, model.predict(X_modified))
            importances.append(r2_full - r2_j)

        return importances

    r2_full = r2_score(y_test, model.predict(X_test))
    importances = []

    for j in range(X_test.shape[1]):
        X_modified = X_test.copy()
        if isinstance(X_modified, pd.DataFrame):
            X_modified.iloc[:, j] = 0
        else:
            X_modified[:, j] = 0
        r2_j = r2_score(y_test, model.predict(X_modified))
        importances.append(r2_full - r2_j)

    return importances


def compute_ssd_importance(model, X_train, epsilon=1e-5):
    """
        Compute feature importance using the Sum of Squared Derivatives (SSD) method.

        This method approximates the partial derivative of the model output with respect
        to each input feature using finite differences. For each feature j, it perturbs
        the feature slightly in both directions (±epsilon), computes the gradient at
        each training sample, squares it, and sums across all samples.

        The result reflects how sensitive the model's predictions are to small changes
        in each input feature, capturing the local effect of each variable.

        Parameters
        ----------
        model : object
            A trained regression model with a `.predict()` method that accepts
            2D input arrays.
        X_train : ndarray, shape (n_samples, n_features)
            The feature matrix used to estimate sensitivity (usually training data).
        epsilon : float, optional (default=1e-5)
            The magnitude of perturbation used to compute numerical gradients.

        Returns
        -------
        list of float
            A list of SSD values (one per feature). Higher values indicate greater
            local influence on the model’s predictions.

        Notes
        -----
        - This method is computationally expensive: O(n_samples × n_features) predictions.
        - It is most appropriate for differentiable, continuous-output models.
        """
    partial_squares = []

    for j in range(X_train.shape[1]):
        squared_grads = []
        for i in range(X_train.shape[0]):
            x_plus = X_train[i:i + 1].copy()
            x_minus = X_train[i:i + 1].copy()

            x_plus[0, j] += epsilon
            x_minus[0, j] -= epsilon

            pred_plus = model.predict(x_plus)[0]
            pred_minus = model.predict(x_minus)[0]
            grad = (pred_plus - pred_minus) / (2 * epsilon)
            squared_grads.append(grad ** 2)

        partial_squares.append(np.sum(squared_grads))

    return partial_squares


# --- Plot Function ---
def plot_importance_heatmap(importance_dict, title="Variable Importance", method="R² Drop"):
    df_importance = pd.DataFrame(importance_dict)
    df_importance.index = [f"Feature {i + 1}" for i in range(df_importance.shape[0])]

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_importance, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"{method} Variable Importance Across Models")
    plt.ylabel("Feature")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.show()


importance_r2 = {}
importance_ssd = {}

# Group 1: Raw features (X_train_val)
linear_models = {
    "OLS": ols_model,
    "Weighted_Linear_Regression": wls_model,
    "Huber": hb_model,
    "ElasticNet": en_model,
    "PLS": pls,
    "Gradient_Boosting": brt_best_model,
    "Random_Forest": rf_best_model,
    "PCR": lr_pcr
}

for name, model in linear_models.items():
    if name == "PCR":
        importance_r2[name] = compute_r2_importance(model, X_test_pca, y_test)
        importance_ssd[name] = compute_ssd_importance(model, X_train_val_pca)
    else:
        importance_r2[name] = compute_r2_importance(model, X_test, y_test)
        importance_ssd[name] = compute_ssd_importance(model, X_train_val.values)

plot_importance_heatmap(importance_r2, method="R² Drop")

# --- Normalización SSD ---
ssd_df_normalized = pd.DataFrame(importance_ssd)
ssd_df_normalized = ssd_df_normalized.div(ssd_df_normalized.sum(axis=0), axis=1)
plot_importance_heatmap(ssd_df_normalized.to_dict(), method="SSD (Normalized)")

spline_models = {
    "Spline_ElasticNet": best_spline_model
}

importance_r2_spline = {}
importance_ssd_spline = {}

for name, model in spline_models.items():
    importance_r2_spline[name] = compute_r2_importance(model, X_test_spline, y_test)
    importance_ssd_spline[name] = compute_ssd_importance(model, X_train_val_spline)

plot_importance_heatmap(importance_r2_spline, method="R² Drop (Spline)")

ssd_spline_df = pd.DataFrame(importance_ssd_spline)
ssd_spline_df = ssd_spline_df.div(ssd_spline_df.sum(axis=0), axis=1)
plot_importance_heatmap(ssd_spline_df.to_dict(), method="SSD (Spline Normalized)")


# -------------------------------
# Part 9: Auxiliary Functions and Decile Portfolio Analysis - to analyze model performance across deciles - to compare predicted vs actual  sharpe ratios
# -------------------------------

def assign_deciles(df, prediction_column):
    """
        Assign decile ranks to predicted values within each month.

        This function groups the data by 'Month' and assigns decile ranks (1 to 10)
        to the values in `prediction_column`. If there are not enough unique values
        to create 10 deciles, it attempts to assign fewer (e.g., 9). If ranking fails,
        it returns NaN for that group.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing at least the columns "Month" and the prediction column.
        prediction_column : str
            The name of the column containing model predictions to be ranked.

        Returns
        -------
        pandas.Series
            A Series of decile labels (1 to 10, or fewer if needed), aligned with `df`.

        Notes
        -----
        - Deciles are computed using `pd.qcut` on the rank-transformed predictions.
        - Ties are resolved by using the `first` method in `rank()`.
        - If `x.nunique()` is less than the number of deciles, the function tries with q-1.
        - Returns NaN if `qcut` fails (e.g., due to all values being equal).
        """

    def try_qcut(x, q):
        try:
            if x.nunique() >= q:
                return pd.qcut(x.rank(method="first"), q, labels=False, duplicates='drop') + 1
            else:
                return pd.qcut(x.rank(method="first"), q - 1, labels=False, duplicates='drop') + 1
        except Exception:
            return pd.Series([np.nan] * len(x), index=x.index)

    return df.groupby("Month")[prediction_column].transform(lambda x: try_qcut(x, 10))


sample_model = "ElasticNet"

df = results_df.copy()
df["Decile"] = assign_deciles(df, sample_model)


def decile_portfolio_analysis(results_df, model_name):
    """
        Perform decile-based portfolio analysis for a given prediction model.

        This function evaluates the predictive quality of a model by sorting predicted
        returns into deciles within each month and analyzing the actual realized returns
        of each decile portfolio.

        It computes:
        - The average realized return per decile.
        - The average standard deviation per decile.
        - The Sharpe ratio per decile (mean / std).
        - The long-short return: return of decile 10 minus decile 1.

        Parameters
        ----------
        results_df : pandas.DataFrame
            A DataFrame containing at least the columns "Month", "True", and the
            specified `model_name` column with predicted values.
        model_name : str
            The name of the column in `results_df` containing predicted values from the model.

        Returns
        -------
        portfolio_returns : pandas.Series
            The mean realized returns for each decile across all months.
        sharpe_ratios : pandas.Series
            The Sharpe ratios for each decile portfolio (mean divided by std deviation).
        long_short : float
            The return spread between the highest and lowest decile portfolios
            (i.e., Decile 10 return minus Decile 1 return).

        Notes
        -----
        - Requires that `assign_deciles()` has reasonable performance assigning deciles.
        - If a decile has only one observation in some months, its standard deviation
          may be NaN, resulting in a NaN Sharpe ratio.
        - Assumes risk-free rate is 0 for Sharpe ratio calculation.
        """
    df = results_df.copy()
    df["Decile"] = assign_deciles(df, model_name)

    grouped = df.groupby(["Month", "Decile"])["True"].agg(['mean', 'std']).reset_index()
    grouped.columns = ["Month", "Decile", "Return", "Std"]

    portfolio_returns = grouped.groupby("Decile")["Return"].mean()
    portfolio_std = grouped.groupby("Decile")["Std"].mean()

    sharpe_ratios = portfolio_returns / portfolio_std.replace(0, np.nan)

    long_short = portfolio_returns.loc[10] - portfolio_returns.loc[1]

    return portfolio_returns, sharpe_ratios, long_short


portfolio_table = {}
sharpe_table = {}
longshort_table = {}
valid_models = []

for model in results_df.columns:
    if model not in ["Stock_ID", "Month", "True"]:
        try:
            port_ret, sharpe, longshort = decile_portfolio_analysis(results_df, model)

            if sharpe.notna().sum() >= 1:
                portfolio_table[model] = port_ret
                sharpe_table[model] = sharpe
                longshort_table[model] = longshort
                valid_models.append(model)

        except Exception as e:
            print(f"Modelo {model} excluido por error: {e}")

portfolio_df = pd.DataFrame(portfolio_table)
sharpe_df = pd.DataFrame(sharpe_table)
longshort_df = pd.Series(longshort_table)

# --- Heatmap only with valid data ---
if not sharpe_df.empty and sharpe_df.notna().values.any():
    plt.figure(figsize=(12, 6))
    sns.heatmap(sharpe_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Sharpe Ratios by Decile and Model")
    plt.ylabel("Decile")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.show()
else:
    print("No valid data")

# --- Long-Short Sharpe Ratios ---
if not longshort_df.empty and longshort_df.notna().values.any():
    plt.figure(figsize=(10, 5))
    longshort_df.sort_values().plot(kind="barh", color="teal")
    plt.title("Long-Short Portfolio Sharpe Ratios")
    plt.xlabel("Sharpe Ratio")
    plt.tight_layout()
    plt.show()
else:
    print(" No valid data")

sharpe_df.to_csv("outputs/Sharpe_Ratios_By_Decile.csv")
longshort_df.to_csv("outputs/LongShort_Sharpe_Ratios.csv")
