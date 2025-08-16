import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import gym
from gym import spaces
from stable_baselines3 import PPO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score

def load_and_preprocess_data(file_path="data.csv"):
    """
    Load and preprocess the stock data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please make sure it's in the root directory.")

    data = pd.read_csv(file_path)
    # Clean column names
    data.columns = data.columns.str.strip()

    # Drop unnamed column if it exists
    if '' in data.columns:
        data = data.drop(columns=[''])

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.set_index('Date', inplace=True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def run_rule_based_strategy(data, save_plot=True):
    """
    Run a rule-based trading strategy using technical indicators.
    """
    # Calculate technical indicators
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd(data['Close'])
    data['Signal_Line'] = ta.trend.macd_signal(data['Close'])

    # Vectorized signal generation
    buy_conditions = (data['SMA_50'] > data['SMA_200']) & (data['RSI'] < 30) & (data['MACD'] > data['Signal_Line'])
    sell_conditions = (data['SMA_50'] < data['SMA_200']) & (data['RSI'] > 70) & (data['MACD'] < data['Signal_Line'])

    data['Signal'] = 'HOLD'
    data.loc[buy_conditions, 'Signal'] = 'BUY'
    data.loc[sell_conditions, 'Signal'] = 'SELL'

    if save_plot:
        if not os.path.exists('charts'):
            os.makedirs('charts')
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
        plt.plot(data.index, data['SMA_50'], label='SMA 50', linestyle='--')
        plt.plot(data.index, data['SMA_200'], label='SMA 200', linestyle='--')
        plt.scatter(data.index[data['Signal'] == 'BUY'], data['Close'][data['Signal'] == 'BUY'], marker='^', color='g', label='Buy Signal', s=100)
        plt.scatter(data.index[data['Signal'] == 'SELL'], data['Close'][data['Signal'] == 'SELL'], marker='v', color='r', label='Sell Signal', s=100)
        plt.legend()
        plt.title('Rule-Based Trading Strategy')
        plt.savefig('charts/rule_based_strategy.png')
        plt.close()

    data.to_csv('results/trading_signals_rule_based.csv')
    return data

def preprocess_for_rl(data, lookback=50):
    """
    Preprocess data for the reinforcement learning model.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 3]) # 'Close' price is at index 3

    return np.array(X), np.array(y), scaler

def build_gru_model(input_shape):
    """
    Build and compile a GRU model for price prediction.
    """
    model = Sequential([
        GRU(32, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class StockTradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning.
    """
    def __init__(self, data, gru_model, lookback=50):
        super(StockTradingEnv, self).__init__()
        self.data = data.values
        self.gru_model = gru_model
        self.lookback = lookback
        self.current_step = lookback
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(lookback * data.shape[1] + 1,), dtype=np.float32)
        self._precompute_gru_predictions()

    def _precompute_gru_predictions(self):
        historical_data = np.array([self.data[i-self.lookback:i] for i in range(self.lookback, len(self.data))])
        self.gru_predictions = self.gru_model.predict(historical_data, batch_size=64, verbose=0)

    def reset(self):
        self.current_step = self.lookback
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        return self._get_observation()

    def step(self, action):
        current_price = self.data[self.current_step, 3] # Close price

        if action == 1:  # Buy
            if self.balance > 0:
                self.shares_held += self.balance / current_price
                self.balance = 0
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - 10000  # Simple reward based on net worth change

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        if self.current_step >= len(self.data):
             self.current_step = len(self.data) -1
        historical_data = self.data[self.current_step - self.lookback:self.current_step].flatten()
        gru_pred_index = self.current_step - self.lookback
        if gru_pred_index >= len(self.gru_predictions):
            gru_pred_index = len(self.gru_predictions) - 1
        gru_pred = self.gru_predictions[gru_pred_index]
        return np.concatenate([historical_data, gru_pred])

def evaluate_rl_agent(env, model, data, save_plots=True):
    """
    Evaluate the trained RL agent and plot the results.
    """
    obs = env.reset()
    rewards, net_worths, actions = [], [], []

    max_steps = len(data) - env.lookback -1
    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        net_worths.append(env.net_worth)
        if done:
            break

    if not os.path.exists('charts'):
        os.makedirs('charts')
    if not os.path.exists('results'):
        os.makedirs('results')

    # Plotting
    if save_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title('Rewards over Time')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.savefig('charts/rl_rewards.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(net_worths)
        plt.title('Net Worth over Time')
        plt.xlabel('Steps')
        plt.ylabel('Net Worth')
        plt.savefig('charts/rl_net_worth.png')
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(actions, bins=3, align='left', rwidth=0.8)
        plt.xticks(ticks=[0, 1, 2], labels=['Hold', 'Buy', 'Sell'])
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.savefig('charts/rl_action_distribution.png')
        plt.close()

    # Confusion Matrix and F1 Score
    net_worths = np.array(net_worths)
    actions = np.array(actions)
    trade_returns = np.diff(net_worths, prepend=net_worths[0])
    true_actions = np.zeros(len(actions), dtype=int)
    true_actions[trade_returns > 0] = 1  # Buy
    true_actions[trade_returns < 0] = 2  # Sell

    cm = confusion_matrix(true_actions, actions)
    f1 = f1_score(true_actions, actions, average='macro')

    action_labels = ['Hold', 'Buy', 'Sell']
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix (F1 Score: {f1:.4f})')
    plt.colorbar(label='Number of Instances')
    tick_marks = np.arange(len(action_labels))
    plt.xticks(tick_marks, action_labels, rotation=45)
    plt.yticks(tick_marks, action_labels)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    plt.xlabel('Predicted Action')
    plt.ylabel('True Action (Hindsight)')
    plt.tight_layout()
    plt.savefig('charts/rl_confusion_matrix.png')
    plt.close()

    print(f"Final Net Worth: {env.net_worth}")
    print(f"F1 Score (Macro): {f1:.4f}")

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()

    # Run Rule-Based Strategy
    print("Running Rule-Based Strategy...")
    rule_based_results = run_rule_based_strategy(data.copy())
    print("Rule-Based Strategy Complete. Results saved to 'results/trading_signals_rule_based.csv' and plot to 'charts/rule_based_strategy.png'.")

    # Prepare data for RL
    X_rl, y_rl, scaler_rl = preprocess_for_rl(data)

    # Build and train GRU model
    print("\nTraining GRU model for price prediction...")
    gru_model = build_gru_model((X_rl.shape[1], X_rl.shape[2]))
    gru_model.fit(X_rl, y_rl, epochs=5, batch_size=64, verbose=1, validation_split=0.2)
    print("GRU model training complete.")

    # Create and train RL agent
    print("\nTraining Reinforcement Learning (PPO) agent...")
    env = StockTradingEnv(data, gru_model)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    print("RL agent training complete.")

    # Evaluate RL agent
    print("\nEvaluating RL agent...")
    evaluate_rl_agent(env, model, data)
    print("RL agent evaluation complete. Plots saved to 'charts/' directory.")

if __name__ == "__main__":
    main()
