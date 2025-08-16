# Adaptive Stock Trading Bot using Reinforcement Learning

This project implements and compares two stock trading strategies: a traditional rule-based approach using technical indicators and an adaptive strategy using a Deep Reinforcement Learning (RL) agent. The goal is to create an intelligent agent that can make profitable trading decisions by learning from historical stock data.

## Features

- **Rule-Based Strategy:** A trading bot that generates BUY, SELL, and HOLD signals based on common technical indicators:
  - Simple Moving Averages (SMA-50 and SMA-200)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)

- **Reinforcement Learning Strategy:** An adaptive trading agent built using:
  - **Deep Q-Network (DQN):** To learn optimal trading policies.
  - **Proximal Policy Optimization (PPO):** An advanced RL algorithm from Stable Baselines3 for robust training.
  - **Gated Recurrent Unit (GRU):** A type of recurrent neural network used to predict future price movements, which are then fed into the RL agent's observations.

- **Performance Evaluation:** The project includes metrics to evaluate the RL agent's performance, including:
  - Net worth over time
  - Rewards per step
  - Action distribution (Buy/Sell/Hold)
  - Confusion matrix and F1-score against a hindsight-optimal strategy.

## Project Structure

```
.
├── charts/
│   ├── rl_action_distribution.png
│   ├── rl_confusion_matrix.png
│   ├── rl_net_worth.png
│   ├── rl_rewards.png
│   └── rule_based_strategy.png
├── results/
│   └── trading_signals_rule_based.csv
├── data.csv
├── trading_bot.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

To run this project, you'll need Python 3. Follow these steps to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/adaptive-stock-trading-bot.git
   cd adaptive-stock-trading-bot
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add data:**
   Place your historical stock data file named `data.csv` in the root of the project directory. The CSV should have the following columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

## Usage

To run the trading bot and generate the results, execute the main script:

```bash
python trading_bot.py
```

The script will:
1. Load and preprocess the data from `data.csv`.
2. Run the rule-based strategy and save the results to `results/` and a plot to `charts/`.
3. Train the GRU model for price prediction.
4. Train the PPO reinforcement learning agent.
5. Evaluate the RL agent and save performance plots to the `charts/` directory.

## Strategies Explained

### Rule-Based Strategy

This strategy uses a combination of technical indicators to make trading decisions. The rules are as follows:
- **BUY Signal:** When the 50-day SMA is above the 200-day SMA, the RSI is below 30 (oversold), and the MACD line is above the signal line.
- **SELL Signal:** When the 50-day SMA is below the 200-day SMA, the RSI is above 70 (overbought), and the MACD line is below the signal line.
- **HOLD Signal:** If neither of the above conditions is met.

### Reinforcement Learning Strategy

The RL agent learns to trade by interacting with a custom `StockTradingEnv` (built with OpenAI Gym). The agent's goal is to maximize its net worth over time.

- **Observation Space:** The agent observes the historical price data (Open, High, Low, Close, Volume) for a given lookback period, plus a price prediction from the GRU model.
- **Action Space:** The agent can take one of three actions: Hold (0), Buy (1), or Sell (2).
- **Reward:** The reward at each step is based on the change in the agent's net worth.

## Results & Visualizations

After running the script, the `charts/` directory will contain the following plots:

- **Rule-Based Strategy Plot:**
  ![Rule-Based Strategy](charts/rule_based_strategy.png)

- **RL Agent Performance:**
  - **Net Worth:**
    ![RL Net Worth](charts/rl_net_worth.png)
  - **Action Distribution:**
    ![RL Action Distribution](charts/rl_action_distribution.png)
  - **Confusion Matrix:**
    ![RL Confusion Matrix](charts/rl_confusion_matrix.png)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
