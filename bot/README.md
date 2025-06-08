# ATM Popularity Prediction Telegram Bot

This Telegram bot allows users to predict ATM popularity using various machine learning models. The bot supports multiple models including linear regression and ridge regression, each with different performance metrics.

## Features

- Welcome message with model metrics overview
- Model selection via interactive buttons
- CSV file processing for predictions
- Support for multiple ML models with different performance characteristics

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a Telegram bot:
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Use the `/newbot` command to create a new bot
   - Copy the API token provided by BotFather

4. Set up environment variable:
   - Create a `.env` file in the project root
   - Add your Telegram bot token:
     ```
     TELEGRAM_BOT_TOKEN=your_bot_token_here
     ```

## Running the Bot

1. Start the bot:
   ```bash
   python telegram_bot.py
   ```

2. Open Telegram and start a chat with your bot
3. Use the following commands:
   - `/start` - Get welcome message and model metrics
   - `/select_model` - Choose a prediction model
   - Send a CSV file with features to get predictions

## CSV File Format

The bot expects CSV files with the following columns:
- `lat` - Latitude
- `long` - Longitude
- `settlement` - Settlement name
- `street_name` - Street name
- `settlement_count` - Number of settlements
- `atm_group` - ATM group
- `postal_code` - Postal code

## Available Models

1. Linear Model
   - MAE: 0.066
   - MAPE: 1.982
   - R2: 0.045

2. Linear Model (Scaled)
   - MAE: 0.066
   - MAPE: 1.982
   - R2: 0.045

3. Ridge Model
   - MAE: 0.032
   - MAPE: 2.451
   - R2: 0.787

## Note

This is a basic implementation. In a production environment, you would need to:
1. Load and use the actual trained models
2. Implement proper error handling
3. Add data validation and preprocessing
4. Add security measures
5. Implement proper logging and monitoring
