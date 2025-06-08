import os
import logging
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
import joblib
import io
import requests
import aiohttp

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Backend API configuration
BACKEND_HOST = os.getenv('BACKEND_HOST', 'backend')
BACKEND_PORT = os.getenv('BACKEND_PORT', '8080')
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# Model metrics from training
MODEL_METRICS = {
    'linear_model': {
        'MAE': 0.066,
        'MAPE': 1.982,
        'R2': 0.045
    },
    'linear_model_scaled': {
        'MAE': 0.066,
        'MAPE': 1.982,
        'R2': 0.045
    },
    'ridge': {
        'MAE': 0.032,
        'MAPE': 2.451,
        'R2': 0.787
    }
}

# Available models
AVAILABLE_MODELS = ['linear_model', 'linear_model_scaled', 'ridge']

# Store user's selected model
user_states = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = (
        "Welcome to the ATM Popularity Prediction Bot!\n\n"
        "This bot uses machine learning models to predict ATM popularity based on various features.\n\n"
        "Available models and their metrics:\n"
    )
    
    for model_name, metrics in MODEL_METRICS.items():
        welcome_message += f"\n{model_name.upper()}:\n"
        welcome_message += f"MAE: {metrics['MAE']:.3f}\n"
        welcome_message += f"MAPE: {metrics['MAPE']:.3f}\n"
        welcome_message += f"R2: {metrics['R2']:.3f}\n"
    
    welcome_message += "\nTo get started:\n"
    welcome_message += "1. Use /select_model to choose a prediction model\n"
    welcome_message += "2. Send a CSV file with features to get predictions\n"
    
    await update.message.reply_text(welcome_message)

async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle model selection."""
    keyboard = []
    for model in AVAILABLE_MODELS:
        keyboard.append([InlineKeyboardButton(model, callback_data=f"model_{model}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Please select a model:', reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("model_"):
        model_name = query.data.replace("model_", "")
        user_id = update.effective_user.id
        user_states[user_id] = {"selected_model": model_name}
        
        await query.edit_message_text(
            f"Selected model: {model_name}\n\n"
            f"Model metrics:\n"
            f"MAE: {MODEL_METRICS[model_name]['MAE']:.3f}\n"
            f"MAPE: {MODEL_METRICS[model_name]['MAPE']:.3f}\n"
            f"R2: {MODEL_METRICS[model_name]['R2']:.3f}\n\n"
            "Now you can send a CSV file with features to get predictions."
        )

async def handle_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming CSV files."""
    user_id = update.effective_user.id
    
    if user_id not in user_states or "selected_model" not in user_states[user_id]:
        await update.message.reply_text(
            "Please select a model first using /select_model command."
        )
        return
    
    try:
        # Get the file
        file = await context.bot.get_file(update.message.document)
        file_bytes = await file.download_as_bytearray()
        
        # Read CSV to validate columns
        df = pd.read_csv(io.BytesIO(file_bytes))
        
        # Validate required columns
        required_columns = ['lat', 'long', 'settlement', 'street_name', 'settlement_count', 'atm_group', 'postal_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            await update.message.reply_text(
                f"Error: Missing required columns: {', '.join(missing_columns)}\n"
                "Please ensure your CSV file contains all required columns."
            )
            return

        # Send processing message
        processing_message = await update.message.reply_text(
            f"Processing your file with {user_states[user_id]['selected_model']} model...\n"
            "This may take a few moments."
        )
        
        # Prepare the file for API request
        csv_data = df.to_csv(index=False).encode('windows-1251')
        files = {"file": ("file.csv", csv_data)}
        
        # Make request to backend API
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{BACKEND_URL}/api/v1/predict/", data=files) as response:
                if response.status == 200:
                    # Get predictions
                    predictions_csv = await response.text()
                    predictions_df = pd.read_csv(io.StringIO(predictions_csv))
                    
                    # Update processing message with results
                    await processing_message.edit_text(
                        f"Predictions completed!\n\n"
                        f"Model used: {user_states[user_id]['selected_model']}\n"
                        f"Number of predictions: {len(predictions_df)}\n\n"
                        "You can download the results using the button below."
                    )
                    
                    # Send predictions file
                    predictions_file = io.BytesIO(predictions_df.to_csv(index=False).encode('utf-8'))
                    predictions_file.name = "predictions.csv"
                    await update.message.reply_document(
                        document=predictions_file,
                        caption="Here are your predictions!"
                    )
                else:
                    error_text = await response.text()
                    await processing_message.edit_text(
                        f"Error getting predictions:\n"
                        f"Status code: {response.status}\n"
                        f"Error: {error_text}"
                    )
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        await update.message.reply_text(
            "Error processing your CSV file. Please ensure it's properly formatted "
            "and contains all required columns."
        )
        if 'processing_message' in locals():
            await processing_message.delete()

def main():
    """Start the bot."""
    # Get token from environment variable
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise ValueError("No TELEGRAM_BOT_TOKEN found in environment variables")
    
    # Create the Application
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("select_model", select_model))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_csv))

    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 