import os
import logging
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import io
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
        'MAE': 0.036,
        'MAPE': 1.90,
        'R2': 0.71
    },
    'catboost_model': {
        'MAE': 0.034,
        'MAPE': 1.94,
        'R2': 0.74
    },
    'neural_network': {
        'MAE': 0.032,
        'MAPE': 2.451,
        'R2': 0.787
    }
}

# Available models
AVAILABLE_MODELS = ['linear_model', 'neural_network', 'catboost_model']

# Store user's selected model
user_states = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = (
        "Добро пожаловать в бот предсказания популярности банкоматов!\n\n"
        "Этот бот использует модели машинного обучения для предсказания популярности банкоматов на основе различных параметров.\n\n"
        "Доступные модели и их метрики:\n"
    )
    
    for model_name, metrics in MODEL_METRICS.items():
        welcome_message += f"\n{model_name.upper()}:\n"
        welcome_message += f"MAE: {metrics['MAE']:.3f}\n"
        welcome_message += f"MAPE: {metrics['MAPE']:.3f}\n"
        welcome_message += f"R2: {metrics['R2']:.3f}\n"
    
    welcome_message += "\nЧтобы начать:\n"
    welcome_message += "1. Используйте /select_model для выбора модели предсказания\n"
    welcome_message += "2. Отправьте CSV файл с параметрами для получения предсказаний\n"
    
    await update.message.reply_text(welcome_message)

async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle model selection."""
    keyboard = []
    for model in AVAILABLE_MODELS:
        keyboard.append([InlineKeyboardButton(model, callback_data=f"model_{model}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Пожалуйста, выберите модель:', reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("model_"):
        model_name = query.data.replace("model_", "")
        user_id = update.effective_user.id
        
        # Call the backend API to change the model
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{BACKEND_URL}/api/v1/change_model/",
                    json={"model_type": model_name}
                ) as response:
                    if response.status == 200:
                        # Update user state only if model change was successful
                        user_states[user_id] = {"selected_model": model_name}
                        
                        await query.edit_message_text(
                            f"Выбрана модель: {model_name}\n\n"
                            f"Метрики модели:\n"
                            f"MAE: {MODEL_METRICS[model_name]['MAE']:.3f}\n"
                            f"MAPE: {MODEL_METRICS[model_name]['MAPE']:.3f}\n"
                            f"R2: {MODEL_METRICS[model_name]['R2']:.3f}\n\n"
                            "Теперь вы можете отправить CSV файл с параметрами для получения предсказаний."
                        )
                    else:
                        error_text = await response.text()
                        await query.edit_message_text(
                            f"Ошибка при смене модели:\n"
                            f"Код статуса: {response.status}\n"
                            f"Ошибка: {error_text}\n\n"
                            "Пожалуйста, попробуйте выбрать модель снова."
                        )
            except Exception as e:
                await query.edit_message_text(
                    f"Ошибка подключения к серверу:\n"
                    f"{str(e)}\n\n"
                    "Пожалуйста, попробуйте выбрать модель снова."
                )

async def handle_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming CSV files."""
    user_id = update.effective_user.id
    
    if user_id not in user_states or "selected_model" not in user_states[user_id]:
        await update.message.reply_text(
            "Пожалуйста, сначала выберите модель с помощью команды /select_model."
        )
        return
    
    try:
        # Get the file
        file = await context.bot.get_file(update.message.document)
        file_bytes = await file.download_as_bytearray()
        
        # Read CSV with windows-1251 encoding
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='windows-1251')
        except UnicodeDecodeError:
            # If windows-1251 fails, try utf-8
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8')
        
        # Validate required columns
        required_columns = ['lat', 'long', 'settlement', 'street_name', 'settlement_count', 'atm_group', 'postal_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            await update.message.reply_text(
                f"Ошибка: Отсутствуют обязательные столбцы: {', '.join(missing_columns)}\n"
                "Пожалуйста, убедитесь, что ваш CSV файл содержит все необходимые столбцы."
            )
            return

        # Send processing message
        processing_message = await update.message.reply_text(
            f"Обработка вашего файла с помощью модели {user_states[user_id]['selected_model']}...\n"
            "Это может занять несколько моментов."
        )
        
        # Prepare the file for API request
        csv_data = df.to_csv(index=False).encode('windows-1251')
        
        # Create a proper multipart form data with the file
        data = aiohttp.FormData()
        data.add_field('file',
                      csv_data,
                      filename='file.csv',
                      content_type='text/csv')
        
        # Make request to backend API
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{BACKEND_URL}/api/v1/predict/", data=data) as response:
                if response.status == 200:
                    # Get predictions - handle windows-1251 encoding
                    predictions_csv = await response.text()
                    try:
                        predictions_df = pd.read_csv(io.StringIO(predictions_csv), encoding='windows-1251')
                    except UnicodeDecodeError:
                        predictions_df = pd.read_csv(io.StringIO(predictions_csv), encoding='utf-8')
                    
                    # Update processing message with results
                    await processing_message.edit_text(
                        f"Предсказания завершены!\n\n"
                        f"Использованная модель: {user_states[user_id]['selected_model']}\n"
                        f"Количество предсказаний: {len(predictions_df)}\n\n"
                        "Вы можете скачать результаты, используя кнопку ниже."
                    )
                    
                    # Send predictions file - ensure windows-1251 encoding
                    predictions_file = io.BytesIO(predictions_df.to_csv(index=False).encode('windows-1251'))
                    predictions_file.name = "predictions.csv"
                    await update.message.reply_document(
                        document=predictions_file,
                        caption="Вот ваши предсказания!"
                    )
                else:
                    error_text = await response.text()
                    await processing_message.edit_text(
                        f"Ошибка при получении предсказаний:\n"
                        f"Код статуса: {response.status}\n"
                        f"Ошибка: {error_text}"
                    )
    except Exception as e:
        logger.error(f"Ошибка обработки CSV файла: {str(e)}")
        await update.message.reply_text(
            f"Ошибка обработки вашего файла:\n{str(e)}\n\n"
            "Пожалуйста, убедитесь, что ваш CSV файл правильно отформатирован и попробуйте снова."
        )

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