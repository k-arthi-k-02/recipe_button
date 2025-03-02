import pandas as pd
import re
import pytz
import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes, filters
)

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define conversation states
INGREDIENTS, CHOICE, EXTRA_INFO = range(3)

# Global variables for dataset, vectorizer, etc.
df = None
ing_col = None
title_col = None
instruction_col = None
vectorizer = None
recipe_vectors = None

def load_dataset() -> bool:
    """Load and preprocess the dataset from CSV."""
    global df, ing_col, title_col, instruction_col, vectorizer, recipe_vectors
    try:
        file_path = r"C:\iopproject\recipe_project\IndianFoodDatasetCSV.csv"
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        return False

    # Determine ingredients column
    if "TranslatedIngredients" in df.columns:
        ing_col = "TranslatedIngredients"
    elif "Ingredients" in df.columns:
        ing_col = "Ingredients"
    else:
        logger.error("No ingredients column found. Available columns: %s", df.columns)
        return False

    # Determine recipe title column
    if "TranslatedRecipeName" in df.columns:
        title_col = "TranslatedRecipeName"
    elif "RecipeName" in df.columns:
        title_col = "RecipeName"
    else:
        logger.error("No recipe title column found. Available columns: %s", df.columns)
        return False

    # Determine instructions column
    if "TranslatedInstructions" in df.columns:
        instruction_col = "TranslatedInstructions"
    elif "Instructions" in df.columns:
        instruction_col = "Instructions"
    else:
        instruction_col = None

    # Define a cleaning function for ingredients
    def clean_ingredients(text):
        text = re.sub(r"[\[\]'\"]+", "", text)
        text = re.sub(r"[^a-zA-Z0-9, ]", " ", text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    df["ingredients_clean"] = df[ing_col].astype(str).apply(clean_ingredients)

    # Build TF-IDF vectorizer and recipe vectors
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=10000)
    recipe_vectors = vectorizer.fit_transform(df["ingredients_clean"])
    return True

def format_instructions(instructions: str) -> str:
    """Splits instructions text into sentences and formats them as bullet points."""
    steps = re.split(r'(?<=\.)\s+', instructions.strip())
    steps = [step.strip() for step in steps if step.strip()]
    return "\n".join(f"- {step.capitalize()}" for step in steps)

def format_ingredients(ingredients_text: str) -> str:
    """Splits ingredients (assumed comma-separated) and returns them as a bullet list."""
    ingredients = [item.strip() for item in ingredients_text.split(',') if item.strip()]
    return "\n".join(f"- {ing}" for ing in ingredients)

# --- Telegram Bot Handlers (async) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Welcome to the Recipe Bot!\n\nPlease send me the main ingredients (comma-separated) you have."
    )
    return INGREDIENTS

async def get_ingredients(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text.strip().lower()
    
    # Clean user input (same cleaning as for dataset)
    def clean_ingredients(text):
        text = re.sub(r"[\[\]'\"]+", "", text)
        text = re.sub(r"[^a-zA-Z0-9, ]", " ", text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    user_input_clean = clean_ingredients(user_input)
    user_vector = vectorizer.transform([user_input_clean])
    similarities = linear_kernel(user_vector, recipe_vectors).flatten()
    df["cosine_score"] = similarities
    top3 = df.sort_values(by="cosine_score", ascending=False).head(3).reset_index(drop=True)
    if top3.empty:
        await update.message.reply_text("No recipes found matching your ingredients.")
        return ConversationHandler.END
    context.user_data['top3'] = top3
    reply = "🔹 Top 3 Matching Recipes:\n"
    for idx, row in top3.iterrows():
        reply += f"{idx+1}. {row[title_col]} - Score: {row['cosine_score']:.2f}\n"
    reply += "\nPlease reply with the number of the recipe you want details for."
    await update.message.reply_text(reply)
    return CHOICE

async def get_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    top3 = context.user_data.get('top3')
    try:
        choice = int(update.message.text.strip()) - 1
    except ValueError:
        await update.message.reply_text("Please enter a valid number.")
        return CHOICE
    if choice < 0 or choice >= len(top3):
        await update.message.reply_text("Invalid selection. Please try again.")
        return CHOICE
    selected_recipe = top3.iloc[choice]
    reply = f"🍽 Recipe Name: {selected_recipe[title_col]}\n\n"
    reply += "📌 Ingredients:\n" + format_ingredients(selected_recipe[ing_col]) + "\n\n"
    if instruction_col and pd.notna(selected_recipe[instruction_col]):
        reply += "📖 Instructions:\n" + format_instructions(selected_recipe[instruction_col]) + "\n"
    else:
        reply += "📖 Instructions: Not available.\n"
    
    info_lines = []
    if "nutrition" in df.columns and pd.notna(selected_recipe["nutrition"]):
        info_lines.append(f"Nutrition: {selected_recipe['nutrition']}")
    time_info = []
    if "PrepTimeInMins" in df.columns and pd.notna(selected_recipe["PrepTimeInMins"]):
        time_info.append(f"Prep Time: {selected_recipe['PrepTimeInMins']} mins")
    if "CookTimeInMins" in df.columns and pd.notna(selected_recipe["CookTimeInMins"]):
        time_info.append(f"Cook Time: {selected_recipe['CookTimeInMins']} mins")
    if "TotalTimeInMins" in df.columns and pd.notna(selected_recipe["TotalTimeInMins"]):
        time_info.append(f"Total Time: {selected_recipe['TotalTimeInMins']} mins")
    if time_info:
        info_lines.append(" | ".join(time_info))
    
    if info_lines:
        reply += "\nWould you like to see additional info (nutrition/time)? Reply with 'y' or 'n'."
        context.user_data['recipe_reply'] = reply
        return EXTRA_INFO
    else:
        await update.message.reply_text(reply)
        return ConversationHandler.END

async def extra_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    answer = update.message.text.strip().lower()
    reply = context.user_data.get('recipe_reply', "")
    if answer == 'y':
        await update.message.reply_text(reply)
    else:
        reply = reply.split("Would you like to see additional info")[0]
        await update.message.reply_text(reply)
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Cancelled. Send /start to try again.")
    return ConversationHandler.END

def main_bot():
    if not load_dataset():
        print("Dataset loading failed.")
        return
    try:
        app = Application.builder().token("8191338085:AAE2JlCXwd9SRdKqHnkpEL9K4uOauBhrc5s").build()
    except TypeError as e:
        print("Error building Application:", e)
        print("Please downgrade python-telegram-bot to version 20.0 to avoid this error.")
        return
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            INGREDIENTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_ingredients)],
            CHOICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_choice)],
            EXTRA_INFO: [MessageHandler(filters.TEXT & ~filters.COMMAND, extra_info)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    app.add_handler(conv_handler)
    print("Bot is now running...")
    app.run_polling()

if __name__ == "__main__":
    main_bot()


