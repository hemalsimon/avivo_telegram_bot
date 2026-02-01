import os
import logging
from dotenv import load_dotenv
from telegram import Update, constants, error
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import asyncio

# Load environment variables
load_dotenv()

from rag import RAGSystem

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize RAG System
rag = RAGSystem()

# In-memory history: {chat_id: ["User: ...", "Bot: ..."]}
user_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hello! I am your Finance RAG Bot.\nUse /ask <question> to query our knowledge base.\nUse /help for more info."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
Commands:
/start - Start the bot
/ask <your question> - Ask a question about our finance policies
/summarize - Summarize our conversation so far
/image - (Placeholder) Image description
/help - Show this message
    """
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=help_text
    )

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please provide a question after /ask. Example: /ask What is the risk policy?"
        )
        return

    status_msg = None
    for attempt in range(3):
        try:
            # Send initial status message
            status_msg = await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Thinking..."
            )
            break # Success
        except error.NetworkError:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            else:
                return # Failed to even send "Thinking..."

    try:
        # Get History
        cid = update.effective_chat.id
        history_list = user_history.get(cid, [])
        # Keep last 3 interactions (6 messages)
        history_str = "\n".join(history_list[-6:])

        # Call RAG System
        answer, sources = rag.generate_answer(query, history=history_str)
        
        # Call RAG System
        answer, sources = rag.generate_answer(query, history=history_str)
        
        # Update History
        history_list.append(f"User: {query}")
        history_list.append(f"Bot: {answer}")
        user_history[cid] = history_list

        # Attach Source Citations
        if sources:
            source_list = ", ".join([f"`{s}`" for s in sources])
            answer += f"\n\n📚 *Sources:* {source_list}"

        # Markdown Sanitization (Basic)
        answer = answer.replace("**", "*").replace("__", "_")
        
        # Edit "Thinking..." to actual answer with Markdown parsing
        for attempt in range(3):
            try:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=status_msg.message_id,
                    text=answer,
                    parse_mode=constants.ParseMode.MARKDOWN
                )
                break
            except error.NetworkError:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                else:
                    # Fallback to console log if we can't edit
                    print(f"Failed to edit message for chat {update.effective_chat.id}")

    except Exception as e:
        # Try to report error to user
        try:
            if status_msg:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=status_msg.message_id,
                    text=f"An error occurred: {str(e)}"
                )
        except:
             pass

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    history_list = user_history.get(cid, [])
    
    if not history_list:
        await context.bot.send_message(chat_id=cid, text="No conversation history to summarize yet.")
        return

    status_msg = await context.bot.send_message(chat_id=cid, text="Summarizing...")
    
    history_text = "\n".join(history_list)
    prompt = f"Summarize the following conversation in 3 bullet points:\n\n{history_text}"
    
    try:
        # We can re-use the RAG LLM model directly (bypassing RAG logic)
        import ollama
        response = ollama.chat(model=rag.llm_model, messages=[{'role': 'user', 'content': prompt}])
        summary = response['message']['content']
        
        await context.bot.edit_message_text(
            chat_id=cid,
            message_id=status_msg.message_id,
            text=f"📝 *Summary:*\n{summary}",
            parse_mode=constants.ParseMode.MARKDOWN
        )
    except Exception as e:
        await context.bot.edit_message_text(chat_id=cid, message_id=status_msg.message_id, text=f"Error: {e}")

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Image description is not yet implemented in this Mini-RAG version."
    )

if __name__ == '__main__':
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TOKEN:
        print("Error: TELEGRAM_TOKEN not found in .env file.")
        exit(1)

    # Increase timeouts for better stability on slower connections
    application = ApplicationBuilder().token(TOKEN).connect_timeout(120).read_timeout(120).write_timeout(120).build()


    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('ask', ask))
    application.add_handler(CommandHandler('summarize', summarize_command))
    application.add_handler(CommandHandler('image', image_command))

    print("Bot is running...")
    application.run_polling()
