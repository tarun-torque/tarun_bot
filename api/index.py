# api/index.py
from mangum import Mangum
from bot import app  # import your FastAPI instance from bot.py

# Vercel serverless entry point
handler = Mangum(app)
