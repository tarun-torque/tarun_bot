from mangum import Mangum
from bot import app  # assuming you have FastAPI instance called 'app' in bot.py

# serverless handler required by Vercel
handler = Mangum(app)
