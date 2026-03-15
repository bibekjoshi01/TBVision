"""Legacy adapter so uvicorn can load the app."""
from .app.main import create_app

app = create_app()
