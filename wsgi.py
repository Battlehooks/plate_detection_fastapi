from fastapi import FastAPI
from main import app

# WSGI entrypoint for the FastAPI app
# In deployment (gunicorn, uvicorn), you will point to wsgi:app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)