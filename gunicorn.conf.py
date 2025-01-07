import logging
import multiprocessing

workers = 4
bind = "0.0.0.0:8001"
worker_class = "uvicorn.workers.UvicornWorker"

# Logging settings
loglevel = "info"
accesslog = "./logs/access.log"
errorlog = "./logs/error.log"

# Enable print statements in logs
capture_output = True
enable_stdio_inheritance = True

raw_env = ["ENV=prod"]
