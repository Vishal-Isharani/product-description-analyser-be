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

# Increase the maximum size of HTTP request headers and fields
limit_request_line = 8190  # Default value is sufficient
limit_request_fields = 100  # Default is sufficient
limit_request_field_size = 20480  # 20 MB (in kilobytes, adjust if needed)

raw_env = ["ENV=prod"]
