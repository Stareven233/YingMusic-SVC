# logger.py
import logging
import os
import threading
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


class Logger:
    _instance = None
    _lock = threading.Lock()  # 确保线程安全

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:  # 双重检查锁
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, LOG_FILE_PATH):
        if self._initialized:
            return
        self._initialized = True

        log_dir = os.path.join(LOG_FILE_PATH, "log")
        level = os.getenv("SERVER_LOG_LEVEL", "DEBUG")
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 配置日志记录器
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(str(level))

        # 创建 TimedRotatingFileHandler
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
        file_handler.setLevel(str(level))
        file_handler.suffix = "%Y-%m-%d.log"

        # 创建格式化器并将其添加到处理器
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(str(level))
        console_handler.setFormatter(formatter)

        # 将处理器添加到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


# 获取单例日志记录器实例的全局方法
def get_logger(LOG_FILE_PATH):
    return Logger(LOG_FILE_PATH).get_logger()
