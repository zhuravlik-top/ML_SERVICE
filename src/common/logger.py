import logging  # Импорт стандартного модуля для логирования в Python

def get_logger(name: str) -> logging.Logger:
    """
    Создаём и возвращаем объект логгера с заданным именем.
    Этот логгер можно использовать в любом модуле вместо print.
    """
    logger = logging.getLogger(name)  # Получаем логгер с именем 'name'.
    
    # Проверяем, есть ли у логгера уже обработчики, чтобы не добавлять их повторно
    if not logger.handlers:
        handler = logging.StreamHandler()  # Создаём обработчик, который выводит логи в консоль
        # Определяем формат сообщений: время, уровень, имя логгера, сообщение
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)  # Присваиваем формат обработчику
        logger.addHandler(handler)       # Добавляем обработчик в логгер
        logger.setLevel(logging.INFO)    # Устанавливаем уровень логирования: INFO и выше
    
    return logger  # Возвращаем готовый логгер для использования
