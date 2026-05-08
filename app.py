"""
Punto de entrada WSGI para Render: startCommand «gunicorn app:app».
No renombrar telegram-bot.py: se carga por ruta para respetar el guion del archivo.
"""
import importlib.util
import os

_root = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_root, "telegram-bot.py")
_spec = importlib.util.spec_from_file_location("_udesa_telegram_main", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

app = getattr(_mod, "web_app", None)
if app is None:
    raise RuntimeError(
        "PORT no está definido (Render inyecta PORT en Web Services). "
        "Start command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2"
    )
