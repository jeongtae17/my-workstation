# -*- coding: utf-8 -*-
import sys
import io
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

os.environ["PYTHONIOENCODING"] = "utf-8"

from stock_analyzer.ui.app import App

if __name__ == "__main__":
    app = App()
    app.mainloop()
