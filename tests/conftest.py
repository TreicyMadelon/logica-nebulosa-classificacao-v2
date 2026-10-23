"""Deixa os testes importarem os módulos do projeto (a pasta de cima)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
