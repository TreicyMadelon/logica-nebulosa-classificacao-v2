"""Testes do config.py em três partes (proteção 9): nenhuma configuração sobrescreve outra.

Lê o config.py como texto (ast), sem importar nada além dele.
"""
import ast
from collections import Counter
from pathlib import Path

import config

RAIZ = Path(__file__).resolve().parent.parent
TEXTO = (RAIZ / "config.py").read_text(encoding="utf-8")
PARTES = {"COMUM": None, "SÓ DO FUZZY K-NN": "FUZZYKNN_", "SÓ DO SVM": "SVM_"}   # parte → prefixo dos nomes


def limites_das_partes():
    """(linha de início, linha de fim) de cada parte, pelos comentários de início e de fim."""
    linhas = TEXTO.splitlines()
    limites = {}
    for parte in PARTES:
        inicio = [n for n, linha in enumerate(linhas, start=1)
                  if linha.startswith("# ===") and linha.rstrip().endswith(f" {parte} · início")]
        fim = [n for n, linha in enumerate(linhas, start=1)
               if linha.startswith("# ===") and linha.rstrip().endswith(f" {parte} · fim")]
        assert len(inicio) == 1 and len(fim) == 1, f"a parte {parte} precisa de um início e um fim"
        limites[parte] = (inicio[0], fim[0])
    return limites


def nomes_atribuidos():
    """(nome, linha) de cada configuração atribuída no config.py."""
    nomes = []
    for no in ast.parse(TEXTO).body:
        alvos = []
        if isinstance(no, ast.Assign):
            alvos = no.targets
        elif isinstance(no, (ast.AnnAssign, ast.AugAssign)):
            alvos = [no.target]
        for alvo in alvos:
            for nome in ast.walk(alvo):
                if isinstance(nome, ast.Name):
                    nomes.append((nome.id, no.lineno))
    return nomes


def parte_da_linha(linha, limites):
    return [parte for parte, (inicio, fim) in limites.items() if inicio < linha < fim]


def test_tres_partes_em_ordem_e_sem_sobrepor():
    limites = limites_das_partes()
    ordem = [limites[parte] for parte in PARTES]
    for (inicio, fim), (proximo_inicio, _) in zip(ordem, ordem[1:]):
        assert inicio < fim < proximo_inicio
    assert ordem[-1][0] < ordem[-1][1]


def test_toda_configuracao_fica_numa_parte_so():
    limites = limites_das_partes()
    for nome, linha in nomes_atribuidos():
        assert len(parte_da_linha(linha, limites)) == 1, f"{nome} (linha {linha}) fora das três partes"


def test_nenhum_nome_aparece_duas_vezes():
    repetidos = [nome for nome, vezes in Counter(nome for nome, _ in nomes_atribuidos()).items() if vezes > 1]
    assert repetidos == []


def test_cada_modelo_usa_o_proprio_nome():
    limites = limites_das_partes()
    for nome, linha in nomes_atribuidos():
        parte = parte_da_linha(linha, limites)[0]
        prefixo = PARTES[parte]
        if prefixo:
            assert nome.startswith(prefixo), f"{nome} está na parte {parte}"
        else:
            assert not nome.startswith(("FUZZYKNN_", "SVM_")), f"{nome} é de um modelo e está na parte comum"


def test_os_nomes_antigos_sairam_e_os_valores_da_v2_estao_fixos():
    assert not hasattr(config, "K") and not hasattr(config, "PASTA_RESULTADOS")
    assert config.FUZZYKNN_K == 5
    assert config.SVM_C == 1.0 and config.SVM_GAMMA == "scale"
    assert config.FUZZYKNN_PASTA_RESULTADOS.name == "resultados_fuzzyknn"
    assert config.SVM_PASTA_RESULTADOS.name == "resultados_svm"


def test_o_codigo_so_le_o_config():
    """Nenhum módulo do projeto atribui, apaga ou muda (setattr) uma configuração."""
    for arquivo in sorted(RAIZ.glob("*.py")):
        if arquivo.name == "config.py":
            continue
        for no in ast.walk(ast.parse(arquivo.read_text(encoding="utf-8"))):
            alvos = []
            if isinstance(no, ast.Assign):
                alvos = no.targets
            elif isinstance(no, (ast.AnnAssign, ast.AugAssign)):
                alvos = [no.target]
            elif isinstance(no, ast.Delete):
                alvos = no.targets
            for alvo in alvos:
                for parte in ast.walk(alvo):
                    assert not (isinstance(parte, ast.Attribute) and isinstance(parte.value, ast.Name)
                                and parte.value.id == "config"), f"{arquivo.name}, linha {no.lineno}"
            if isinstance(no, ast.Call) and isinstance(no.func, ast.Name) and no.func.id in ("setattr", "delattr"):
                primeiro = no.args[0] if no.args else None
                assert not (isinstance(primeiro, ast.Name) and primeiro.id == "config"), \
                    f"{arquivo.name}, linha {no.lineno}"
