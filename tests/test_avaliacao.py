"""Testes da matriz, da meta e da galeria da área cinzenta (sem imagens)."""
import numpy as np
import pytest

from avaliacao import casa, contar_casas, galeria_area_cinzenta, meta_atingida, percentual
from fuzzyknn import avaliar_teste_fixo


def test_casas_da_matriz():
    assert [casa(1, 1), casa(1, 0), casa(0, 1), casa(0, 0)] == ["VP", "FN", "FP", "VN"]
    assert contar_casas([1, 1, 1, 0, 0], [1, 0, 1, 1, 0]) == {"VP": 2, "FN": 1, "FP": 1, "VN": 1}


def test_meta_no_teste_fixo():
    assert meta_atingida(73 / 80, 3, 4)[0]          # exemplo do card Matriz: cerca de 91%
    assert meta_atingida(72 / 80, 4, 4)[0]          # 90% exatos também atingem
    assert not meta_atingida(0.95, 7, 0)[0]         # FN acima de 6
    assert not meta_atingida(71 / 80, 2, 2)[0]      # acurácia abaixo de 90%


def test_percentual():
    assert percentual(0.9125) == "91,3%"


def test_avaliacao_e_galeria():
    # Bairro com 1 número por imagem: cachorros em 1, 2 e 3; não cachorros em -1, -2 e -3.
    bairro = {"X": np.array([[1.0], [2.0], [3.0], [-1.0], [-2.0], [-3.0]]),
              "y": np.array([1, 1, 1, 0, 0, 0]),
              "arquivos": ["c1.jpg", "c2.jpg", "c3.jpg", "n1.jpg", "n2.jpg", "n3.jpg"]}
    # Um cachorro, um gato e um lobo (não cachorro) quase no meio do caminho.
    teste = {"X": np.array([[1.0], [-1.0], [0.05]]), "y": np.array([1, 0, 0]),
             "arquivos": ["t_cao.jpg", "t_gato.jpg", "t_lobo.jpg"],
             "datasets": ["voc", "voc", "imagenet"], "categorias": ["dog", "cat", "n02114367"]}
    medidas = avaliar_teste_fixo(bairro, teste, k=5)
    assert medidas["casas"] == {"VP": 1, "FN": 0, "FP": 1, "VN": 1}
    assert medidas["acuracia"] == pytest.approx(2 / 3)

    galeria = galeria_area_cinzenta(medidas["resultados"])
    assert [item["arquivo"] for item in galeria["FP"]] == ["t_lobo.jpg"]   # não cachorro parecido com cachorro
    assert galeria["VP"] == galeria["FN"] == galeria["VN"] == []
    assert galeria["FP"][0]["vizinhas"].split(";") == ["c1.jpg", "n1.jpg", "c2.jpg", "n2.jpg", "c3.jpg"]
