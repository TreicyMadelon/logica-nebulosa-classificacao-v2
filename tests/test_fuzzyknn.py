"""Testes do Fuzzy k-NN com números inventados (não precisam do TensorFlow nem das imagens).

As bordas da régua comum (duas casas, decisão e faixa) estão em test_regua.py; aqui ficam
as pertinências, as vizinhas, os textos com "pertinência" e a avaliação nas 80.
"""
import numpy as np
import pytest

import rodadas
from fuzzyknn import avaliar_teste_fixo, classificar, fuzzy_knn, mensagem, texto_pertinencias
from regua import decidir

# As mesmas bordas do test_regua.py: (pertinência a cachorro, o que a tela mostra,
# classe esperada, na área cinzenta?).
BORDAS = [
    (0.6048, "0,60", 1, True),
    (0.3965, "0,40", 0, True),
    (0.3993, "0,40", 0, True),
    (0.6050, "0,61", 1, False),
    (0.5049, "0,50", 0, True),
    (0.5050, "0,51", 1, True),
    (0.4950, "0,50", 0, True),
]

# (classe, pertinência a não cachorro, pertinência a cachorro, começo esperado da frase).
# Cachorro vencendo, não cachorro vencendo, empate, dentro e fora da área cinzenta.
CASOS = [
    (1, 0.35, 0.65, "É cachorro"),
    (1, 0.12, 0.88, "É cachorro"),
    (1, 0.3488, 0.6512, "É cachorro"),       # mais de duas casas: a tela arredonda
    (0, 0.55, 0.45, "Não é cachorro"),
    (0, 0.90, 0.10, "Não é cachorro"),
    (0, 0.50, 0.50, "Não é cachorro"),       # empate → não cachorro
    (1, 0.41, 0.59, "É cachorro"),           # dentro da área cinzenta
    (0, 0.93, 0.07, "Não é cachorro"),       # fora da área cinzenta
]


def numero(texto):
    """Lê de volta um número escrito como "0,65"."""
    return float(texto.replace(",", "."))


def test_exemplo_do_desenho_do_plano_a():
    # As 5 vizinhas do desenho do card Grau fuzzy: 3 cães (d = 0,79; 1,00; 1,24) e
    # 2 gatos (d = 0,90; 1,40), mais duas imagens longe, que não podem entrar entre as 5.
    X_treino_num = np.array([[0.79], [1.00], [1.24], [-0.90], [-1.40], [10.0], [-10.0]])
    y_treino = np.array([1, 1, 1, 0, 0, 1, 0])
    pert, viz = fuzzy_knn(X_treino_num, y_treino, [np.array([0.0])], k=5)
    assert pert[0][1] == pytest.approx(0.65, abs=0.005)     # 3,25 ÷ 4,99
    assert pert[0].sum() == pytest.approx(1.0)
    assert list(viz[0]) == [0, 3, 1, 2, 4]                  # da mais próxima para a mais distante


def test_pertinencias_somam_1():
    rng = np.random.default_rng(0)
    X_treino_num = rng.normal(size=(40, 8))
    y_treino = [0, 1] * 20                                   # lista comum também serve
    pert, viz = fuzzy_knn(X_treino_num, y_treino, rng.normal(size=(10, 8)), k=5)
    assert pert.shape == (10, 2)
    assert np.allclose(pert.sum(axis=1), 1.0)
    assert ((pert >= 0) & (pert <= 1)).all()
    assert all(len(v) == 5 for v in viz)


@pytest.mark.parametrize("p_cachorro, na_tela, classe, cinzenta", BORDAS)
def test_bordas_a_tela_nao_desmente_a_classe(p_cachorro, na_tela, classe, cinzenta):
    texto = mensagem(classe, p_cachorro)
    vencedora, perdedora = texto_pertinencias(classe, p_cachorro).split(" · ")
    mostrado = numero(vencedora.split()[-1])
    assert mostrado == numero(texto.split("pertinência")[1])  # título e caixa, o mesmo número
    assert mostrado >= 0.50                                   # a vencedora nunca é a menor
    assert mostrado + numero(perdedora.split()[-1]) == pytest.approx(1.0)


def test_borda_do_empate_em_0_495():
    # 0,495 sobe para 0,50: empate na tela, e o empate fica como não cachorro.
    assert decidir(0.495) == 0
    assert mensagem(0, 0.495) == "Não é cachorro · pertinência 0,50"
    assert texto_pertinencias(0, 0.495) == "não cachorro 0,50 · cachorro 0,50"


@pytest.mark.parametrize("classe, p_nao, p_cachorro, comeco", CASOS)
def test_mensagem_mostra_a_pertinencia_da_classe_vencedora(classe, p_nao, p_cachorro, comeco):
    texto = mensagem(classe, p_cachorro)
    assert texto.startswith(comeco)                       # a classe certa vem na frente
    assert "pertinência" in texto
    assert "%" not in texto                               # pertinência não é porcentagem
    mostrado = numero(texto.split("pertinência")[1])
    assert mostrado == pytest.approx(max(p_nao, p_cachorro), abs=0.005)


@pytest.mark.parametrize("classe, p_nao, p_cachorro, comeco", CASOS)
def test_linha_das_pertinencias_traz_a_vencedora_primeiro(classe, p_nao, p_cachorro, comeco):
    vencedora, perdedora = texto_pertinencias(classe, p_cachorro).split(" · ")
    assert "%" not in vencedora and "%" not in perdedora
    assert vencedora.startswith("cachorro" if classe == 1 else "não cachorro")
    assert perdedora.startswith("não cachorro" if classe == 1 else "cachorro")
    valor_vencedora = numero(vencedora.split()[-1])
    assert valor_vencedora == pytest.approx(max(p_nao, p_cachorro), abs=0.005)
    assert valor_vencedora + numero(perdedora.split()[-1]) == pytest.approx(1.0)


def test_imagem_na_area_cinzenta_continua_com_a_classe():
    # 2 não cachorros a 0,9 e 3 cachorros a 1,0 da imagem nova, mais 2 imagens longe.
    X_treino_num = np.array([[0.9, 0.0], [0.0, 0.9], [-1.0, 0.0], [0.0, -1.0], [0.6, 0.8],
                             [5.0, 5.0], [-5.0, -5.0]])
    y_treino = np.array([0, 0, 1, 1, 1, 1, 0])
    resultado = classificar(X_treino_num, y_treino, np.array([0.0, 0.0]), k=5)
    assert resultado["p_cachorro"] == pytest.approx(3 / (3 + 2 / 0.81), abs=1e-6)   # ≈ 0,55
    assert resultado["classe"] == 1
    assert resultado["area_cinzenta"]                         # o aviso não muda a classe
    assert resultado["mensagem"].startswith("É cachorro")
    assert "pertinência" in resultado["mensagem"] and "%" not in resultado["mensagem"]
    mostrado = numero(resultado["mensagem"].split("pertinência")[1])
    assert mostrado == pytest.approx(resultado["p_cachorro"], abs=0.005)
    assert sorted(resultado["vizinhas"][:2]) == [0, 1]        # as duas mais próximas são não cachorro


def test_avaliacao_nas_80_traz_as_colunas_do_fuzzy_knn():
    bairro = {"X": np.array([[1.0], [2.0], [3.0], [-1.0], [-2.0], [-3.0]]),
              "y": np.array([1, 1, 1, 0, 0, 0]),
              "arquivos": ["c1.jpg", "c2.jpg", "c3.jpg", "n1.jpg", "n2.jpg", "n3.jpg"]}
    teste = {"X": np.array([[1.5], [-1.5]]), "y": np.array([1, 0]),
             "arquivos": ["t_cao.jpg", "t_gato.jpg"], "datasets": ["voc", "voc"], "categorias": ["dog", "cat"]}
    medidas = avaliar_teste_fixo(bairro, teste, k=5)
    for linha in medidas["resultados"]:
        assert list(linha) == rodadas.CAMPOS_RESULTADOS["fuzzyknn"]   # as colunas que a tabela aceita
        assert len(linha["vizinhas"].split(";")) == 5
    assert medidas["casas"] == {"VP": 1, "FN": 0, "FP": 0, "VN": 1}
