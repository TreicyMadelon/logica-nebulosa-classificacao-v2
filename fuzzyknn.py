"""Fuzzy k-NN (Keller, Gray e Givens, 1985): pertinências, vizinhas, avaliação nas 80 e textos.

O EfficientNetB0 só dá os números de cada imagem. O Fuzzy k-NN faz o resto: mede a
distância da imagem nova até cada imagem do bairro, ordena, pega as k vizinhas mais
próximas, dá a cada uma o peso 1/d² (mais perto = mais peso) e calcula a pertinência:
pertinência a cachorro = pesos das vizinhas cachorro ÷ soma de todos os pesos.
As duas pertinências somam 1. Pertinência não é probabilidade: diz o quanto a
imagem pertence ao conjunto, não a chance de um evento.

A régua (duas casas, decisão pelo maior grau e área cinzenta) é comum aos dois modelos
e fica no regua.py.
"""
import numpy as np

import avaliacao
import config
import dados
from regua import decidir, formatar_grau, grau_na_tela, na_area_cinzenta

MODELO = "fuzzyknn"
NOME = "Fuzzy k-NN"


def fuzzy_knn(X_treino_num, y_treino, X_novo_num, k=config.FUZZYKNN_K):
    """Devolve as pertinências [não cachorro, cachorro] de cada imagem nova e os
    índices das k vizinhas no bairro (da mais próxima para a mais distante)."""
    X_treino_num = np.asarray(X_treino_num, dtype=np.float64)
    y_treino = np.asarray(y_treino)
    pertinencias = []
    vizinhas = []                                         # quem são as vizinhas de cada imagem
    for x in X_novo_num:
        d = np.linalg.norm(X_treino_num - x, axis=1)     # distância a cada imagem de treino
        viz = np.argsort(d)[:k]                           # as k vizinhas mais próximas
        peso = 1 / (d[viz] ** 2 + 1e-9)                   # mais perto = mais peso
        p_cachorro = peso[y_treino[viz] == 1].sum() / peso.sum()
        pertinencias.append([1 - p_cachorro, p_cachorro])
        vizinhas.append(viz)                              # índices das k vizinhas
    return np.array(pertinencias), vizinhas               # ex.: [[0.35, 0.65], ...] e [[12, 7, 431, 88, 5], ...]


def mensagem(classe, p_cachorro):
    """A classe e a pertinência da classe vencedora: "<classe> · pertinência <valor>".

    O valor sai do mesmo número em duas casas que decidiu a classe, então nunca
    desmente a classe que vem antes dele.

    Pertinência, não porcentagem: o número é contínuo de 0 a 1 e diz o quanto a imagem
    pertence ao conjunto, não a chance de um evento.
    """
    p_cachorro = grau_na_tela(p_cachorro)
    texto = "É cachorro" if classe == 1 else "Não é cachorro"
    return f"{texto} · pertinência {formatar_grau(max(p_cachorro, 1 - p_cachorro))}"


def texto_pertinencias(classe, p_cachorro):
    """As duas pertinências, a vencedora primeiro:
    "<classe vencedora> <valor> · <outra classe> <1 − valor>".

    As duas saem do mesmo número em duas casas: a de não cachorro é 1 menos a de
    cachorro, então somam 1,00 na tela sempre.
    """
    nomes = {1: "cachorro", 0: "não cachorro"}
    p_cachorro = grau_na_tela(p_cachorro)
    valores = {1: p_cachorro, 0: 1 - p_cachorro}
    return (f"{nomes[classe]} {formatar_grau(valores[classe])} · "
            f"{nomes[1 - classe]} {formatar_grau(valores[1 - classe])}")


def classificar(X_treino_num, y_treino, x_num, k=config.FUZZYKNN_K):
    """Classifica uma imagem (os 1.280 números dela) contra o bairro."""
    pert, viz = fuzzy_knn(X_treino_num, y_treino, [x_num], k)
    p_nao, p_cachorro = float(pert[0][0]), float(pert[0][1])
    classe = decidir(p_cachorro)
    return {
        "p_nao": p_nao,                                  # medida crua; a tela usa a de duas casas
        "p_cachorro": p_cachorro,
        "p_cachorro_tela": grau_na_tela(p_cachorro),
        "classe": classe,
        "mensagem": mensagem(classe, p_cachorro),
        "area_cinzenta": na_area_cinzenta(p_cachorro),
        "vizinhas": [int(i) for i in viz[0]],
    }


def avaliar_teste_fixo(bairro, teste, k=config.FUZZYKNN_K):
    """As 80 imagens passam pelo mesmo Fuzzy k-NN; a matriz compara as respostas com os rótulos."""
    X_treino_num, y_treino = bairro["X"], bairro["y"]
    pert, viz = fuzzy_knn(X_treino_num, y_treino, teste["X"], k=k)   # pertinências e vizinhas das 80
    resultados = []
    for i in range(len(teste["arquivos"])):
        resultados.append({
            **avaliacao.resultado_da_imagem(teste, i, pert[i, 0], pert[i, 1]),
            "vizinhas": ";".join(bairro["arquivos"][j] for j in viz[i]),
            "classes_vizinhas": ";".join(dados.classe_do_rotulo(y_treino[j]) for j in viz[i]),
        })
    return avaliacao.medir(resultados)
