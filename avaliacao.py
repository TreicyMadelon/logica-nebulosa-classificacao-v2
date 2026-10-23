"""Avaliação no teste fixo, igual para os dois modelos: matriz de confusão, acurácia, meta,
galeria da área cinzenta e comparação imagem a imagem.

Os dois modelos são avaliados só nas 80 imagens do teste fixo, que nunca entram no bairro.
Cada modelo calcula o seu grau de cachorro (fuzzyknn.py e svm.py); daqui em diante a conta
é a mesma para os dois: a régua comum decide a classe e a faixa, e o scikit-learn só conta
os acertos (confusion_matrix e accuracy_score).
"""
from decimal import ROUND_HALF_UP, Decimal

from sklearn.metrics import accuracy_score, confusion_matrix

import config
import dados
from regua import decidir, grau_na_tela, na_area_cinzenta

CASAS = ("VP", "FN", "VN", "FP")

# A parte comum das linhas imagem a imagem; cada modelo acrescenta as suas colunas.
CAMPOS_RESULTADOS_COMUNS = ["arquivo", "classe_real", "dataset", "categoria_original", "p_nao_cachorro",
                            "p_cachorro", "p_cachorro_tela", "resposta", "casa", "area_cinzenta"]

# Cada sigla por extenso, para a legenda da visão admin: (nome, o que aconteceu, acertou?).
LEGENDA_CASAS = {
    "VP": ("verdadeiro positivo", "rótulo real cachorro e classe cachorro", True),
    "FN": ("falso negativo", "rótulo real cachorro e classe não cachorro", False),
    "FP": ("falso positivo", "rótulo real não cachorro e classe cachorro", False),
    "VN": ("verdadeiro negativo", "rótulo real não cachorro e classe não cachorro", True),
}

# O que significa a imagem estar nesta casa E na área cinzenta (a dúvida não muda a classe).
LEITURA_AREA_CINZENTA = {
    "VP": "Caiu na área cinzenta, mas a classe saiu certa: o sistema ficou em dúvida e "
          "ainda assim acertou que é cachorro.",
    "FN": "Caiu na área cinzenta e a classe saiu errada: era cachorro, e o sistema disse que não. "
          "O aviso de dúvida já estava na tela.",
    "VN": "Caiu na área cinzenta, mas a classe saiu certa: o sistema ficou em dúvida e "
          "ainda assim acertou que não é cachorro.",
    "FP": "Caiu na área cinzenta e a classe saiu errada: não era cachorro, e o sistema disse que era. "
          "O aviso de dúvida já estava na tela.",
}

# Os 4 grupos da comparação imagem a imagem, na ordem da tela.
GRUPOS_COMPARACAO = ("os_dois_acertaram", "so_o_primeiro", "so_o_segundo", "os_dois_erraram")


def situacoes(formatar):
    """As 4 situações da galeria da área cinzenta, na ordem do PRD:
    (rótulo real, grau de cachorro, o modelo disse, resultado).

    formatar escreve o grau como o modelo mostra: com duas casas no Fuzzy k-NN
    (regua.formatar_grau → "0,51 a 0,60") e em porcentagem no SVM
    (svm.formatar_porcentagem → "51% a 60%"). Os limites da faixa vêm do config; o 0,50 e
    o 0,51 são fixos porque é onde a classe vira, e com duas casas o degrau acima de 0,50 é 0,51.
    """
    minimo, maximo = config.FAIXA_CINZENTA
    acima = f"{formatar(0.51)} a {formatar(maximo)}"
    abaixo = f"{formatar(minimo)} a {formatar(0.50)}"
    return {
        "VP": ("cachorro", acima, "é cachorro", "acertou (VP)"),
        "FN": ("cachorro", abaixo, "não é cachorro", "errou (FN)"),
        "VN": ("não cachorro", abaixo, "não é cachorro", "acertou (VN)"),
        "FP": ("não cachorro", acima, "é cachorro", "errou (FP)"),
    }


def percentual(valor, casas_decimais=1):
    """0.9125 → "91,3%".

    Arredonda meio-para-cima. No teste fixo de 80 imagens a acurácia é múltipla de
    1,25 ponto, então metade dos valores cai exatamente no empate (73/80 = 91,25%);
    o arredondamento do Python desempataria ora para cima, ora para baixo.

    Decimal(str(valor)), e não Decimal(valor), pela mesma razão do regua.formatar_grau:
    partindo do float, 23/80 e mais duas acurácias desciam uma casa (28,7% no lugar de 28,8%).
    """
    casas = Decimal(10) ** -casas_decimais
    arredondado = (Decimal(str(valor)) * 100).quantize(casas, rounding=ROUND_HALF_UP)
    return f"{arredondado}%".replace(".", ",")


def casa(real, previsto):
    """Em que casa da matriz a imagem cai."""
    if real == 1:
        return "VP" if previsto == 1 else "FN"
    return "FP" if previsto == 1 else "VN"


def contar_casas(y_teste, y_previsto):
    """Matriz de confusão. Ordem do sklearn: [[VN, FP], [FN, VP]] (linha = rótulo, coluna = resposta)."""
    vn, fp, fn, vp = confusion_matrix(y_teste, y_previsto, labels=[0, 1]).ravel()
    return {"VP": int(vp), "FN": int(fn), "FP": int(fp), "VN": int(vn)}


def meta_atingida(acuracia, fn, fp):
    """Meta no teste fixo, a mesma para os dois modelos: acurácia ≥ 90% (72 das 80), com FN ≤ 6 e FP ≤ 6."""
    criterios = [
        (acuracia >= config.META_ACURACIA, f"acurácia ≥ {config.META_ACURACIA:.0%}"),
        (fn <= config.META_MAX_FN, f"FN ≤ {config.META_MAX_FN}"),
        (fp <= config.META_MAX_FP, f"FP ≤ {config.META_MAX_FP}"),
    ]
    return all(ok for ok, _ in criterios), criterios


def resultado_da_imagem(teste, i, p_nao, p_cachorro):
    """A parte comum da linha de uma imagem do teste fixo; cada modelo acrescenta as suas colunas.

    Mesma regra da tela: o decidir arredonda o grau de cachorro para duas casas antes de comparar.
    """
    real, previsto = int(teste["y"][i]), decidir(p_cachorro)   # o rótulo real fica guardado; o modelo não vê
    return {
        "arquivo": teste["arquivos"][i],
        "classe_real": dados.classe_do_rotulo(real),
        "dataset": teste["datasets"][i],
        "categoria_original": teste["categorias"][i],
        "p_nao_cachorro": f"{p_nao:.4f}",                        # medida crua, para conferir depois
        "p_cachorro": f"{p_cachorro:.4f}",
        "p_cachorro_tela": f"{grau_na_tela(p_cachorro):.2f}",    # o que decidiu classe e faixa
        "resposta": dados.classe_do_rotulo(previsto),
        "casa": casa(real, previsto),
        "area_cinzenta": "sim" if na_area_cinzenta(p_cachorro) else "não",
    }


def medir(resultados):
    """A matriz e a acurácia, tiradas das linhas imagem a imagem: (VP + VN) ÷ total."""
    y_teste = [dados.rotulo(resultado["classe_real"]) for resultado in resultados]
    y_previsto = [dados.rotulo(resultado["resposta"]) for resultado in resultados]
    return {"resultados": resultados,
            "casas": contar_casas(y_teste, y_previsto),
            "acuracia": float(accuracy_score(y_teste, y_previsto))}


def contar_area_cinzenta(resultados):
    """Quantas imagens do teste fixo caíram na área cinzenta."""
    return sum(1 for resultado in resultados if resultado["area_cinzenta"] == "sim")


def galeria_area_cinzenta(resultados):
    """As imagens de teste que caíram na faixa, separadas nas 4 situações (os erros também contam na matriz)."""
    galeria = {situacao: [] for situacao in CASAS}
    for resultado in resultados:
        if resultado["area_cinzenta"] == "sim":
            galeria[resultado["casa"]].append(resultado)
    for lista in galeria.values():
        lista.sort(key=lambda resultado: float(resultado["p_cachorro"]))
    return galeria


# ---------------------------------------------------------------- comparação dos dois modelos
def acertou(resultado):
    return resultado["casa"] in ("VP", "VN")


def comparar_imagens(resultados_1, resultados_2):
    """Imagem a imagem, nas mesmas 80: os dois acertaram, só o primeiro, só o segundo, os dois erraram.

    Cada grupo guarda os pares (linha do primeiro, linha do segundo). As duas listas
    precisam ter as mesmas imagens, na mesma ordem.
    """
    if [r["arquivo"] for r in resultados_1] != [r["arquivo"] for r in resultados_2]:
        raise ValueError("Os dois resultados não listam as mesmas imagens, na mesma ordem.")
    grupos = {grupo: [] for grupo in GRUPOS_COMPARACAO}
    for linha_1, linha_2 in zip(resultados_1, resultados_2):
        acertou_1, acertou_2 = acertou(linha_1), acertou(linha_2)
        if acertou_1 and acertou_2:
            grupo = "os_dois_acertaram"
        elif acertou_1:
            grupo = "so_o_primeiro"
        elif acertou_2:
            grupo = "so_o_segundo"
        else:
            grupo = "os_dois_erraram"
        grupos[grupo].append((linha_1, linha_2))
    return grupos


def quem_foi_melhor(acertos_1, acertos_2, area_cinzenta_1, area_cinzenta_2):
    """Ganha quem acerta mais imagens nas mesmas 80; não há desempate por FP ou FN.

    Devolve {"vencedor": 1, 2 ou None (empate), "diferenca": em imagens,
    "menos_area_cinzenta": 1, 2 ou None}. A área cinzenta só entra como nota no empate
    (None quando ela também empata) e não muda o resultado: ela mostra dúvida, não acerto.
    """
    if acertos_1 != acertos_2:
        return {"vencedor": 1 if acertos_1 > acertos_2 else 2,
                "diferenca": abs(acertos_1 - acertos_2), "menos_area_cinzenta": None}
    menos = None
    if area_cinzenta_1 != area_cinzenta_2:
        menos = 1 if area_cinzenta_1 < area_cinzenta_2 else 2
    return {"vencedor": None, "diferenca": 0, "menos_area_cinzenta": menos}
