"""A régua comum aos dois modelos: duas casas, decisão pelo maior grau e área cinzenta.

O grau é a pertinência no Fuzzy k-NN e a probabilidade no SVM. Os dois passam pela
mesma régua, definida uma vez só: o grau de cachorro é arredondado para duas casas
(o meio sobe), e desse único número saem a classe, a faixa, a cor da caixa e o que a
tela mostra. O grau de não cachorro é 1 menos esse valor.

Cada modelo escreve o número do seu jeito: o Fuzzy k-NN com duas casas ("0,65"), o SVM
em porcentagem ("65%", o mesmo número vezes 100, sem arredondar de novo).
"""
from decimal import ROUND_HALF_UP, Decimal

import config


def formatar_grau(valor):
    """Duas casas decimais e vírgula, com o arredondamento convencional: o meio sobe.

    O f"{v:.2f}" do Python não serve aqui. Ele arredonda sobre o que o float guarda, e
    0,605 em binário é 0,60499999…, um tiquinho abaixo do meio: desceria para 0,60.
    Decimal(str(valor)) lê o número como ele é escrito, e ROUND_HALF_UP sobe o empate.
    """
    arredondado = Decimal(str(valor)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{arredondado}".replace(".", ",")


def grau_na_tela(p_cachorro):
    """O grau de cachorro com duas casas: o único número de que tudo sai.

    A classe, a faixa, a cor da caixa e o que a tela mostra saem daqui. O grau de não
    cachorro é 1 menos este valor — os dois nunca são arredondados em separado, senão
    poderiam não somar 1,00 (ou 100%) na tela.
    """
    return float(formatar_grau(float(p_cachorro)).replace(",", "."))


def decidir(p_cachorro):
    """Classe = a de maior grau, por um if comum (sem corte).

    Com o grau de cachorro em duas casas, o de não cachorro é 1 menos ele, então
    comparar os dois é comparar com 0,50. Exatamente 0,50 fica como não cachorro: só é
    cachorro quando o grau de cachorro é o maior.
    """
    p_cachorro = grau_na_tela(p_cachorro)
    if p_cachorro > 1 - p_cachorro:
        return 1        # é cachorro
    return 0            # não é cachorro


def na_area_cinzenta(p_cachorro):
    """Faixa do professor, lida de config.FAIXA_CINZENTA (hoje grau de cachorro de 0,40 a
    0,60, limites inclusive, com o grau da classe vencedora de 0,50 a 0,60, com duas casas).

    A faixa sai do mesmo número em duas casas que decide a classe: um grau de 0,3965
    conta como 0,40 e entra na faixa. Só marca a dúvida; não muda a classe.
    """
    minimo, maximo = config.FAIXA_CINZENTA
    return minimo <= grau_na_tela(p_cachorro) <= maximo


def limites_area_cinzenta():
    """Os dois limites da faixa, com duas casas: ("0,40", "0,60").

    Todo texto que cita a faixa sai daqui, e daqui sai de config.FAIXA_CINZENTA: mudar a
    faixa no config muda a classificação e os textos juntos, sem sobrar número escrito à mão.
    """
    minimo, maximo = config.FAIXA_CINZENTA
    return formatar_grau(minimo), formatar_grau(maximo)
