"""Testes da régua comum: duas casas, decisão pelo maior grau e área cinzenta.

São as bordas de antes do SVM, sem mudar nenhum valor esperado: só os nomes das funções
mudaram (formatar_pertinencia → formatar_grau, pertinencia_na_tela → grau_na_tela),
porque a régua agora serve para a pertinência e para a probabilidade.
"""
import pytest

from regua import decidir, formatar_grau, grau_na_tela, limites_area_cinzenta, na_area_cinzenta

# As bordas: a classe e a faixa saem do grau de cachorro com duas casas.
# (grau de cachorro, o que a tela mostra, classe esperada, na área cinzenta?)
BORDAS = [
    (0.6048, "0,60", 1, True),    # arredonda para baixo e entra na faixa
    (0.3965, "0,40", 0, True),    # arredonda para cima e entra na faixa
    (0.3993, "0,40", 0, True),
    (0.6050, "0,61", 1, False),   # o meio sobe e sai da faixa: caixa verde
    (0.5049, "0,50", 0, True),    # desce até o empate: não cachorro
    (0.5050, "0,51", 1, True),    # o meio sobe: cachorro
    (0.4950, "0,50", 0, True),    # sobe até o empate: não cachorro
]


def test_decisao_pelo_maior_grau():
    assert decidir(0.65) == 1
    assert decidir(0.45) == 0
    assert decidir(0.50) == 0                                # exatamente 0,50 → não cachorro


def test_area_cinzenta_inclui_os_limites():
    assert na_area_cinzenta(0.40) and na_area_cinzenta(0.50) and na_area_cinzenta(0.60)
    assert not na_area_cinzenta(0.39) and not na_area_cinzenta(0.61)


@pytest.mark.parametrize("p_cachorro, na_tela, classe, cinzenta", BORDAS)
def test_bordas_decidem_pelo_grau_de_duas_casas(p_cachorro, na_tela, classe, cinzenta):
    assert formatar_grau(p_cachorro) == na_tela
    assert grau_na_tela(p_cachorro) == float(na_tela.replace(",", "."))
    assert decidir(p_cachorro) == classe                     # a classe sai do arredondado
    assert na_area_cinzenta(p_cachorro) is cinzenta          # a faixa também


def test_borda_do_empate_em_0_495():
    assert decidir(0.495) == 0                               # 0,495 sobe para 0,50: empate → não cachorro


def test_formatar_grau_usa_duas_casas_e_virgula():
    assert formatar_grau(0.6512) == "0,65"
    assert formatar_grau(0.0) == "0,00"
    assert formatar_grau(1.0) == "1,00"
    assert "." not in formatar_grau(0.5)


@pytest.mark.parametrize("valor, esperado", [
    (0.6049, "0,60"),      # abaixo do meio: desce
    (0.6050, "0,61"),      # o meio sobe, mesmo com 0,605 valendo 0,60499… em binário
    (0.6051, "0,61"),
    (0.5950, "0,60"),
    (0.5049, "0,50"),
    (0.5050, "0,51"),
])
def test_formatar_grau_arredonda_o_meio_para_cima(valor, esperado):
    assert formatar_grau(valor) == esperado


def test_limites_da_faixa_saem_do_config():
    assert limites_area_cinzenta() == ("0,40", "0,60")
