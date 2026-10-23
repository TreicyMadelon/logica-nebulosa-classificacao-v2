"""Testes do SVM com números inventados (não precisam do TensorFlow nem das imagens):
o pipeline do plano, a coluna de cachorro, a classe e a faixa pela régua comum, a
porcentagem nas bordas, os textos com "probabilidade" e a conta das imagens que mais pesaram.
"""
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import rodadas
import svm
from regua import decidir, grau_na_tela, na_area_cinzenta


def inventar_bairro(n_por_classe=40, dimensoes=8, semente=0):
    """Dois grupos de números, cachorro em volta de +0,7 e não cachorro em volta de −0,7,
    intercalados na ordem do "arquivo"."""
    rng = np.random.default_rng(semente)
    X = np.empty((2 * n_por_classe, dimensoes))
    X[0::2] = rng.normal(0.7, 1.0, (n_por_classe, dimensoes))    # cachorro
    X[1::2] = rng.normal(-0.7, 1.0, (n_por_classe, dimensoes))   # não cachorro
    y = np.array([1, 0] * n_por_classe)
    arquivos = [f"{'c' if rotulo == 1 else 'n'}{i:03d}.jpg" for i, rotulo in enumerate(y)]
    return {"X": X.astype(np.float32), "y": y, "arquivos": arquivos, "extrator": "efficientnetb0"}


def inventar_teste(n=20, dimensoes=8, semente=1):
    """Imagens de teste espalhadas, inclusive perto da fronteira (para cair na área cinzenta)."""
    rng = np.random.default_rng(semente)
    y = np.array([1, 0] * (n // 2))
    centro = np.where(y == 1, 0.3, -0.3)[:, None]
    X = rng.normal(centro, 1.0, (n, dimensoes)).astype(np.float32)
    return {"X": X, "y": y, "arquivos": [f"t{i:02d}.jpg" for i in range(n)],
            "datasets": ["voc"] * n, "categorias": ["dog" if r == 1 else "cat" for r in y]}


@pytest.fixture(scope="module")
def bairro():
    return inventar_bairro()


@pytest.fixture(scope="module")
def modelo(bairro):
    return svm.treinar(bairro["X"], bairro["y"])


# ---------------------------------------------------------------- o pipeline do plano
def test_pipeline_e_o_do_plano(modelo):
    escala, calibrado = modelo.steps[0][1], modelo.steps[-1][1]
    assert len(modelo.steps) == 2
    assert isinstance(escala, StandardScaler)
    assert isinstance(calibrado, CalibratedClassifierCV) and calibrado.ensemble is False
    base = calibrado.estimator
    assert isinstance(base, SVC)
    assert base.C == 1.0 and base.gamma == "scale" and base.kernel == "rbf"   # os padrões, fixos na V2
    assert base.class_weight is None                    # o bairro é 160 + 160
    assert base.random_state is None
    assert base.probability is not True              # sem o probability=True, descontinuado no 1.9
    assert len(calibrado.calibrated_classifiers_) == 1  # um SVM só, treinado no bairro inteiro


def test_coluna_1_e_a_de_cachorro(modelo):
    assert list(modelo.classes_) == [0, 1]

    class ModeloTrocado:
        classes_ = np.array(["cachorro", "nao_cachorro"])

    with pytest.raises(ValueError):
        svm.conferir_colunas(ModeloTrocado())


def test_fit_nao_sorteia(bairro, modelo):
    # Mesmo bairro, na mesma ordem → as mesmas probabilidades, sem random_state.
    de_novo = svm.treinar(bairro["X"], bairro["y"])
    teste = inventar_teste()
    assert np.array_equal(svm.probabilidades(modelo, teste["X"]), svm.probabilidades(de_novo, teste["X"]))


def test_probabilidades_somam_1(modelo):
    proba = svm.probabilidades(modelo, inventar_teste()["X"])
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------- classe e faixa pela régua comum
def test_classe_e_faixa_saem_da_probabilidade_com_duas_casas(modelo):
    teste = inventar_teste()
    for x in teste["X"]:
        resultado = svm.classificar(modelo, x)
        p = resultado["p_cachorro"]
        assert resultado["classe"] == decidir(p)
        assert resultado["area_cinzenta"] is na_area_cinzenta(p)
        assert resultado["p_cachorro_tela"] == grau_na_tela(p)
        assert resultado["mensagem"] == svm.mensagem(resultado["classe"], p)
        assert len(resultado["mais_pesaram"]) == 5


def test_o_predict_nao_e_usado(bairro):
    modelo = svm.treinar(bairro["X"], bairro["y"])

    def proibido(*args, **kwargs):
        raise AssertionError("o predict não pode ser usado: a classe sai do if comum")

    modelo.predict = proibido
    teste = inventar_teste()
    svm.classificar(modelo, teste["X"][0])
    svm.avaliar_teste_fixo(bairro, teste, modelo=modelo)


# ---------------------------------------------------------------- porcentagem nas bordas
# (probabilidade de cachorro, porcentagem na tela, classe, na área cinzenta?, mensagem)
BORDAS = [
    (0.6048, 60, 1, True, "É cachorro · probabilidade 60%"),
    (0.3965, 40, 0, True, "Não é cachorro · probabilidade 60%"),
    (0.3993, 40, 0, True, "Não é cachorro · probabilidade 60%"),
    (0.6050, 61, 1, False, "É cachorro · probabilidade 61%"),
    (0.5049, 50, 0, True, "Não é cachorro · probabilidade 50%"),     # empate → não cachorro
    (0.5050, 51, 1, True, "É cachorro · probabilidade 51%"),
    (0.4950, 50, 0, True, "Não é cachorro · probabilidade 50%"),
    (0.395, 40, 0, True, "Não é cachorro · probabilidade 60%"),      # sobe para 0,40: entra na faixa
    (0.605, 61, 1, False, "É cachorro · probabilidade 61%"),         # sobe para 0,61: sai da faixa
    (0.505, 51, 1, True, "É cachorro · probabilidade 51%"),
]


@pytest.mark.parametrize("p_cachorro, pct, classe, cinzenta, frase", BORDAS)
def test_porcentagem_nas_bordas(p_cachorro, pct, classe, cinzenta, frase):
    assert svm.porcentagem(p_cachorro) == pct
    assert decidir(p_cachorro) == classe
    assert na_area_cinzenta(p_cachorro) is cinzenta
    assert svm.mensagem(classe, p_cachorro) == frase


def test_porcentagem_nunca_desmente_a_classe_nem_a_faixa():
    # Todos os valores de 0 a 1, de 0,0001 em 0,0001.
    for i in range(10001):
        p = i / 10000
        pct, classe = svm.porcentagem(p), decidir(p)
        assert (pct > 50) == (classe == 1)                   # 50% é empate → não cachorro
        assert (40 <= pct <= 60) == na_area_cinzenta(p)
        assert pct == round(grau_na_tela(p) * 100)           # o número de duas casas vezes 100


def test_textos_falam_de_probabilidade_em_porcentagem():
    assert svm.limites_em_porcentagem() == ("40%", "60%")
    for p in (0.07, 0.3965, 0.5, 0.5049, 0.55, 0.6512, 0.93):
        classe = decidir(p)
        frase = svm.mensagem(classe, p)
        assert frase.startswith("É cachorro" if classe == 1 else "Não é cachorro")
        assert "probabilidade" in frase and "pertinência" not in frase and frase.endswith("%")
        vencedora, perdedora = svm.texto_probabilidades(classe, p).split(" · ")
        assert vencedora.startswith("cachorro" if classe == 1 else "não cachorro")
        valor_vencedora, valor_perdedora = int(vencedora.split()[-1][:-1]), int(perdedora.split()[-1][:-1])
        assert valor_vencedora + valor_perdedora == 100          # somam 100% sempre
        assert valor_vencedora >= 50
        assert frase.endswith(f"{valor_vencedora}%")             # título e caixa, o mesmo número


# ---------------------------------------------------------------- as imagens que mais pesaram
def test_a_conta_das_imagens_bate_com_a_pontuacao_do_svm(bairro, modelo):
    escala = modelo.steps[0][1]
    base = modelo.steps[-1][1].calibrated_classifiers_[0].estimator
    for x in inventar_teste()["X"]:
        _, contrib = svm.contribuicoes(modelo, x)
        pontuacao = base.decision_function(escala.transform(x.reshape(1, -1).astype(np.float64)))[0]
        assert contrib.sum() + base.intercept_[0] == pytest.approx(pontuacao, abs=1e-9)


def test_o_sinal_da_contribuicao_e_a_classe(bairro, modelo):
    # Contribuição positiva puxa para cachorro: os vetores de suporte de cachorro têm dual_coef > 0.
    base = modelo.steps[-1][1].calibrated_classifiers_[0].estimator
    assert np.array_equal(base.dual_coef_[0] > 0, bairro["y"][base.support_] == 1)


def test_mais_pesaram_em_ordem_e_dentro_do_bairro(bairro, modelo):
    escala = modelo.steps[0][1]
    base = modelo.steps[-1][1].calibrated_classifiers_[0].estimator
    for x in inventar_teste()["X"]:
        for classe in (0, 1):
            indices, pesos = svm.imagens_que_mais_pesaram(modelo, x, classe)
            _, contrib = svm.contribuicoes(modelo, x)
            lado = contrib if classe == 1 else -contrib
            puxadas = [p if classe == 1 else -p for p in pesos]
            assert len(indices) == 5
            assert all(p > 0 for p in puxadas)                               # todas puxam para a vencedora
            assert all(int(bairro["y"][i]) == classe for i in indices)        # e são dessa classe
            assert puxadas == sorted(puxadas, reverse=True)                  # da que mais pesou
            assert puxadas[-1] >= np.sort(lado)[-5]                          # nenhuma de fora puxou mais
            # Os índices apontam para as linhas do bairro, na ordem do arquivo: o código não reordena.
            posicoes = [list(base.support_).index(i) for i in indices]
            assert np.allclose(escala.transform(bairro["X"][indices].astype(np.float64)),
                               base.support_vectors_[posicoes])


def test_as_5_e_o_resto_fecham_a_pontuacao(modelo):
    for x in inventar_teste()["X"]:
        resultado = svm.classificar(modelo, x)
        assert resultado["pontuacao_mostradas"] + resultado["pontuacao_resto"] == pytest.approx(resultado["pontuacao"])
        assert resultado["pontuacao_mostradas"] == pytest.approx(sum(resultado["pesos"]))
        assert (resultado["pontuacao_mostradas"] > 0) == (resultado["classe"] == 1)
    assert svm.formatar_pontuacao(-0.8523) == "−0,85"
    assert svm.formatar_pontuacao(1.2) == "+1,20"
    assert svm.formatar_pontuacao(-0.001) == "+0,00"


# ---------------------------------------------------------------- avaliação nas 80
def test_avaliacao_nas_80_traz_as_colunas_do_svm(bairro, modelo):
    teste = inventar_teste()
    medidas = svm.avaliar_teste_fixo(bairro, teste, modelo=modelo)
    assert sum(medidas["casas"].values()) == len(teste["arquivos"])
    for linha in medidas["resultados"]:
        assert list(linha) == rodadas.CAMPOS_RESULTADOS["svm"]      # as colunas que a tabela aceita
        nomes = linha["mais_pesaram"].split(";")
        assert len(nomes) == 5 and set(nomes) <= set(bairro["arquivos"])
        assert linha["classes_mais_pesaram"].split(";") == [
            "cachorro" if bairro["y"][bairro["arquivos"].index(nome)] == 1 else "nao_cachorro" for nome in nomes]
        assert linha["resposta"] == ("cachorro" if decidir(float(linha["p_cachorro_tela"])) == 1 else "nao_cachorro")
    acertos = medidas["casas"]["VP"] + medidas["casas"]["VN"]
    assert medidas["acuracia"] == pytest.approx(acertos / len(teste["arquivos"]))


def test_avaliacao_sem_modelo_treina_no_bairro(bairro, modelo):
    teste = inventar_teste()
    assert svm.avaliar_teste_fixo(bairro, teste) == svm.avaliar_teste_fixo(bairro, teste, modelo=modelo)
