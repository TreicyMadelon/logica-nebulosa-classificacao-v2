"""SVM: o segundo modelo da V2. Pipeline, probabilidade, imagens que mais pesaram,
avaliação nas 80 e textos com "probabilidade".

O pipeline: make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), ensemble=False)).
- StandardScaler: põe os 1.280 números na mesma escala, aprendida só no bairro (o SVM
  depende da escala). O Fuzzy k-NN fica sem esse passo, para não mudar o plano A.
- SVC: traça a fronteira entre cachorro e não cachorro, com C = 1 e gamma "scale", os
  padrões do scikit-learn, fixos na V2. Sem class_weight, porque o bairro é 160 + 160, e
  sem validação cruzada para escolher C e gamma.
- CalibratedClassifierCV(..., ensemble=False): transforma a pontuação do SVM em
  probabilidade, de 0 a 1, com um SVM só, treinado no bairro inteiro; é dele que saem as
  imagens que mais pesaram. No scikit-learn 1.9, o SVC(probability=True) está
  descontinuado, e esta é a troca que o próprio scikit-learn indica.

O fit é refeito a partir do bairro sempre que precisa, sem arquivo guardado, e não sorteia
nada: a calibração divide o bairro em 5 partes estratificadas pela classe, na ordem do
arquivo. Por isso o código nunca reordena o bairro, e a assinatura do bairro acusa uma
ordem diferente.

O predict não é usado: perto de 0,50, ele discorda da classe com duas casas. A classe e a
faixa saem da probabilidade de cachorro com duas casas, pela régua comum (regua.py), e a
tela mostra esse mesmo número vezes 100, em porcentagem, sem arredondar de novo.

Probabilidade não é pertinência: é a chance de a imagem ser cachorro.
"""
from decimal import Decimal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import avaliacao
import config
import dados
from regua import decidir, grau_na_tela, na_area_cinzenta

MODELO = "svm"
NOME = "SVM"
N_MAIS_PESARAM = 5               # quantas imagens do bairro a tela mostra


# ---------------------------------------------------------------- fit e probabilidade
def novo_pipeline():
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(C=config.SVM_C, gamma=config.SVM_GAMMA), ensemble=False))


def treinar(X_bairro, y_bairro):
    """O fit no bairro, com as linhas na ordem do arquivo (o código nunca reordena)."""
    modelo = novo_pipeline()
    modelo.fit(np.asarray(X_bairro, dtype=np.float64), np.asarray(y_bairro))
    conferir_colunas(modelo)
    return modelo


def conferir_colunas(modelo):
    """A coluna 1 do predict_proba precisa ser a de cachorro (rótulo 1), e a 0, a de não cachorro."""
    if list(modelo.classes_) != [0, 1]:
        raise ValueError(f"As classes do SVM são {list(modelo.classes_)}, e não [0, 1]: "
                         "a coluna 1 do predict_proba não seria a de cachorro.")


def probabilidades(modelo, X):
    """[[p_nao_cachorro, p_cachorro], ...] para cada imagem, pelo predict_proba."""
    conferir_colunas(modelo)
    return modelo.predict_proba(np.asarray(X, dtype=np.float64))


# ---------------------------------------------------------------- imagens que mais pesaram
def _escala_e_svc(modelo):
    """A padronização do pipeline e o SVM único que a calibração treinou no bairro inteiro."""
    escala, calibrado = modelo.steps[0][1], modelo.steps[-1][1]
    svc = calibrado.calibrated_classifiers_[0].estimator
    if svc.kernel != "rbf":
        raise ValueError(f"A conta das imagens que mais pesaram vale para o kernel rbf, não {svc.kernel!r}.")
    return escala, svc


def contribuicoes(modelo, x):
    """Quanto cada vetor de suporte pesou na pontuação do SVM para esta imagem.

    contribuição = dual_coef · exp(−gamma · ‖vetor de suporte − x‖²), com os números já
    padronizados. A soma das contribuições mais o intercept_ é a pontuação do SVM
    (decision_function): positiva puxa para cachorro, negativa para não cachorro.
    Devolve (índices no bairro, contribuições), na ordem dos vetores de suporte.
    """
    escala, svc = _escala_e_svc(modelo)
    x_padronizado = escala.transform(np.asarray(x, dtype=np.float64).reshape(1, -1))
    distancia2 = ((svc.support_vectors_ - x_padronizado) ** 2).sum(axis=1)
    return svc.support_, svc.dual_coef_[0] * np.exp(-svc._gamma * distancia2)


def imagens_que_mais_pesaram(modelo, x, classe, n=N_MAIS_PESARAM):
    """As n imagens do bairro que mais puxaram para a classe vencedora, da que mais pesou
    para a que menos pesou: contribuição positiva (cachorro) quando a classe é cachorro,
    negativa (não cachorro) quando é não cachorro. Os outros vetores de suporte também contam.

    Devolve (índices no bairro, contribuições).
    """
    indices, contrib = contribuicoes(modelo, x)
    lado = contrib if classe == 1 else -contrib            # quanto cada uma puxou para a vencedora
    ordem = [i for i in np.argsort(-lado, kind="stable") if lado[i] > 0][:n]
    return [int(i) for i in indices[ordem]], [float(c) for c in contrib[ordem]]


def partes_da_pontuacao(modelo, x, pesos):
    """A pontuação do SVM separada em duas partes: o que as imagens mostradas somam e o que
    vem do resto (os outros vetores de suporte e o termo fixo, intercept_).
    Devolve (pontuação, soma das mostradas, resto); pontuação = mostradas + resto."""
    _, contrib = contribuicoes(modelo, x)
    _, svc = _escala_e_svc(modelo)
    pontuacao = float(contrib.sum() + svc.intercept_[0])
    mostradas = float(sum(pesos))
    return pontuacao, mostradas, pontuacao - mostradas


def formatar_pontuacao(valor):
    """−0,8523 → "−0,85": duas casas, vírgula e sinal sempre escrito."""
    texto = f"{abs(valor):.2f}".replace(".", ",")
    return ("−" if valor < 0 and texto != "0,00" else "+") + texto


# ---------------------------------------------------------------- textos com "probabilidade"
def porcentagem(p_cachorro):
    """O grau de duas casas da régua vezes 100, sem arredondar de novo: 0,6512 → 0,65 → 65.

    Como sai do mesmo número que decide a classe e a faixa, a porcentagem nunca desmente
    as duas, e o empate aparece como 50%, "Não é cachorro".
    """
    return int((Decimal(str(grau_na_tela(p_cachorro))) * 100).quantize(Decimal("1")))


def formatar_porcentagem(p_cachorro):
    """0,65 → "65%"."""
    return f"{porcentagem(p_cachorro)}%"


def limites_em_porcentagem():
    """Os dois limites da área cinzenta em porcentagem: ("40%", "60%"), tirados do config."""
    minimo, maximo = config.FAIXA_CINZENTA
    return formatar_porcentagem(minimo), formatar_porcentagem(maximo)


def mensagem(classe, p_cachorro):
    """A classe e a probabilidade da classe vencedora: "<classe> · probabilidade <valor>%"."""
    pct = porcentagem(p_cachorro)
    texto = "É cachorro" if classe == 1 else "Não é cachorro"
    return f"{texto} · probabilidade {max(pct, 100 - pct)}%"


def texto_probabilidades(classe, p_cachorro):
    """As duas probabilidades em porcentagem, a vencedora primeiro; somam 100% sempre."""
    nomes = {1: "cachorro", 0: "não cachorro"}
    pct = porcentagem(p_cachorro)
    valores = {1: pct, 0: 100 - pct}
    return f"{nomes[classe]} {valores[classe]}% · {nomes[1 - classe]} {valores[1 - classe]}%"


# ---------------------------------------------------------------- uma imagem e as 80
def classificar(modelo, x_num, n=N_MAIS_PESARAM):
    """Classifica uma imagem (os 1.280 números dela) com o SVM treinado no bairro."""
    p_nao, p_cachorro = (float(valor) for valor in probabilidades(modelo, [x_num])[0])
    classe = decidir(p_cachorro)                     # o if comum, não o predict
    indices, pesos = imagens_que_mais_pesaram(modelo, x_num, classe, n)
    pontuacao, mostradas, resto = partes_da_pontuacao(modelo, x_num, pesos)
    return {
        "p_nao": p_nao,                              # medida crua; a tela usa a de duas casas
        "p_cachorro": p_cachorro,
        "p_cachorro_tela": grau_na_tela(p_cachorro),
        "classe": classe,
        "mensagem": mensagem(classe, p_cachorro),
        "area_cinzenta": na_area_cinzenta(p_cachorro),
        "mais_pesaram": indices,
        "pesos": pesos,
        "pontuacao": pontuacao,                      # soma de tudo; a probabilidade sai dela
        "pontuacao_mostradas": mostradas,
        "pontuacao_resto": resto,
    }


def avaliar_teste_fixo(bairro, teste, modelo=None):
    """As 80 imagens passam pelo SVM treinado no bairro; a matriz compara as respostas com os rótulos."""
    if modelo is None:
        modelo = treinar(bairro["X"], bairro["y"])
    proba = probabilidades(modelo, teste["X"])
    resultados = []
    for i in range(len(teste["arquivos"])):
        indices, _ = imagens_que_mais_pesaram(modelo, teste["X"][i], decidir(proba[i, 1]))
        resultados.append({
            **avaliacao.resultado_da_imagem(teste, i, proba[i, 0], proba[i, 1]),
            "mais_pesaram": ";".join(bairro["arquivos"][j] for j in indices),
            "classes_mais_pesaram": ";".join(dados.classe_do_rotulo(bairro["y"][j]) for j in indices),
        })
    return avaliacao.medir(resultados)
