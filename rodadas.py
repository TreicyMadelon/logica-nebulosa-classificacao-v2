"""Rodadas dos dois modelos: caminhos, gravar e ler, e as travas contra salvar errado.

O que é de um modelo leva o nome dele, e a rodada tem o mesmo número nos dois:
    resultados_fuzzyknn/rodadas_fuzzyknn.csv                      uma linha por rodada
    resultados_fuzzyknn/rodada_1_fuzzyknn/resultados_fuzzyknn.csv imagem por imagem
    resultados_fuzzyknn/rodada_1_fuzzyknn/FP_fuzzyknn/p0.55_<foto>_fuzzyknn.<extensão>
e o mesmo com svm.

As proteções que moram aqui (seção 6 do plano), cada uma com teste em tests/test_rodadas.py:
 1. os caminhos de cada modelo saem só de caminhos(modelo);
 2. trava de pasta: nada é gravado fora da pasta do modelo, nem com o nome do outro;
 3. trava de tabela: uma linha (ou uma tabela) de um modelo no lugar da do outro é recusada;
 4. trava de rodada: substituir uma rodada só com pedido explícito, e nas duas tabelas;
 5. grava no fim: tudo é conferido antes da primeira gravação, e as duas tabelas têm as
    mesmas rodadas;
 6. as mesmas 80 imagens nos dois;
 7. a home avisa se o modelo em uso não foi avaliado, pela assinatura do bairro;
12. ordem do bairro: a assinatura muda se a ordem mudar;
13. o comparativo só compara a mesma rodada, com as mesmas 80 e a mesma assinatura.
E a regra das rodadas: o modelo que não mudou repete os números da rodada anterior; se
não repetir, nada é gravado e a mensagem mostra a diferença.
"""
import csv
import hashlib
import io
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

import config
import dados
from avaliacao import CAMPOS_RESULTADOS_COMUNS, contar_area_cinzenta, meta_atingida

MODELOS = ("fuzzyknn", "svm")
NOMES = {"fuzzyknn": "Fuzzy k-NN", "svm": "SVM"}

# Colunas das tabelas de rodadas: as comuns, com as próprias de cada modelo depois do extrator.
CAMPOS_PROPRIOS_RODADAS = {"fuzzyknn": ["k"], "svm": ["C", "gamma"]}
CAMPOS_RODADAS = {
    modelo: ["rodada", "o_que_mudou", "extrator", *proprios, "bairro", "assinatura_bairro",
             "VP", "FN", "FP", "VN", "acuracia", "area_cinzenta", "meta", "data"]
    for modelo, proprios in CAMPOS_PROPRIOS_RODADAS.items()
}

# Colunas dos resultados imagem por imagem: as comuns, mais as 5 imagens do bairro que
# explicam o grau (as vizinhas, no Fuzzy k-NN; as que mais pesaram, no SVM) e as classes delas.
CAMPOS_PROPRIOS_RESULTADOS = {"fuzzyknn": ["vizinhas", "classes_vizinhas"],
                              "svm": ["mais_pesaram", "classes_mais_pesaram"]}
CAMPOS_RESULTADOS = {modelo: CAMPOS_RESULTADOS_COMUNS + proprios
                     for modelo, proprios in CAMPOS_PROPRIOS_RESULTADOS.items()}

# Os números que o modelo que não mudou precisa repetir.
NUMEROS_DA_RODADA = ("VP", "FN", "FP", "VN", "acuracia", "area_cinzenta")


class Trava(Exception):
    """Uma proteção barrou a operação. Quando ela barra uma gravação, nada foi gravado."""


def conferir_modelo(modelo):
    if modelo not in MODELOS:
        raise Trava(f"Modelo desconhecido: {modelo!r}. Os modelos são: {', '.join(MODELOS)}.")


# ---------------------------------------------------------------- 1. caminhos de uma função só
@dataclass(frozen=True)
class Caminhos:
    """Os caminhos de um modelo. Use caminhos(modelo), que confere a pasta antes."""
    modelo: str
    pasta: Path

    @property
    def tabela(self):
        return self.pasta / f"rodadas_{self.modelo}.csv"

    def pasta_rodada(self, numero):
        return self.pasta / f"rodada_{int(numero)}_{self.modelo}"

    def resultados(self, numero):
        return self.pasta_rodada(numero) / f"resultados_{self.modelo}.csv"

    def pasta_erros(self, numero, casa):
        if casa not in ("FP", "FN"):
            raise Trava(f"Só as imagens de FP e FN são copiadas, não as de {casa}.")
        return self.pasta_rodada(numero) / f"{casa}_{self.modelo}"

    def copia_imagem(self, numero, casa, p_cachorro_tela, arquivo):
        """p<valor>_<foto>_<modelo>.<extensão>. O valor é o grau de cachorro com duas casas,
        o que já decidiu a classe e a faixa, sem arredondar de novo; a foto mantém o nome."""
        foto = Path(arquivo)
        return self.pasta_erros(numero, casa) / f"p{p_cachorro_tela}_{foto.stem}_{self.modelo}{foto.suffix}"


def _pasta_do_config(modelo):
    """Lida do config na hora (e não no import), para valer sempre o que está lá."""
    return Path({"fuzzyknn": config.FUZZYKNN_PASTA_RESULTADOS,
                 "svm": config.SVM_PASTA_RESULTADOS}[modelo])


def caminhos(modelo):
    """Todos os caminhos de um modelo saem daqui (proteção 1).

    Confere que a pasta leva o nome do modelo (resultados_<modelo>) e que as pastas dos
    dois modelos são separadas: nenhuma fica dentro da outra.
    """
    conferir_modelo(modelo)
    pasta = _pasta_do_config(modelo)
    if pasta.name != f"resultados_{modelo}":
        raise Trava(f"A pasta de resultados do {NOMES[modelo]} precisa se chamar resultados_{modelo}, "
                    f"não {pasta.name} (veja o config.py).")
    for outro in MODELOS:
        if outro == modelo:
            continue
        a, b = pasta.resolve(), _pasta_do_config(outro).resolve()
        if a == b or a in b.parents or b in a.parents:
            raise Trava(f"As pastas de resultados do {NOMES[modelo]} e do {NOMES[outro]} precisam ser "
                        "separadas (veja o config.py).")
    return Caminhos(modelo, pasta)


# ---------------------------------------------------------------- 2. trava de pasta
def _padroes(modelo):
    """Os únicos caminhos que um modelo grava, relativos à pasta dele."""
    m, n = re.escape(modelo), r"[1-9][0-9]*"
    return (
        rf"rodadas_{m}\.csv",
        rf"rodada_{n}_{m}",
        rf"rodada_{n}_{m}/resultados_{m}\.csv",
        rf"rodada_{n}_{m}/(FP|FN)_{m}",
        rf"rodada_{n}_{m}/(FP|FN)_{m}/p[01]\.[0-9]{{2}}_[^/]+_{m}\.[A-Za-z0-9]+",
    )


def conferir_caminho(modelo, caminho):
    """Trava de pasta (proteção 2): só grava dentro da pasta do modelo, com os nomes dele."""
    pasta = caminhos(modelo).pasta.resolve()
    alvo = Path(caminho).resolve()
    try:
        relativo = alvo.relative_to(pasta).as_posix()
    except ValueError:
        raise Trava(f"{alvo} fica fora da pasta do {NOMES[modelo]} ({pasta}). Nada foi gravado.") from None
    if not any(re.fullmatch(padrao, relativo) for padrao in _padroes(modelo)):
        raise Trava(f"{relativo} não segue os nomes do {NOMES[modelo]} dentro de {pasta.name}/. "
                    "Nada foi gravado.")
    return alvo


# ---------------------------------------------------------------- 3. trava de tabela
def _conferir_campos(linha, campos, onde):
    faltam = [campo for campo in campos if campo not in linha]
    sobram = [campo for campo in linha if campo not in campos]
    if faltam or sobram:
        detalhes = "; ".join(parte for parte in (
            f"faltam {', '.join(faltam)}" if faltam else "",
            f"sobram {', '.join(sobram)}" if sobram else "") if parte)
        raise Trava(f"Esta linha não é de {onde} ({detalhes}). Nada foi gravado.")


def _cabecalho(caminho):
    """As colunas da primeira linha, lidas como o dados.ler_csv lê (vírgula ou ponto e vírgula)."""
    bruto = Path(caminho).read_bytes()
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = bruto.decode("latin-1")
    linhas = texto.splitlines()
    if not linhas:
        return []
    separador = ";" if linhas[0].count(";") > linhas[0].count(",") else ","
    return [campo.strip() for campo in next(csv.reader(io.StringIO(linhas[0]), delimiter=separador))]


def _ler_conferido(caminho, campos, onde):
    """Lê um CSV de um modelo, só se as colunas forem as dele (trava de tabela na leitura)."""
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    cabecalho = _cabecalho(caminho)
    if cabecalho and cabecalho != campos:
        raise Trava(f"{caminho.name} não tem as colunas de {onde}: o arquivo é de outro modelo "
                    "ou foi mudado à mão.")
    return dados.ler_csv(caminho)


def _gravar_csv(modelo, caminho, linhas, campos):
    """Grava um CSV do modelo de uma vez: escreve ao lado e troca no fim."""
    alvo = conferir_caminho(modelo, caminho)
    alvo.parent.mkdir(parents=True, exist_ok=True)
    temporario = alvo.with_name(f".{alvo.name}.gravando")
    dados.gravar_csv(temporario, linhas, campos)
    os.replace(temporario, alvo)


# ---------------------------------------------------------------- leitura
def ler_rodadas(modelo):
    """As rodadas de um modelo, em ordem; a tabela precisa ser a dele."""
    tabela = caminhos(modelo).tabela
    linhas = _ler_conferido(tabela, CAMPOS_RODADAS[modelo], f"rodadas do {NOMES[modelo]}")
    return sorted(linhas, key=lambda linha: int(linha["rodada"]))


def ler_resultados(modelo, numero):
    """Os resultados imagem por imagem de uma rodada de um modelo."""
    arquivo = caminhos(modelo).resultados(numero)
    return _ler_conferido(arquivo, CAMPOS_RESULTADOS[modelo], f"resultados do {NOMES[modelo]}")


# ---------------------------------------------------------------- assinatura do bairro (7 e 12)
def assinatura_bairro(bairro):
    """Um código curto tirado do extrator e dos números, dos rótulos e dos nomes das imagens
    do bairro, na ordem do arquivo. Muda se um rótulo for corrigido, se uma imagem for
    trocada (mesmo mantendo o total), se a ordem mudar ou se o extrator mudar."""
    hash_ = hashlib.sha1()
    hash_.update(str(bairro["extrator"]).encode("utf-8"))
    hash_.update(np.ascontiguousarray(bairro["X"], dtype=np.float32).tobytes())
    hash_.update(np.ascontiguousarray(bairro["y"], dtype=np.int64).tobytes())
    hash_.update("\n".join(bairro["arquivos"]).encode("utf-8"))
    return hash_.hexdigest()[:12]


def parametros(modelo):
    """Os parâmetros próprios de cada modelo, como vão para a tabela."""
    conferir_modelo(modelo)
    if modelo == "fuzzyknn":
        return {"k": config.FUZZYKNN_K}
    return {"C": config.SVM_C, "gamma": config.SVM_GAMMA}


# ---------------------------------------------------------------- gravar a rodada dos dois
def linha_da_rodada(modelo, numero, o_que_mudou, medidas, bairro):
    """A linha da tabela de rodadas de um modelo."""
    casas = medidas["casas"]
    atingida, _ = meta_atingida(medidas["acuracia"], casas["FN"], casas["FP"])
    return {
        "rodada": int(numero),
        "o_que_mudou": o_que_mudou,
        "extrator": bairro["extrator"],
        **parametros(modelo),
        "bairro": len(bairro["arquivos"]),
        "assinatura_bairro": assinatura_bairro(bairro),
        **{casa: casas[casa] for casa in ("VP", "FN", "FP", "VN")},
        "acuracia": f"{medidas['acuracia']:.4f}",
        "area_cinzenta": contar_area_cinzenta(medidas["resultados"]),
        "meta": "sim" if atingida else "não",
        "data": date.today().isoformat(),
    }


def _mesmas_entradas(modelo, nova, anterior):
    """O modelo recebeu o mesmo de antes: mesmo extrator, mesmo bairro e mesmos parâmetros."""
    chaves = ["extrator", "assinatura_bairro", *CAMPOS_PROPRIOS_RODADAS[modelo]]
    return all(str(nova[chave]) == str(anterior[chave]) for chave in chaves)


def diferencas(nova, anterior):
    """O que mudou nos números de uma rodada para a outra: ["FN: 3 → 4", ...]."""
    return [f"{chave}: {anterior[chave]} → {nova[chave]}"
            for chave in NUMEROS_DA_RODADA if str(nova[chave]) != str(anterior[chave])]


def _copias(modelo, numero, resultados):
    """As cópias das imagens de FP e FN: (origem, destino), com o destino já conferido."""
    pasta = caminhos(modelo)
    copias = []
    for resultado in resultados:
        if resultado["casa"] in ("FP", "FN"):
            origem = dados.localizar_imagem(resultado["arquivo"])
            if origem:
                destino = pasta.copia_imagem(numero, resultado["casa"], resultado["p_cachorro_tela"],
                                             resultado["arquivo"])
                copias.append((origem, conferir_caminho(modelo, destino)))
    return copias


def registrar_rodada(numero, o_que_mudou, medidas_por_modelo, bairro, substituir=False, mudou=()):
    """Grava a rodada dos dois modelos: uma linha em cada tabela, com o mesmo número e o
    mesmo "o que mudou", os resultados imagem por imagem e as cópias das imagens de FP e FN.

    Tudo é conferido antes da primeira gravação (proteção 5). substituir=True é o pedido
    explícito para gravar de novo uma rodada que já existe, e vale para as duas tabelas
    (proteção 4). mudou lista o modelo (ou os modelos) cujo código mudou nesta rodada: os
    outros, se receberam o mesmo bairro, o mesmo extrator e os mesmos parâmetros, precisam
    repetir os números da rodada anterior.
    """
    numero = int(numero)
    if numero < 1:
        raise Trava("O número da rodada começa em 1.")
    if not str(o_que_mudou).strip():
        raise Trava('Toda rodada precisa dizer o que mudou (--mudanca "...").')
    if set(medidas_por_modelo) != set(MODELOS):
        raise Trava("A rodada grava os dois modelos juntos, no mesmo comando.")
    for modelo in mudou:
        conferir_modelo(modelo)

    # 6. as mesmas 80 nos dois, com as colunas de cada modelo
    listas = {modelo: [r["arquivo"] for r in medidas_por_modelo[modelo]["resultados"]] for modelo in MODELOS}
    if not listas["fuzzyknn"] or listas["fuzzyknn"] != listas["svm"]:
        raise Trava("Os dois modelos precisam ser avaliados nas mesmas imagens do teste fixo, na mesma "
                    "ordem. Nada foi gravado.")
    for modelo in MODELOS:
        for resultado in medidas_por_modelo[modelo]["resultados"]:
            _conferir_campos(resultado, CAMPOS_RESULTADOS[modelo], f"resultados do {NOMES[modelo]}")

    # 3 e 5. as duas tabelas, cada uma do seu modelo e com as mesmas rodadas
    tabelas = {modelo: ler_rodadas(modelo) for modelo in MODELOS}
    numeros = {modelo: [int(linha["rodada"]) for linha in tabelas[modelo]] for modelo in MODELOS}
    if numeros["fuzzyknn"] != numeros["svm"]:
        raise Trava(f"As duas tabelas não têm as mesmas rodadas (Fuzzy k-NN: {numeros['fuzzyknn']}; "
                    f"SVM: {numeros['svm']}). Confira as pastas de resultados antes de gravar. "
                    "Nada foi gravado.")
    existentes = numeros["fuzzyknn"]

    # 4. trava de rodada
    pastas_existentes = [modelo for modelo in MODELOS if caminhos(modelo).pasta_rodada(numero).exists()]
    ja_existe = numero in existentes or bool(pastas_existentes)
    if ja_existe and not substituir:
        onde = "nas tabelas" if numero in existentes else "nas pastas de resultados"
        raise Trava(f"A rodada {numero} já existe {onde}. Para gravar de novo, nas duas tabelas, peça "
                    "explicitamente com --substituir. Nada foi gravado.")
    if substituir and not ja_existe:
        raise Trava(f"Não há rodada {numero} para substituir. Nada foi gravado.")
    if not substituir and existentes and numero < max(existentes):
        raise Trava(f"A rodada {numero} viria antes da última registrada ({max(existentes)}). "
                    f"Use o próximo número, {max(existentes) + 1}. Nada foi gravado.")

    novas = {modelo: linha_da_rodada(modelo, numero, o_que_mudou, medidas_por_modelo[modelo], bairro)
             for modelo in MODELOS}

    # o modelo que não mudou repete os números da rodada anterior
    anteriores = [n for n in existentes if n < numero]
    if anteriores:
        numero_anterior = max(anteriores)
        for modelo in MODELOS:
            if modelo in mudou:
                continue
            anterior = next(linha for linha in tabelas[modelo] if int(linha["rodada"]) == numero_anterior)
            if _mesmas_entradas(modelo, novas[modelo], anterior):
                diferenca = diferencas(novas[modelo], anterior)
                if diferenca:
                    raise Trava(
                        f"O {NOMES[modelo]} recebeu o mesmo bairro, o mesmo extrator e os mesmos parâmetros "
                        f"da rodada {numero_anterior}, mas não repetiu os números dela "
                        f"({'; '.join(diferenca)}). Nada foi gravado. Se a mudança desta rodada mexeu no "
                        f"código do {NOMES[modelo]}, rode de novo com --mudou {modelo}.")

    # tudo conferido antes de gravar: tabelas novas, caminhos e cópias
    tabelas_novas = {}
    for modelo in MODELOS:
        linhas = [linha for linha in tabelas[modelo] if int(linha["rodada"]) != numero] + [novas[modelo]]
        linhas.sort(key=lambda linha: int(linha["rodada"]))
        for linha in linhas:
            _conferir_campos(linha, CAMPOS_RODADAS[modelo], f"rodadas do {NOMES[modelo]}")
        tabelas_novas[modelo] = linhas
    planos = {}
    for modelo in MODELOS:
        pasta = caminhos(modelo)
        planos[modelo] = {
            "pasta": conferir_caminho(modelo, pasta.pasta_rodada(numero)),
            "resultados": conferir_caminho(modelo, pasta.resultados(numero)),
            "copias": _copias(modelo, numero, medidas_por_modelo[modelo]["resultados"]),
        }

    # 5. grava no fim: primeiro as pastas das rodadas, depois as duas tabelas
    for modelo in MODELOS:
        plano = planos[modelo]
        if plano["pasta"].exists():                   # só chega aqui com --substituir
            shutil.rmtree(plano["pasta"])
        plano["pasta"].mkdir(parents=True)
        _gravar_csv(modelo, plano["resultados"], medidas_por_modelo[modelo]["resultados"],
                    CAMPOS_RESULTADOS[modelo])
        for origem, destino in plano["copias"]:
            destino.parent.mkdir(exist_ok=True)
            shutil.copy2(origem, destino)
    for modelo in MODELOS:
        _gravar_csv(modelo, caminhos(modelo).tabela, tabelas_novas[modelo], CAMPOS_RODADAS[modelo])
    return novas


# ---------------------------------------------------------------- 7. aviso da home
def aviso_da_home(modelo, bairro):
    """Proteção 7: um aviso se o modelo em uso não foi avaliado no teste fixo com este bairro
    (ou com estes parâmetros); None se a última rodada valer para ele."""
    rodadas = ler_rodadas(modelo)
    if not rodadas:
        return (f"O {NOMES[modelo]} ainda não foi avaliado no teste fixo. Rode no terminal: "
                'python avaliar.py --rodada 1 --mudanca "ponto de partida: bairro com 320 (160 + 160)"')
    ultima = rodadas[-1]
    em_uso = assinatura_bairro(bairro)
    if ultima["assinatura_bairro"] != em_uso:
        return (f"O bairro em uso não é o da última rodada avaliada do {NOMES[modelo]} (rodada "
                f"{ultima['rodada']}): a assinatura do bairro é {em_uso}, e a da rodada é "
                f"{ultima['assinatura_bairro']}. Os números da visão admin não valem para este bairro; "
                "registre uma rodada nova com python avaliar.py.")
    diferentes = [f"{chave} = {valor} agora e {ultima[chave]} na rodada {ultima['rodada']}"
                  for chave, valor in parametros(modelo).items() if str(valor) != ultima[chave]]
    if diferentes:
        return (f"Os parâmetros do {NOMES[modelo]} mudaram depois da última rodada avaliada "
                f"({'; '.join(diferentes)}). Registre uma rodada nova com python avaliar.py.")
    return None


# ---------------------------------------------------------------- 13. comparativo
def rodadas_nas_duas_tabelas():
    """Os números de rodada que existem nas duas tabelas: só eles entram no comparativo."""
    numeros = [{int(linha["rodada"]) for linha in ler_rodadas(modelo)} for modelo in MODELOS]
    return sorted(numeros[0] & numeros[1])


def carregar_comparacao(numero):
    """Proteção 13: a rodada N dos dois modelos, só se as duas linhas tiverem a mesma assinatura
    do bairro e os dois resultados listarem as mesmas 80 imagens, na mesma ordem.

    Devolve ({modelo: linha da tabela}, {modelo: resultados imagem por imagem}).
    """
    linhas = {}
    for modelo in MODELOS:
        linha = next((l for l in ler_rodadas(modelo) if int(l["rodada"]) == int(numero)), None)
        if linha is None:
            raise Trava(f"A rodada {numero} não está na tabela do {NOMES[modelo]}: o comparativo só "
                        "compara a mesma rodada nos dois modelos.")
        linhas[modelo] = linha
    if linhas["fuzzyknn"]["assinatura_bairro"] != linhas["svm"]["assinatura_bairro"]:
        raise Trava(f"Na rodada {numero}, os dois modelos não foram avaliados com o mesmo bairro "
                    f"(assinaturas {linhas['fuzzyknn']['assinatura_bairro']} e "
                    f"{linhas['svm']['assinatura_bairro']}): a comparação não vale.")
    resultados = {modelo: ler_resultados(modelo, numero) for modelo in MODELOS}
    arquivos = {modelo: [r["arquivo"] for r in resultados[modelo]] for modelo in MODELOS}
    if not arquivos["fuzzyknn"] or arquivos["fuzzyknn"] != arquivos["svm"]:
        raise Trava(f"Na rodada {numero}, os resultados dos dois modelos não listam as mesmas imagens "
                    "do teste fixo: a comparação não vale.")
    return linhas, resultados
