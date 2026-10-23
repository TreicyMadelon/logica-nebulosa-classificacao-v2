"""A base de imagens: metadados, categorias, distribuição, teste fixo e vetores."""
import csv
import io
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import config
from codigos_imagenet import e_animal, e_raca_de_cachorro, nome_imagenet

CAMPOS_METADADOS = ["arquivo", "classe", "dataset", "categoria_original"]
CAMPOS_DISTRIBUICAO = ["classe", "dataset", "categoria", "quantidade"]


# ---------------------------------------------------------------- CSV
def ler_csv(caminho):
    """Lê um CSV como lista de dicionários.

    Aceita vírgula ou ponto e vírgula, porque o Excel em português salva com
    ponto e vírgula quando você edita uma decisão da triagem.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    bruto = caminho.read_bytes()
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = bruto.decode("latin-1")
    if not texto.strip():
        return []
    cabecalho = texto.splitlines()[0]
    separador = ";" if cabecalho.count(";") > cabecalho.count(",") else ","
    linhas = []
    for linha in csv.DictReader(io.StringIO(texto), delimiter=separador):
        limpa = {chave.strip(): (valor or "").strip() for chave, valor in linha.items() if chave is not None}
        if any(limpa.values()):          # ignora linhas vazias que a planilha às vezes acrescenta
            linhas.append(limpa)
    return linhas


def gravar_csv(caminho, linhas, campos):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(linhas)


# ---------------------------------------------------------------- metadados
def ler_metadados(caminho=None):
    """Uma linha por imagem, na ordem do arquivo (é essa ordem que o split usa)."""
    return ler_csv(caminho or config.ARQ_METADADOS)


def gravar_metadados(linhas, caminho=None):
    gravar_csv(caminho or config.ARQ_METADADOS, linhas, CAMPOS_METADADOS)


def rotulo(classe):
    """cachorro → 1; nao_cachorro → 0."""
    return 1 if classe == "cachorro" else 0


def classe_do_rotulo(valor):
    return "cachorro" if int(valor) == 1 else "nao_cachorro"


def texto_classe(classe):
    """Como a classe aparece na tela: "cachorro" ou "não cachorro"."""
    return "cachorro" if classe in ("cachorro", 1) else "não cachorro"


def nome_dataset(dataset):
    return {"imagenet": "ImageNet (validação)", "voc": "Pascal VOC"}.get(dataset, dataset)


def caminho_imagem(linha):
    return config.PASTAS_CLASSE[linha["classe"]] / linha["arquivo"]


def localizar_imagem(arquivo):
    """Procura a imagem nas pastas da base e nas descartadas (a classe pode ter sido corrigida)."""
    for pasta in (*config.PASTAS_CLASSE.values(), config.PASTA_DESCARTADAS):
        caminho = pasta / arquivo
        if caminho.exists():
            return caminho
    return None


# ---------------------------------------------------------------- categorias
def categoria_planejada(classe, dataset, categoria_original):
    """A categoria real da tabela de distribuição (ex.: "outros animais")."""
    if dataset == "imagenet":
        if classe == "cachorro":
            return config.CAT_RACAS if e_raca_de_cachorro(categoria_original) else config.CAT_FORA_DO_PLANO
        if e_raca_de_cachorro(categoria_original):
            return config.CAT_FORA_DO_PLANO
        return config.CAT_OUTROS_ANIMAIS if e_animal(categoria_original) else config.CAT_OBJETOS
    if classe == "cachorro":
        return config.CAT_CACHORRO if categoria_original == config.VOC_CACHORRO else config.CAT_FORA_DO_PLANO
    if categoria_original in config.VOC_OUTROS_ANIMAIS:
        return config.CAT_OUTROS_ANIMAIS
    if categoria_original == config.VOC_PESSOAS:
        return config.CAT_PESSOAS
    return config.CAT_FORA_DO_PLANO


def nome_categoria(dataset, categoria_original):
    """Nome legível da categoria original (ex.: "lobo-cinzento (n02114367)", "gato")."""
    if dataset == "imagenet":
        nome = nome_imagenet(categoria_original)
        return nome if nome == categoria_original else f"{nome} ({categoria_original})"
    return config.NOMES_VOC.get(categoria_original, categoria_original)


# ---------------------------------------------------------------- distribuição
def contar_distribuicao(linhas):
    """Quantas imagens há em cada (classe, dataset de origem, categoria real)."""
    return Counter(
        (l["classe"], l["dataset"], categoria_planejada(l["classe"], l["dataset"], l["categoria_original"]))
        for l in linhas)


def gravar_distribuicao_planejada(contagem, caminho=None):
    linhas = [{"classe": classe, "dataset": dataset, "categoria": categoria, "quantidade": n}
              for (classe, dataset, categoria), n in sorted(contagem.items())]
    gravar_csv(caminho or config.ARQ_DISTRIBUICAO, linhas, CAMPOS_DISTRIBUICAO)


def ler_distribuicao_planejada(caminho=None):
    contagem = Counter()
    for linha in ler_csv(caminho or config.ARQ_DISTRIBUICAO):
        contagem[(linha["classe"], linha["dataset"], linha["categoria"])] += int(linha["quantidade"])
    return contagem


def diferencas_distribuicao(atual, planejada):
    """Lista o que saiu da distribuição planejada (vazia = distribuição preservada)."""
    diferencas = []
    for chave in sorted(set(atual) | set(planejada)):
        tem, deveria = atual.get(chave, 0), planejada.get(chave, 0)
        if tem != deveria:
            classe, dataset, categoria = chave
            diferencas.append(f"{texto_classe(classe)} · {nome_dataset(dataset)} · {categoria}: "
                              f"{tem} (planejado: {deveria})")
    return diferencas


# ---------------------------------------------------------------- split 80/20 e teste fixo
def grupos_do_split(linhas):
    """Classe + dataset de origem de cada imagem, para o stratify (ex.: "1_imagenet", "0_voc")."""
    y = [rotulo(l["classe"]) for l in linhas]
    datasets_origem = [l["dataset"] for l in linhas]
    return [f"{r}_{d}" for r, d in zip(y, datasets_origem)]


def sortear_teste_fixo(linhas):
    """Split 80/20 estratificado por classe + dataset de origem, com random_state=42."""
    arquivos = [l["arquivo"] for l in linhas]
    grupo = grupos_do_split(linhas)
    arq_treino, arq_teste = train_test_split(
        arquivos,
        test_size=config.FRACAO_TESTE,   # quantas vão para o teste (20%)
        stratify=grupo,                  # mantém a proporção de classe + dataset de origem
        random_state=config.SEMENTE)     # quais vão (sorteio sempre igual)
    ordem = {arquivo: i for i, arquivo in enumerate(arquivos)}
    return sorted(arq_teste, key=ordem.get)   # na ordem dos metadados


def ler_teste_fixo(caminho=None):
    caminho = Path(caminho or config.ARQ_TESTE_FIXO)
    if not caminho.exists():
        return []
    return [linha.strip() for linha in caminho.read_text(encoding="utf-8").splitlines() if linha.strip()]


def definir_teste_fixo(linhas, caminho=None):
    """Devolve (lista do teste fixo, se foi criada agora, avisos).

    Na primeira execução, sorteia as 80 imagens e grava a lista em
    base_imagens/teste_fixo.txt. Nas seguintes, só lê essa lista: imagens
    acrescentadas numa melhoria vão para o bairro, e o teste continua o mesmo.
    """
    caminho = Path(caminho or config.ARQ_TESTE_FIXO)
    if not caminho.exists():
        teste = sortear_teste_fixo(linhas)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text("\n".join(teste) + "\n", encoding="utf-8")
        return teste, True, []

    teste = ler_teste_fixo(caminho)
    na_base = {l["arquivo"] for l in linhas}
    faltando = [arquivo for arquivo in teste if arquivo not in na_base]
    avisos = []
    if faltando:
        avisos.append(
            f"Atenção: {len(faltando)} imagem(ns) do teste fixo não está(ão) mais na base "
            f"(removida ou renomeada): {', '.join(faltando)}. O teste fica com "
            f"{len(teste) - len(faltando)} imagens e deixa de ser comparável com as rodadas anteriores.")
    return [arquivo for arquivo in teste if arquivo in na_base], False, avisos


# ---------------------------------------------------------------- vetores (bairro e teste fixo)
def salvar_vetores(caminho, X_num, linhas, extrator):
    """Guarda os números de cada imagem junto com o rótulo e a referência à imagem."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    np.savez(caminho,
             X=np.asarray(X_num, dtype=np.float32),
             y=np.array([rotulo(l["classe"]) for l in linhas], dtype=np.int64),
             arquivos=np.array([l["arquivo"] for l in linhas], dtype=str),
             datasets=np.array([l["dataset"] for l in linhas], dtype=str),
             categorias=np.array([l["categoria_original"] for l in linhas], dtype=str),
             extrator=np.array(extrator))


def carregar_vetores(caminho):
    with np.load(caminho) as dados_npz:
        return {
            "X": dados_npz["X"],
            "y": dados_npz["y"],
            "arquivos": [str(a) for a in dados_npz["arquivos"]],
            "datasets": [str(d) for d in dados_npz["datasets"]],
            "categorias": [str(c) for c in dados_npz["categorias"]],
            "extrator": str(dados_npz["extrator"]),
        }


def conferir_vetores(linhas, bairro, teste):
    """Avisa se a base ou o teste fixo mudaram depois da última montagem do bairro."""
    avisos = []
    na_base = {l["arquivo"]: rotulo(l["classe"]) for l in linhas}
    nos_vetores = {**dict(zip(bairro["arquivos"], bairro["y"].tolist())),
                   **dict(zip(teste["arquivos"], teste["y"].tolist()))}
    if na_base != nos_vetores:
        avisos.append("A base mudou depois da última montagem do bairro (imagens acrescentadas, "
                      "trocadas ou com rótulo corrigido). Rode python montar_bairro.py antes.")
    teste_fixo = set(ler_teste_fixo())
    if teste_fixo and set(teste["arquivos"]) != (teste_fixo & set(na_base)):
        avisos.append("Os vetores do teste não batem com base_imagens/teste_fixo.txt. "
                      "Rode python montar_bairro.py antes.")
    return avisos
