"""Bases de origem: leitura das anotações (XML) e sorteio das imagens.

As duas bases têm anotações em XML no mesmo formato, e o mesmo leitor serve
para as duas. No Pascal VOC, a categoria vem escrita ("dog", "cat", "person");
no ImageNet, vem como código (n02342885 = hamster), traduzido pela tabela de
codigos_imagenet.py. O XML do ImageNet marca só a categoria principal da foto:
quem confere se há um cachorro ao fundo é a triagem com YOLO.
"""
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import config
import dados
from codigos_imagenet import e_raca_de_cachorro

EXTENSOES_IMAGEM = {".jpeg", ".jpg"}


def _numero(texto):
    try:
        return float(texto)
    except (TypeError, ValueError):
        return 0.0


def ler_anotacao(caminho_xml):
    """Lê um XML de anotação (ImageNet ou Pascal VOC).

    Só olha os objetos de primeiro nível: as "partes" do Pascal VOC
    (cabeça, mão e pé dentro de uma pessoa) ficam de fora.
    """
    raiz = ET.parse(caminho_xml).getroot()
    objetos = []
    for objeto in raiz.findall("object"):
        caixa = objeto.find("bndbox")
        area = 0.0
        if caixa is not None:
            largura = _numero(caixa.findtext("xmax")) - _numero(caixa.findtext("xmin"))
            altura = _numero(caixa.findtext("ymax")) - _numero(caixa.findtext("ymin"))
            area = max(largura, 0.0) * max(altura, 0.0)
        objetos.append({
            "nome": (objeto.findtext("name") or "").strip(),
            "area": area,
            "dificil": (objeto.findtext("difficult") or "0").strip() == "1",
        })
    return {"arquivo": (raiz.findtext("filename") or "").strip(), "objetos": objetos}


def candidatas_imagenet():
    """Todas as fotos do conjunto de validação do ImageNet, com o código da categoria."""
    pasta_xml, pasta_img = config.PASTA_IMAGENET_ANOTACOES, config.PASTA_IMAGENET_IMAGENS
    if not pasta_xml.is_dir() or not pasta_img.is_dir():
        sys.exit(f"Não encontrei o ImageNet (validação) em:\n  {pasta_img}\n  {pasta_xml}\nVeja o README.")
    imagens = {p.stem: p for p in pasta_img.rglob("*") if p.suffix.lower() in EXTENSOES_IMAGEM}
    candidatas = []
    for caminho_xml in sorted(pasta_xml.rglob("*.xml")):
        codigos = {objeto["nome"] for objeto in ler_anotacao(caminho_xml)["objetos"]}
        imagem = imagens.get(caminho_xml.stem)
        if len(codigos) != 1 or imagem is None:
            continue
        codigo = codigos.pop()
        candidatas.append({
            "dataset": "imagenet",
            "origem": imagem,
            "arquivo": f"imagenet_{imagem.name}",
            "categoria_original": codigo,
            "tem_cachorro": e_raca_de_cachorro(codigo),
            "principal_dificil": False,
        })
    return candidatas


def candidatas_voc():
    """Todas as fotos do Pascal VOC 2012, com a categoria do maior objeto anotado."""
    pasta_xml, pasta_img = config.PASTA_VOC / "Annotations", config.PASTA_VOC / "JPEGImages"
    if not pasta_xml.is_dir() or not pasta_img.is_dir():
        sys.exit(f"Não encontrei o Pascal VOC 2012 em:\n  {config.PASTA_VOC}\nVeja o README.")
    candidatas = []
    for caminho_xml in sorted(pasta_xml.glob("*.xml")):
        objetos = ler_anotacao(caminho_xml)["objetos"]
        imagem = pasta_img / f"{caminho_xml.stem}.jpg"
        if not objetos or not imagem.exists():
            continue
        principal = max(objetos, key=lambda objeto: objeto["area"])   # o maior objeto anotado
        candidatas.append({
            "dataset": "voc",
            "origem": imagem,
            "arquivo": f"voc_{imagem.name}",
            "categoria_original": principal["nome"],
            "tem_cachorro": any(objeto["nome"] == config.VOC_CACHORRO for objeto in objetos),
            "principal_dificil": principal["dificil"],
        })
    return candidatas


def indexar_fontes():
    """Agrupa as fotos das duas bases por (dataset de origem, categoria original)."""
    print("Lendo as anotações do ImageNet e do Pascal VOC (pode levar alguns minutos)...")
    indice = defaultdict(list)
    for candidata in candidatas_imagenet() + candidatas_voc():
        indice[(candidata["dataset"], candidata["categoria_original"])].append(candidata)
    return dict(indice)


def candidatas_do_codigo(indice, classe, dataset, codigo):
    """Fotos de uma categoria original que respeitam a regra de inclusão da classe.

    Cachorro: o cão claramente visível e principal. Não cachorro: nenhum cachorro.
    """
    lista = indice.get((dataset, codigo), [])
    if dataset == "imagenet":
        if (classe == "cachorro") != e_raca_de_cachorro(codigo):
            return []
        return lista
    if classe == "cachorro":
        # o cachorro é o maior objeto anotado e não está marcado como difícil
        if codigo != config.VOC_CACHORRO:
            return []
        return [c for c in lista if not c["principal_dificil"]]
    # não cachorro: nenhum cachorro anotado na foto
    return [c for c in lista if not c["tem_cachorro"]]


def codigos_padrao(dataset, categoria, indice):
    """Os códigos sorteados para uma categoria real do plano."""
    if dataset == "imagenet":
        if categoria == config.CAT_RACAS:
            return sorted(codigo for (ds, codigo) in indice if ds == "imagenet" and e_raca_de_cachorro(codigo))
        return sorted(config.CODIGOS_IMAGENET.get(categoria, {}))
    if categoria == config.CAT_CACHORRO:
        return [config.VOC_CACHORRO]
    if categoria == config.CAT_OUTROS_ANIMAIS:
        return sorted(config.VOC_OUTROS_ANIMAIS)
    if categoria == config.CAT_PESSOAS:
        return [config.VOC_PESSOAS]
    return []


def sortear_em_rodizio(grupos, n, semente):
    """Sorteia n fotos passando por uma categoria de cada vez (rodízio).

    grupos: {código da categoria: [fotos]}. Assim cada categoria fica com quase a
    mesma quantidade (ex.: 50 fotos em 5 categorias → 10 de cada). A semente fixa
    faz o sorteio sair sempre igual.
    """
    sorteio = random.Random(semente)
    codigos = sorted(grupos)
    sorteio.shuffle(codigos)
    filas = {}
    for codigo in codigos:
        fila = sorted(grupos[codigo], key=lambda foto: foto["arquivo"])
        sorteio.shuffle(fila)
        filas[codigo] = fila
    escolhidas = []
    while len(escolhidas) < n and any(filas.values()):
        for codigo in codigos:
            if len(escolhidas) < n and filas[codigo]:
                escolhidas.append(filas[codigo].pop())
    return escolhidas


def sortear(indice, classe, dataset, codigos, n, semente, ja_usadas):
    """Sorteia n fotos entre os códigos dados, sem repetir as que já foram usadas."""
    grupos = {}
    for codigo in codigos:
        livres = [c for c in candidatas_do_codigo(indice, classe, dataset, codigo)
                  if c["arquivo"] not in ja_usadas]
        if livres:
            grupos[codigo] = livres
    return sortear_em_rodizio(grupos, n, semente)


def sortear_substituta(indice, linha, ja_usadas):
    """Substituta da mesma categoria e do mesmo dataset de origem.

    Primeiro tenta a mesma categoria original (ex.: outro lobo); se não houver,
    outra foto da mesma categoria real (ex.: outro animal do ImageNet).
    """
    classe, dataset, codigo = linha["classe"], linha["dataset"], linha["categoria_original"]
    semente = f"{config.SEMENTE}-substituta-{linha['arquivo']}"
    escolhidas = sortear(indice, classe, dataset, [codigo], 1, semente, ja_usadas)
    if not escolhidas:
        categoria = dados.categoria_planejada(classe, dataset, codigo)
        escolhidas = sortear(indice, classe, dataset, codigos_padrao(dataset, categoria, indice),
                             1, semente, ja_usadas)
    return escolhidas[0] if escolhidas else None


def copiar_para_base(candidata, classe):
    """Copia a foto para base_imagens/<classe>/ e devolve a linha dos metadados."""
    destino = config.PASTAS_CLASSE[classe] / candidata["arquivo"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidata["origem"], destino)
    return {
        "arquivo": candidata["arquivo"],
        "classe": classe,
        "dataset": candidata["dataset"],
        "categoria_original": candidata["categoria_original"],
    }


def copiar_da_lista(linhas):
    """Copia para a base exatamente as fotos que as linhas do metadados listam, sem sortear.

    É o caminho de quem clona o repositório: o metadados.csv vem no Git, as imagens não (o
    ImageNet e o Pascal VOC não permitem redistribuição). Sortear de novo não serviria —
    a triagem substituiu uma foto, e um sorteio novo mudaria algumas escolhas.

    O nome na base é "<dataset>_<nome original>", então dá para achar cada foto na origem.
    Devolve (copiadas, já estavam, não encontradas).
    """
    precisa = {linha["dataset"] for linha in linhas}
    pasta_voc = config.PASTA_VOC / "JPEGImages"
    if "imagenet" in precisa and not config.PASTA_IMAGENET_IMAGENS.is_dir():
        sys.exit(f"Não encontrei as imagens do ImageNet em:\n  {config.PASTA_IMAGENET_IMAGENS}\nVeja o README.")
    if "voc" in precisa and not pasta_voc.is_dir():
        sys.exit(f"Não encontrei as imagens do Pascal VOC em:\n  {pasta_voc}\nVeja o README.")

    imagenet = {}
    if "imagenet" in precisa:
        imagenet = {p.name: p for p in config.PASTA_IMAGENET_IMAGENS.rglob("*")
                    if p.suffix.lower() in EXTENSOES_IMAGEM}

    copiadas, ja_estavam, sem_origem = [], [], []
    for linha in linhas:
        nome, dataset = linha["arquivo"], linha["dataset"]
        destino = config.PASTAS_CLASSE[linha["classe"]] / nome
        if destino.exists():
            ja_estavam.append(nome)
            continue
        original = nome[len(dataset) + 1:] if nome.startswith(f"{dataset}_") else nome
        origem = imagenet.get(original) if dataset == "imagenet" else pasta_voc / original
        if origem is None or not origem.exists():
            sem_origem.append(nome)
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origem, destino)
        copiadas.append(nome)
    return copiadas, ja_estavam, sem_origem


def arquivos_ja_usados():
    """Fotos que já estão na base ou que foram descartadas na triagem (não voltam)."""
    usados = {linha["arquivo"] for linha in dados.ler_metadados()}
    usados |= {linha["arquivo"] for linha in dados.ler_csv(config.ARQ_DESCARTADAS)}
    return usados
