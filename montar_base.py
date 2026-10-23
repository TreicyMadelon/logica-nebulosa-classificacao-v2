"""Parte 1 · Etapas 1 a 3: monta a base única de 400 imagens (ou acrescenta imagens numa melhoria).

Uso:
    python montar_base.py
        monta a base: 200 cachorro + 200 não cachorro, 100 de cada dataset de origem em cada classe
    python montar_base.py --remontar
        clone do repositório: copia as imagens que o metadados.csv já lista, sem sortear de novo
    python montar_base.py --acrescentar 20
        melhoria: + 20 imagens de cada dataset em cada classe (80 no total)
    python montar_base.py --acrescentar 20 --categorias n02114367,n02114548
        a parte das categorias citadas fica só nelas (ex.: lobos em não cachorro do ImageNet);
        as outras partes seguem o sorteio normal

As imagens vão para base_imagens/cachorro/ e base_imagens/nao_cachorro/, e cada uma
ganha uma linha em base_imagens/metadados.csv (arquivo, classe, dataset, categoria_original).
Depois, todas passam pela triagem: python triagem.py
"""
import argparse
import re
import sys

import config
import dados
import fontes
from codigos_imagenet import e_raca_de_cachorro


def sortear_partes(indice, partes):
    """partes: [(classe, dataset, códigos, quantidade, semente)].

    Sorteia tudo antes de copiar qualquer arquivo: se faltar foto, nada é copiado.
    """
    usadas = fontes.arquivos_ja_usados()
    escolhas, faltas = [], []
    for classe, dataset, codigos, quantidade, semente in partes:
        escolhidas = fontes.sortear(indice, classe, dataset, codigos, quantidade, semente, usadas)
        if len(escolhidas) < quantidade:
            faltas.append(f"{dados.texto_classe(classe)} · {dados.nome_dataset(dataset)}: "
                          f"pedi {quantidade}, achei {len(escolhidas)}")
        usadas |= {foto["arquivo"] for foto in escolhidas}
        escolhas += [(classe, foto) for foto in escolhidas]
    if faltas:
        sys.exit("Não há fotos suficientes nas bases de origem:\n  " + "\n  ".join(faltas))
    return escolhas


def imprimir_distribuicao(linhas):
    for (classe, dataset, categoria), n in sorted(dados.contar_distribuicao(linhas).items()):
        print(f"  {dados.texto_classe(classe):13} {dados.nome_dataset(dataset):21} {categoria:20} {n:4}")


def pasta_tem_arquivos(pasta):
    return pasta.exists() and any(not p.name.startswith(".") for p in pasta.iterdir())


def montar_base_inicial():
    if config.ARQ_METADADOS.exists():
        sys.exit("A base já existe (base_imagens/metadados.csv). "
                 "Para copiar as imagens que ele lista, use --remontar; "
                 "para acrescentar imagens numa melhoria, use --acrescentar.")
    for pasta in config.PASTAS_CLASSE.values():
        if pasta_tem_arquivos(pasta):
            sys.exit(f"A pasta {pasta} já tem arquivos, mas não há metadados. Esvazie a pasta e rode de novo.")

    indice = fontes.indexar_fontes()
    partes = [(classe, dataset, fontes.codigos_padrao(dataset, categoria, indice), quantidade,
               f"{config.SEMENTE}-{classe}-{dataset}-{categoria}")
              for classe, dataset, categoria, quantidade in config.PLANO_BASE]
    escolhas = sortear_partes(indice, partes)

    linhas = [fontes.copiar_para_base(foto, classe) for classe, foto in escolhas]
    dados.gravar_metadados(linhas)
    dados.gravar_distribuicao_planejada(dados.contar_distribuicao(linhas))
    print(f"\nBase montada: {len(linhas)} imagens.")
    imprimir_distribuicao(linhas)
    print("\nPróximo passo: python triagem.py")


def remontar_base():
    """Refaz a base copiando exatamente as imagens que o metadados.csv lista (Etapas 1 a 3).

    Para quem clona o repositório: os metadados e o teste fixo vêm no Git, as imagens não.
    Não sorteia nada, então a base sai idêntica à que gerou os resultados registrados —
    inclusive a substituição que a triagem fez.
    """
    if not config.ARQ_METADADOS.exists():
        sys.exit(f"Não encontrei {config.ARQ_METADADOS}. Para montar a base do zero, "
                 "rode python montar_base.py sem --remontar.")
    linhas = dados.ler_metadados()
    print(f"Procurando nas bases de origem as {len(linhas)} imagens do metadados.csv...")
    copiadas, ja_estavam, sem_origem = fontes.copiar_da_lista(linhas)

    if sem_origem:
        print(f"\nNão achei {len(sem_origem)} imagem(ns) nas bases de origem:")
        for nome in sem_origem[:10]:
            print(f"  {nome}")
        if len(sem_origem) > 10:
            print(f"  ... e mais {len(sem_origem) - 10}")
        sys.exit("Confira se as duas bases de origem estão completas. Veja o README.")

    print(f"\nBase remontada: {len(copiadas)} imagem(ns) copiada(s), "
          f"{len(ja_estavam)} já estava(m) na pasta.")
    imprimir_distribuicao(linhas)
    print("\nAs imagens são as mesmas que geraram os resultados registrados: a triagem já foi "
          "aplicada e não precisa ser refeita.")
    print("Próximo passo: python montar_bairro.py")


def separar_categorias(texto):
    """Separa os códigos de --categorias por (classe, dataset de origem)."""
    alvo = {}
    for parte in texto.split(","):
        escrito = parte.strip()
        if not escrito:
            continue
        nome_voc = escrito.lower()
        if re.fullmatch(r"n\d{8}", escrito):
            codigo = escrito
            chave = ("cachorro" if e_raca_de_cachorro(codigo) else "nao_cachorro", "imagenet")
        elif nome_voc == config.VOC_CACHORRO:
            codigo, chave = nome_voc, ("cachorro", "voc")
        elif nome_voc in config.VOC_OUTROS_ANIMAIS or nome_voc == config.VOC_PESSOAS:
            codigo, chave = nome_voc, ("nao_cachorro", "voc")
        else:
            sys.exit(f"Categoria desconhecida em --categorias: {escrito}\n"
                     "Use códigos do ImageNet (ex.: n02114367) ou nomes do Pascal VOC "
                     "(dog, bird, cat, cow, horse, sheep, person).")
        alvo.setdefault(chave, set()).add(codigo)
    return alvo


def acrescentar(quantidade, texto_categorias):
    """Melhoria: acrescenta a mesma quantidade em cada classe e em cada dataset de origem."""
    linhas = dados.ler_metadados()
    if not linhas:
        sys.exit("Ainda não há base. Rode primeiro: python montar_base.py")
    alvo = separar_categorias(texto_categorias or "")
    indice = fontes.indexar_fontes()

    semente = f"{config.SEMENTE}-acrescimo-{len(linhas)}"
    partes = []
    for classe in ("cachorro", "nao_cachorro"):
        for dataset in ("imagenet", "voc"):
            if (classe, dataset) in alvo:
                partes.append((classe, dataset, sorted(alvo[(classe, dataset)]), quantidade,
                               f"{semente}-{classe}-{dataset}"))
                continue
            # sem categorias citadas: divide igualmente entre as categorias reais do plano
            categorias = [cat for c, d, cat, _ in config.PLANO_BASE if c == classe and d == dataset]
            for i, categoria in enumerate(categorias):
                parte = quantidade // len(categorias) + (1 if i < quantidade % len(categorias) else 0)
                if parte:
                    partes.append((classe, dataset, fontes.codigos_padrao(dataset, categoria, indice), parte,
                                   f"{semente}-{classe}-{dataset}-{categoria}"))
    escolhas = sortear_partes(indice, partes)

    novas = [fontes.copiar_para_base(foto, classe) for classe, foto in escolhas]
    planejada = dados.ler_distribuicao_planejada() or dados.contar_distribuicao(linhas)
    dados.gravar_metadados(linhas + novas)   # as novas entram no fim; o teste fixo não muda
    dados.gravar_distribuicao_planejada(planejada + dados.contar_distribuicao(novas))
    print(f"\nAcrescentadas {len(novas)} imagens (vão só para o bairro; o teste fixo continua o mesmo):")
    imprimir_distribuicao(novas)
    print("\nPróximo passo: python triagem.py (as imagens novas também passam pela triagem)")


def main():
    parser = argparse.ArgumentParser(description="Monta a base única de imagens (Parte 1, Etapas 1 a 3).")
    parser.add_argument("--acrescentar", type=int, metavar="N",
                        help="melhoria: acrescenta N imagens de cada dataset em cada classe (4 × N no total)")
    parser.add_argument("--categorias",
                        help="só com --acrescentar: códigos do ImageNet ou nomes do Pascal VOC, separados por vírgula")
    parser.add_argument("--remontar", action="store_true",
                        help="copia as imagens que o metadados.csv já lista, sem sortear "
                             "(para quem clonou o repositório)")
    args = parser.parse_args()
    if args.categorias and args.acrescentar is None:
        parser.error("--categorias só vale junto com --acrescentar")
    if args.remontar and args.acrescentar is not None:
        parser.error("--remontar não vai junto com --acrescentar")
    if args.remontar:
        remontar_base()
    elif args.acrescentar is not None:
        if args.acrescentar <= 0:
            parser.error("--acrescentar precisa ser maior que zero")
        acrescentar(args.acrescentar, args.categorias)
    else:
        montar_base_inicial()


if __name__ == "__main__":
    main()
