"""Avaliação dos dois modelos no teste fixo (RF8) e registro das rodadas (RF9).

Uso:
    python avaliar.py
        só mostra, para o Fuzzy k-NN e para o SVM, a matriz, a acurácia, a meta e a área cinzenta
    python avaliar.py --rodada 1 --mudanca "ponto de partida: bairro com 320 (160 + 160)"
        registra a rodada na tabela de cada modelo, com o mesmo número, e copia as imagens
        de FP e FN de cada um para a pasta da rodada dele (resultados_fuzzyknn/ e resultados_svm/)
    python avaliar.py --rodada 4 --mudanca "o que mudou no código do SVM" --mudou svm
        rodada de código que mexeu só no SVM: o Fuzzy k-NN precisa repetir os números da
        rodada anterior. Sem --mudou, com o mesmo bairro, os dois precisam repetir.
    python avaliar.py --rodada 4 --mudanca "o que mudou" --substituir
        grava de novo uma rodada que já existe, nas duas tabelas (só com este pedido explícito)

Regras das rodadas: uma mudança por rodada (dados, extrator ou código), sempre testada nas
mesmas 80 imagens, nos dois modelos. O k, o C e o gamma ficam fixos na V2. O SVM é treinado
no bairro nesta hora; nada dele fica guardado em arquivo.
"""
import argparse
import sys

import avaliacao
import config
import dados
import fuzzyknn
import rodadas
import svm
from avaliacao import percentual
from regua import limites_area_cinzenta


def imprimir_matriz(casas):
    print("Matriz de confusão")
    print(f"{'':16}{'disse cachorro':>16}{'disse não':>12}")
    print(f"{'era cachorro':16}{'VP ' + str(casas['VP']):>16}{'FN ' + str(casas['FN']):>12}")
    print(f"{'não era':16}{'FP ' + str(casas['FP']):>16}{'VN ' + str(casas['VN']):>12}")


def imprimir_modelo(titulo, medidas, faixa):
    """A matriz, a acurácia, a meta e a área cinzenta de um modelo."""
    casas, total = medidas["casas"], len(medidas["resultados"])
    print(f"\n== {titulo} ==")
    imprimir_matriz(casas)
    print(f"\nAcurácia: ({casas['VP']} + {casas['VN']}) ÷ {total} = {percentual(medidas['acuracia'])}")
    atingida, criterios = avaliacao.meta_atingida(medidas["acuracia"], casas["FN"], casas["FP"])
    marcas = " · ".join(f"{texto} {'✓' if ok else '✗'}" for ok, texto in criterios)
    print(f"Meta: {marcas} → {'atingida' if atingida else 'não atingida'}")
    galeria = avaliacao.galeria_area_cinzenta(medidas["resultados"])
    na_faixa = sum(len(lista) for lista in galeria.values())
    print(f"Área cinzenta ({faixa}): {na_faixa} de {total} imagens")
    print("  " + " · ".join(f"{situacao} {len(galeria[situacao])}" for situacao in avaliacao.CASAS))


def relativo(caminho):
    """O caminho como aparece no terminal: a partir da pasta do projeto."""
    try:
        return caminho.relative_to(config.RAIZ).as_posix()
    except ValueError:
        return str(caminho)


def main():
    parser = argparse.ArgumentParser(description="Avalia os dois modelos no teste fixo e registra a rodada.")
    parser.add_argument("--rodada", type=int, help="número da rodada a registrar (ex.: 1)")
    parser.add_argument("--mudanca", help='o que mudou nesta rodada (ex.: "+ 80 imagens")')
    parser.add_argument("--mudou", nargs="+", choices=rodadas.MODELOS,
                        help="numa rodada de código: o modelo (ou os modelos) cujo código mudou")
    parser.add_argument("--substituir", action="store_true",
                        help="grava de novo uma rodada que já existe, nas duas tabelas")
    args = parser.parse_args()
    if args.rodada is not None and not args.mudanca:
        parser.error('--rodada precisa de --mudanca "o que mudou nesta rodada"')
    if args.rodada is None and (args.mudou or args.substituir):
        parser.error("--mudou e --substituir só valem junto com --rodada")
    if not config.ARQ_BAIRRO.exists() or not config.ARQ_TESTE.exists():
        sys.exit("O bairro ainda não foi montado. Rode primeiro: python montar_bairro.py")

    bairro = dados.carregar_vetores(config.ARQ_BAIRRO)
    teste = dados.carregar_vetores(config.ARQ_TESTE)
    for aviso in dados.conferir_vetores(dados.ler_metadados(), bairro, teste):
        print(f"Atenção: {aviso}")

    medidas_fuzzyknn = fuzzyknn.avaliar_teste_fixo(bairro, teste)
    print("Treinando o SVM no bairro...")
    medidas_svm = svm.avaliar_teste_fixo(bairro, teste)

    print(f"\nTeste fixo: {len(teste['arquivos'])} imagens · bairro: {len(bairro['arquivos'])} imagens · "
          f"extrator: {bairro['extrator']} · assinatura do bairro: {rodadas.assinatura_bairro(bairro)}")
    minimo, maximo = limites_area_cinzenta()
    imprimir_modelo(f"{fuzzyknn.NOME} (k = {config.FUZZYKNN_K})", medidas_fuzzyknn,
                    f"pertinência a cachorro de {minimo} a {maximo}")
    minimo, maximo = svm.limites_em_porcentagem()
    imprimir_modelo(f"{svm.NOME} (C = {config.SVM_C:g}, gamma = {config.SVM_GAMMA})", medidas_svm,
                    f"probabilidade de cachorro de {minimo} a {maximo}")

    if args.rodada is not None:
        try:
            rodadas.registrar_rodada(args.rodada, args.mudanca,
                                     {"fuzzyknn": medidas_fuzzyknn, "svm": medidas_svm}, bairro,
                                     substituir=args.substituir, mudou=args.mudou or ())
        except rodadas.Trava as erro:
            sys.exit(f"\nA rodada {args.rodada} não foi registrada. {erro}")
        print()
        for modelo in rodadas.MODELOS:
            pasta = rodadas.caminhos(modelo)
            print(f"{rodadas.NOMES[modelo]}: rodada {args.rodada} registrada em {relativo(pasta.tabela)}; "
                  f"imagens de FP e FN copiadas para {relativo(pasta.pasta_rodada(args.rodada))}/.")
        print("Veja as matrizes, as rodadas, o comparativo e as galerias na visão admin: streamlit run app.py")


if __name__ == "__main__":
    main()
