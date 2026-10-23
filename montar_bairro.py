"""Parte 1 · Etapas 5 e 6: extrai os 1.280 números de cada imagem e separa o bairro do teste fixo.

Uso:
    python montar_bairro.py                          extrator EfficientNetB0
    python montar_bairro.py --extrator mobilenetv2   rodada de troca do extrator

Na primeira execução, o split 80/20 (estratificado por classe + dataset de origem,
random_state=42) sorteia as 80 imagens de teste e grava a lista em
base_imagens/teste_fixo.txt. Nas execuções seguintes, o código lê essa lista: o teste
continua o mesmo, e as imagens acrescentadas numa melhoria vão só para o bairro.

Monte o bairro de novo só quando mudarem os dados ou o extrator.
"""
import argparse
import sys
from collections import Counter

import config
import dados
from extrator import EXTRATORES, carregar_extrator, extrair
from triagem import imprimir_situacao, situacao_triagem


def resumo(linhas):
    contagem = Counter((l["classe"], l["dataset"]) for l in linhas)
    partes = [f"{dados.texto_classe(classe)} · {dados.nome_dataset(dataset)}: {n}"
              for (classe, dataset), n in sorted(contagem.items())]
    return f"{len(linhas)} imagens ({'; '.join(partes)})"


def main():
    parser = argparse.ArgumentParser(description="Extrai as características e monta o bairro (Parte 1, Etapas 5 e 6).")
    parser.add_argument("--extrator", choices=EXTRATORES, default=config.EXTRATOR_PADRAO)
    parser.add_argument("--ignorar-triagem", action="store_true",
                        help="monta mesmo com a triagem incompleta (não recomendado)")
    args = parser.parse_args()

    linhas = dados.ler_metadados()
    if not linhas:
        sys.exit("Ainda não há base. Rode primeiro: python montar_base.py")
    situacao = situacao_triagem()
    if not situacao["concluida"]:
        imprimir_situacao(situacao)
        if not args.ignorar_triagem:
            sys.exit("\nA triagem vem antes da extração. Termine a triagem (ou use --ignorar-triagem).")
        print("\nAtenção: montando o bairro com a triagem incompleta (--ignorar-triagem).")
    faltando = [l["arquivo"] for l in linhas if not dados.caminho_imagem(l).exists()]
    if faltando:
        sys.exit(f"Imagens dos metadados que não estão na pasta da classe: {', '.join(faltando[:10])}"
                 + (" ..." if len(faltando) > 10 else ""))

    # Etapa 5: cada imagem vira 1.280 números, na ordem dos metadados.
    print(f"Extraindo as características de {len(linhas)} imagens com o {args.extrator}...")
    extrator, preparar = carregar_extrator(args.extrator)
    X_num = extrair(extrator, preparar, [dados.caminho_imagem(l) for l in linhas], mostrar_progresso=True)

    # Etapa 6: o split 80/20 sai na primeira execução; depois, só a leitura de teste_fixo.txt.
    teste, criado, avisos = dados.definir_teste_fixo(linhas)
    for aviso in avisos:
        print(aviso)
    no_teste = set(teste)
    i_bairro = [i for i, l in enumerate(linhas) if l["arquivo"] not in no_teste]
    i_teste = [i for i, l in enumerate(linhas) if l["arquivo"] in no_teste]
    linhas_bairro = [linhas[i] for i in i_bairro]
    linhas_teste = [linhas[i] for i in i_teste]

    # Só os vetores do bairro entram na base de referência; os do teste fixo ficam à parte.
    dados.salvar_vetores(config.ARQ_BAIRRO, X_num[i_bairro], linhas_bairro, args.extrator)
    dados.salvar_vetores(config.ARQ_TESTE, X_num[i_teste], linhas_teste, args.extrator)

    origem_teste = "sorteado agora e gravado em" if criado else "lido de"
    print(f"\nBairro: {resumo(linhas_bairro)}")
    print(f"Teste fixo ({origem_teste} base_imagens/teste_fixo.txt): {resumo(linhas_teste)}")
    print(f"Vetores de {X_num.shape[1]} números guardados em vetores/bairro.npz e vetores/teste.npz.")
    print('\nPróximo passo: python avaliar.py --rodada N --mudanca "o que mudou"')


if __name__ == "__main__":
    main()
