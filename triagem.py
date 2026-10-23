"""Parte 1 · Etapa 4: triagem e controle de qualidade com YOLO, antes da extração.

Uso:
    python triagem.py             roda o YOLO nas imagens ainda não triadas e gera o relatório
    python triagem.py --aplicar   aplica as decisões escritas em triagem/relatorio.csv
    python triagem.py --situacao  só mostra se a triagem terminou

O ciclo:
1. python triagem.py → o YOLO sinaliza os casos suspeitos e sorteia uma amostra das
   imagens não sinalizadas (o YOLO também erra).
2. Abra triagem/relatorio.html para ver as fotos e escreva a decisão de cada linha na
   coluna "decisao" de triagem/relatorio.csv: manter, corrigir (o rótulo), substituir
   (mesma categoria e mesmo dataset) ou revisar (fica pendente).
3. python triagem.py --aplicar → aplica as decisões. Se houve substituição, volte ao
   passo 1: as imagens novas também passam pela triagem.

A triagem termina quando não há imagens sem triagem nem casos pendentes, a
distribuição planejada está preservada e a amostra foi conferida. O objetivo não
é zerar os alertas do YOLO: nem toda detecção de outro objeto é um erro.
"""
import argparse
import html
import math
import os
import random
import shutil
import sys
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import quote

import config
import dados
import fontes

ARQ_DETECCOES = config.PASTA_TRIAGEM / "deteccoes.csv"     # tudo o que o YOLO viu, imagem por imagem
ARQ_RELATORIO = config.PASTA_TRIAGEM / "relatorio.csv"     # casos sinalizados + amostra, com as decisões
ARQ_RELATORIO_HTML = config.PASTA_TRIAGEM / "relatorio.html"

CAMPOS_DETECCOES = ["arquivo", "classe", "deteccoes", "sinalizada"]
CAMPOS_RELATORIO = ["arquivo", "classe", "dataset", "categoria", "categoria_original", "tipo",
                    "motivo", "deteccoes", "decisao", "aplicada", "observacao"]
CAMPOS_DESCARTADAS = ["arquivo", "classe", "dataset", "categoria_original", "motivo", "data"]
DECISOES = ("manter", "corrigir", "substituir", "revisar")


# ---------------------------------------------------------------- YOLO
def detectar(caminhos):
    """Roda o YOLO em cada imagem e devolve [(objeto, confiança), ...], da maior confiança para a menor."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("A triagem precisa da Ultralytics: pip install -r requirements-triagem.txt")
    modelo = YOLO(config.YOLO_MODELO)   # na primeira vez, baixa o yolov8n.pt (~6 MB)
    for i, caminho in enumerate(caminhos, start=1):
        resultado = modelo.predict(str(caminho), conf=config.YOLO_CONFIANCA_MINIMA, verbose=False)[0]
        nomes = [resultado.names[int(c)] for c in resultado.boxes.cls.tolist()]
        confiancas = resultado.boxes.conf.tolist()
        if i % 25 == 0 or i == len(caminhos):
            print(f"  YOLO: {i}/{len(caminhos)} imagens")
        yield sorted(zip(nomes, confiancas), key=lambda deteccao: -deteccao[1])


def motivos_da_sinalizacao(classe, deteccoes):
    """As situações suspeitas da Etapa 4.1 (Fluxo completo do plano A)."""
    caes = [conf for nome, conf in deteccoes if nome == "dog"]
    outros_animais = [f"{nome} {conf:.2f}" for nome, conf in deteccoes if nome in config.ANIMAIS_COCO]
    motivos = []
    if classe == "cachorro":
        if not caes:
            motivos.append("nenhum cachorro detectado")
        elif max(caes) < config.YOLO_CONFIANCA_INCERTA:
            motivos.append(f"detecção incerta: cachorro com confiança {max(caes):.2f}")
        if outros_animais:
            motivos.append("outro animal junto: " + ", ".join(outros_animais))
    elif caes:
        if max(caes) < config.YOLO_CONFIANCA_INCERTA:
            motivos.append(f"detecção incerta: possível cachorro ({max(caes):.2f})")
        else:
            motivos.append(f"cachorro detectado ({max(caes):.2f})")
    return motivos


def texto_deteccoes(deteccoes):
    return "; ".join(f"{nome} {conf:.2f}" for nome, conf in deteccoes) or "nada detectado"


def linha_do_relatorio(linha, tipo, motivo, deteccoes):
    return {
        "arquivo": linha["arquivo"],
        "classe": linha["classe"],
        "dataset": linha["dataset"],
        "categoria": dados.categoria_planejada(linha["classe"], linha["dataset"], linha["categoria_original"]),
        "categoria_original": dados.nome_categoria(linha["dataset"], linha["categoria_original"]),
        "tipo": tipo,
        "motivo": motivo,
        "deteccoes": deteccoes,
        "decisao": "",
        "aplicada": "",
        "observacao": "",
    }


def rodar_triagem():
    linhas = dados.ler_metadados()
    if not linhas:
        sys.exit("Ainda não há base. Rode primeiro: python montar_base.py")
    deteccoes = dados.ler_csv(ARQ_DETECCOES)
    triadas = {d["arquivo"] for d in deteccoes}
    pendentes = [linha for linha in linhas if linha["arquivo"] not in triadas]

    if not pendentes:
        print("Todas as imagens da base já passaram pelo YOLO.")
    else:
        print(f"Triagem de {len(pendentes)} imagem(ns) com o YOLO...")
        relatorio = dados.ler_csv(ARQ_RELATORIO)
        nao_sinalizadas = []
        caminhos = [dados.caminho_imagem(linha) for linha in pendentes]
        for linha, achados in zip(pendentes, detectar(caminhos)):
            motivos = motivos_da_sinalizacao(linha["classe"], achados)
            texto = texto_deteccoes(achados)
            deteccoes.append({"arquivo": linha["arquivo"], "classe": linha["classe"],
                              "deteccoes": texto, "sinalizada": "sim" if motivos else "não"})
            if motivos:
                relatorio.append(linha_do_relatorio(linha, "sinalizada", "; ".join(motivos), texto))
            else:
                nao_sinalizadas.append((linha, texto))

        # Amostra das não sinalizadas, para conferir à mão: o YOLO também erra.
        quantidade = math.ceil(len(nao_sinalizadas) * config.FRACAO_AMOSTRA)
        sorteio = random.Random(f"{config.SEMENTE}-amostra-{len(triadas)}")
        for linha, texto in sorteio.sample(nao_sinalizadas, quantidade):
            relatorio.append(linha_do_relatorio(linha, "amostra", "conferir à mão (não sinalizada)", texto))

        dados.gravar_csv(ARQ_DETECCOES, deteccoes, CAMPOS_DETECCOES)
        dados.gravar_csv(ARQ_RELATORIO, relatorio, CAMPOS_RELATORIO)
        print(f"Sinalizadas: {len(pendentes) - len(nao_sinalizadas)} · amostra para conferir: {quantidade}")
    gerar_html()
    imprimir_situacao(situacao_triagem())


# ---------------------------------------------------------------- decisões
def normalizar_decisao(texto):
    """Aceita variações como "Manter", "corrigir o rótulo" ou "revisar adicionalmente"."""
    simples = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode().strip().lower()
    for decisao in DECISOES:
        if simples.startswith(decisao[:5]):
            return decisao
    return ""


def corrigir_rotulo(linha):
    """Leva a imagem para a pasta da outra classe e corrige a classe nos metadados."""
    nova_classe = "nao_cachorro" if linha["classe"] == "cachorro" else "cachorro"
    origem = config.PASTAS_CLASSE[linha["classe"]] / linha["arquivo"]
    destino = config.PASTAS_CLASSE[nova_classe] / linha["arquivo"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    if origem.exists():
        shutil.move(str(origem), str(destino))
    linha["classe"] = nova_classe


def descartar(linha, motivo):
    """Tira a imagem da base sem apagar: ela vai para triagem/descartadas/ e não volta a ser sorteada."""
    origem = dados.caminho_imagem(linha)
    config.PASTA_DESCARTADAS.mkdir(parents=True, exist_ok=True)
    if origem.exists():
        shutil.move(str(origem), str(config.PASTA_DESCARTADAS / linha["arquivo"]))
    registro = dados.ler_csv(config.ARQ_DESCARTADAS)
    registro.append({**linha, "motivo": motivo, "data": date.today().isoformat()})
    dados.gravar_csv(config.ARQ_DESCARTADAS, registro, CAMPOS_DESCARTADAS)


def aplicar_decisoes():
    relatorio = dados.ler_csv(ARQ_RELATORIO)
    if not relatorio:
        sys.exit("Não há relatório de triagem. Rode primeiro: python triagem.py")
    linhas = dados.ler_metadados()
    posicao = {linha["arquivo"]: i for i, linha in enumerate(linhas)}
    no_teste = set(dados.ler_teste_fixo())
    ja_usadas = fontes.arquivos_ja_usados()
    indice = None            # as bases de origem só são lidas se houver substituição
    aplicadas = Counter()

    for item in relatorio:
        if item.get("aplicada") == "sim":
            continue
        decisao = normalizar_decisao(item.get("decisao"))
        if decisao in ("", "revisar"):
            continue                      # continua pendente
        arquivo = item["arquivo"]
        if arquivo not in posicao:
            item["observacao"] = "a imagem não está mais na base"
            item["aplicada"] = "sim"
            continue
        if arquivo in no_teste:
            print(f"Atenção: {arquivo} está no teste fixo. Mudar essa imagem muda a prova.")
        linha = linhas[posicao[arquivo]]

        if decisao == "corrigir":
            corrigir_rotulo(linha)
            item["classe"] = linha["classe"]
            item["categoria"] = dados.categoria_planejada(linha["classe"], linha["dataset"],
                                                          linha["categoria_original"])
            item["observacao"] = f"rótulo corrigido para {dados.texto_classe(linha['classe'])}"
        elif decisao == "substituir":
            if indice is None:
                indice = fontes.indexar_fontes()
            substituta = fontes.sortear_substituta(indice, linha, ja_usadas)
            if substituta is None:
                print(f"Não achei substituta para {arquivo} (mesma categoria e mesmo dataset). Ficou pendente.")
                continue
            descartar(linha, item.get("motivo", ""))
            nova = fontes.copiar_para_base(substituta, linha["classe"])
            linhas[posicao[arquivo]] = nova      # a substituta fica no mesmo lugar dos metadados
            posicao[nova["arquivo"]] = posicao.pop(arquivo)
            ja_usadas.add(nova["arquivo"])
            item["observacao"] = f"substituída por {nova['arquivo']}"
        item["decisao"] = decisao
        item["aplicada"] = "sim"
        aplicadas[decisao] += 1

    dados.gravar_metadados(linhas)
    dados.gravar_csv(ARQ_RELATORIO, relatorio, CAMPOS_RELATORIO)
    resumo = ", ".join(f"{decisao}: {n}" for decisao, n in aplicadas.items())
    print("Decisões aplicadas: " + (resumo or "nenhuma"))
    if aplicadas["substituir"]:
        print("As substitutas ainda não passaram pelo YOLO: rode python triagem.py de novo.")
    gerar_html()
    imprimir_situacao(situacao_triagem())


# ---------------------------------------------------------------- critério de parada
def situacao_triagem():
    """Confere o critério de parada da Etapa 4.3."""
    linhas = dados.ler_metadados()
    triadas = {d["arquivo"] for d in dados.ler_csv(ARQ_DETECCOES)}
    sem_triagem = [linha["arquivo"] for linha in linhas if linha["arquivo"] not in triadas]
    pendentes = [item for item in dados.ler_csv(ARQ_RELATORIO) if item.get("aplicada") != "sim"]
    planejada = dados.ler_distribuicao_planejada()
    if planejada:
        diferencas = dados.diferencas_distribuicao(dados.contar_distribuicao(linhas), planejada)
    else:
        diferencas = ["não encontrei base_imagens/distribuicao_planejada.csv"]
    return {
        "imagens": len(linhas),
        "sem_triagem": sem_triagem,
        "pendentes": pendentes,
        "diferencas": diferencas,
        "concluida": bool(linhas) and not sem_triagem and not pendentes and not diferencas,
    }


def imprimir_situacao(situacao):
    sinalizadas = sum(1 for item in situacao["pendentes"] if item.get("tipo") == "sinalizada")
    amostra = len(situacao["pendentes"]) - sinalizadas
    print("\nSituação da triagem")
    print(f"  Imagens na base: {situacao['imagens']}")
    print(f"  Ainda sem triagem (YOLO): {len(situacao['sem_triagem'])}")
    print(f"  Casos sem decisão aplicada: {sinalizadas} sinalizado(s) e {amostra} da amostra")
    if situacao["diferencas"]:
        print("  Distribuição planejada: não preservada")
        for diferenca in situacao["diferencas"]:
            print(f"    - {diferenca}")
    else:
        print("  Distribuição planejada: preservada")

    if situacao["concluida"]:
        print("→ Triagem concluída. A base está pronta para a extração: python montar_bairro.py")
    elif situacao["sem_triagem"]:
        print("→ Rode python triagem.py para passar as imagens novas pelo YOLO.")
    elif situacao["pendentes"]:
        print("→ Veja as fotos em triagem/relatorio.html, escreva as decisões em triagem/relatorio.csv "
              "e rode python triagem.py --aplicar.")
    else:
        print("→ A base saiu da distribuição planejada. Ajuste (por exemplo, substituindo uma imagem) "
              "para voltar ao plano.")


# ---------------------------------------------------------------- relatório em HTML
ESTILO_HTML = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; background: #fafafa; }
h1 { margin-bottom: 4px; } h2 { margin-top: 32px; }
.grade { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }
.cartao { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; font-size: 13px; }
.cartao img { width: 100%; height: 180px; object-fit: contain; background: #eee; border-radius: 4px; }
.sem { height: 180px; display: grid; place-items: center; background: #eee; color: #777; }
.motivo { color: #a33; } .amostra { color: #2a5d9f; } .decisao { font-weight: 600; }
code { background: #eee; padding: 1px 4px; border-radius: 3px; }
"""


def cartao_html(item):
    caminho = dados.localizar_imagem(item["arquivo"])
    if caminho:
        relativo = quote(Path(os.path.relpath(caminho, config.PASTA_TRIAGEM)).as_posix())
        imagem = f"<img src='{relativo}' alt=''>"
    else:
        imagem = "<div class='sem'>imagem não encontrada</div>"
    classe_motivo = "amostra" if item.get("tipo") == "amostra" else "motivo"
    decisao = item.get("decisao") or "sem decisão"
    if item.get("observacao"):
        decisao += f" · {item['observacao']}"
    campos = [
        f"<b>{html.escape(item['arquivo'])}</b>",
        html.escape(f"{dados.texto_classe(item.get('classe', ''))} · {dados.nome_dataset(item.get('dataset', ''))}"),
        html.escape(f"{item.get('categoria', '')} · {item.get('categoria_original', '')}"),
        f"<span class='{classe_motivo}'>{html.escape(item.get('motivo', ''))}</span>",
        "YOLO: " + html.escape(item.get("deteccoes", "")),
        f"<span class='decisao'>Decisão: {html.escape(decisao)}</span>",
    ]
    return f"<div class='cartao'>{imagem}<div>{'<br>'.join(campos)}</div></div>"


def gerar_html():
    relatorio = dados.ler_csv(ARQ_RELATORIO)
    pendentes = [item for item in relatorio if item.get("aplicada") != "sim"]
    decididos = [item for item in relatorio if item.get("aplicada") == "sim"]
    partes = [
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Triagem com YOLO</title>",
        f"<style>{ESTILO_HTML}</style></head><body>",
        "<h1>Triagem com YOLO</h1>",
        f"<p>{len(pendentes)} caso(s) sem decisão aplicada · {len(decididos)} decidido(s). "
        "Escreva a decisão na coluna <code>decisao</code> de <code>triagem/relatorio.csv</code> "
        "(manter, corrigir, substituir ou revisar) e rode <code>python triagem.py --aplicar</code>.</p>",
    ]
    for titulo, grupo in (("Sem decisão aplicada", pendentes), ("Decididos", decididos)):
        partes.append(f"<h2>{titulo} ({len(grupo)})</h2><div class='grade'>")
        partes += [cartao_html(item) for item in grupo]
        partes.append("</div>")
    partes.append("</body></html>")
    ARQ_RELATORIO_HTML.parent.mkdir(parents=True, exist_ok=True)
    ARQ_RELATORIO_HTML.write_text("\n".join(partes), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Triagem da base com YOLO (Parte 1, Etapa 4).")
    opcoes = parser.add_mutually_exclusive_group()
    opcoes.add_argument("--aplicar", action="store_true", help="aplica as decisões de triagem/relatorio.csv")
    opcoes.add_argument("--situacao", action="store_true", help="só mostra se a triagem terminou")
    args = parser.parse_args()
    if args.aplicar:
        aplicar_decisoes()
    elif args.situacao:
        gerar_html()
        imprimir_situacao(situacao_triagem())
    else:
        rodar_triagem()


if __name__ == "__main__":
    main()
