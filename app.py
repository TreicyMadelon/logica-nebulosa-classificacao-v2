"""Aplicação em Streamlit (Parte 2 · Etapas 7 a 11, e os resultados da Etapa 12).

Rodar:  streamlit run app.py

Duas homes, uma por modelo, e a visão admin com três abas.

Home · Fuzzy k-NN: upload → a classe com o grau (pertinência da classe vencedora), a caixa
colorida da faixa em que a pertinência caiu (verde, amarela ou vermelha) com as duas
pertinências, e as fotos das 5 vizinhas que explicam de onde saiu o número.

Home · SVM: o mesmo formato, com a probabilidade da classe vencedora em porcentagem e as 5
imagens do bairro que mais pesaram na conta do SVM. O SVM é treinado no bairro quando o
app precisa dele, sem arquivo guardado.

Cada home fecha com o seu bloco "Imagens anteriores", que fica só na sessão
(st.session_state, com nomes separados por home) e nunca entra na base.

A classe, a faixa, a cor da caixa e todo número na tela saem do grau de cachorro com duas
casas (regua.grau_na_tela): nada aqui arredonda por conta própria.

Visão admin: os resultados no teste fixo, lidos do que o avaliar.py registrou. Uma aba por
modelo (rodadas, matriz de confusão, acurácia, meta e galeria da área cinzenta) e a aba
Comparativo, com a mesma rodada dos dois lado a lado.
"""
import hashlib
import io

import streamlit as st

import avaliacao
import config
import dados
import fuzzyknn
import rodadas
import svm
from avaliacao import percentual
from extrator import carregar_extrator, extrair
from regua import formatar_grau, limites_area_cinzenta

MODELOS = rodadas.MODELOS
NOMES = rodadas.NOMES

# A faixa como cada modelo escreve: pertinência com duas casas, probabilidade em porcentagem.
# Os números vêm do config: mudar a faixa lá muda estes textos.
LIMITES = {"fuzzyknn": limites_area_cinzenta(), "svm": svm.limites_em_porcentagem()}


def chave(dono, nome, *extra):
    """Nome na memória da sessão e chave de widget, sempre com o dono (o modelo ou o
    comparativo): historico_fuzzyknn, envio_svm, rever_fuzzyknn_<id>. Assim as duas homes
    não dividem estado (proteção 10)."""
    return "_".join([nome, dono, *(str(parte) for parte in extra)])


# ---------------------------------------------------------------- carregamento
@st.cache_resource(show_spinner="Carregando o bairro...")
def carregar_bairro(marca):
    """marca = horário da última montagem: se o bairro for montado de novo, ele é recarregado."""
    return dados.carregar_vetores(config.ARQ_BAIRRO)


@st.cache_resource(show_spinner="Treinando o SVM no bairro (nada fica guardado em arquivo)...")
def carregar_svm(marca):
    """O fit do SVM no bairro, refeito quando o bairro é montado de novo."""
    bairro = carregar_bairro(marca)
    return svm.treinar(bairro["X"], bairro["y"])


@st.cache_resource(show_spinner="Carregando o extrator (na primeira vez, o Keras baixa os pesos)...")
def carregar_modelo(nome):
    return carregar_extrator(nome)


def analisar(modelo, conteudo, bairro, marca):
    """Etapas 8 a 10: o mesmo extrator do bairro, o modelo da home e a decisão."""
    extrator, preparar = carregar_modelo(bairro["extrator"])
    x_num = extrair(extrator, preparar, [io.BytesIO(conteudo)])[0]    # a imagem enviada → 1.280 números
    if modelo == "fuzzyknn":
        resultado = fuzzyknn.classificar(bairro["X"], bairro["y"], x_num, k=config.FUZZYKNN_K)
        indices = resultado["vizinhas"]
    else:
        resultado = svm.classificar(carregar_svm(marca), x_num)
        indices = resultado["mais_pesaram"]
    # As 5 imagens do bairro que explicam o grau: (arquivo, rótulo).
    resultado["fotos"] = [(bairro["arquivos"][i], int(bairro["y"][i])) for i in indices]
    return resultado


# ---------------------------------------------------------------- pedaços da tela
def mostrar_imagem(caminho, legenda, largura):
    if caminho is not None:
        st.image(str(caminho), caption=legenda, width=largura)
    else:
        st.caption(f"{legenda} (imagem não encontrada)")


def mostrar_fotos(fotos, largura=130):
    """fotos: [(arquivo, rótulo)], na ordem da tela (da mais próxima, ou da que mais pesou)."""
    for posicao, (coluna, (arquivo, rotulo)) in enumerate(zip(st.columns(len(fotos)), fotos), start=1):
        with coluna:
            mostrar_imagem(dados.localizar_imagem(arquivo), f"{posicao}ª · {dados.texto_classe(rotulo)}", largura)


def plural(n, singular, plural_):
    return f"{n} {singular if n == 1 else plural_}"


def faixas(modelo):
    """Em qual das três faixas o grau de cachorro caiu: título, regra da faixa e a cor da caixa.

    A caixa só explica a decisão que o modelo já tomou; ela não muda a classe. A regra traz
    a faixa como o modelo escreve: "entre 0,40 e 0,60" no Fuzzy k-NN, "entre 40% e 60%" no SVM.
    """
    minimo, maximo = LIMITES[modelo]
    return {
        "cinzenta": ("⚠ Imagem na área cinzenta",
                     f"Toda imagem enviada pelo usuário que cair entre {minimo} e {maximo}",
                     "warning"),
        "cachorro": ("🐶 Imagem na área é cachorro",
                     f"Toda imagem enviada pelo usuário que cair acima de {maximo}",
                     "success"),
        "nao_cachorro": ("❌ Imagem na área não é cachorro",
                         f"Toda imagem enviada pelo usuário que cair abaixo de {minimo}",
                         "error"),
    }


def faixa_do_resultado(resultado):
    if resultado["area_cinzenta"]:
        return "cinzenta"
    return "cachorro" if resultado["classe"] == 1 else "nao_cachorro"


def texto_caixa(modelo, resultado):
    """O título da faixa, os dois graus (o vencedor primeiro) e a contagem das 5 imagens do bairro."""
    classe, fotos = resultado["classe"], resultado["fotos"]
    titulo, regra, _ = faixas(modelo)[faixa_do_resultado(resultado)]
    caes = sum(1 for _, rotulo in fotos if rotulo == 1)
    contagem = {1: plural(caes, "cachorro", "cachorros"),
                0: plural(len(fotos) - caes, "não cachorro", "não cachorros")}
    if modelo == "fuzzyknn":
        return (f"**{titulo}**  \n"
                f"{regra}  \n"
                f"pertinência da imagem: {fuzzyknn.texto_pertinencias(classe, resultado['p_cachorro'])}  \n"
                f"parecida com: {contagem[classe]} e {contagem[1 - classe]} "
                f"entre as {len(fotos)} imagens mais próximas")
    # As três partes com duas casas; o resto é a diferença já arredondada, então a conta fecha na tela.
    total = float(svm.formatar_pontuacao(resultado["pontuacao"]).replace("−", "-").replace(",", "."))
    cinco = float(svm.formatar_pontuacao(resultado["pontuacao_mostradas"]).replace("−", "-").replace(",", "."))
    lado = "cachorro" if classe == 1 else "não cachorro"
    return (f"**{titulo}**  \n"
            f"{regra}  \n"
            f"probabilidade da imagem: {svm.texto_probabilidades(classe, resultado['p_cachorro'])}  \n"
            f"pontuação do SVM: {svm.formatar_pontuacao(total)} "
            "(positivo puxa para cachorro, negativo para não cachorro)  \n"
            f"as {len(fotos)} imagens que mais puxaram para {lado} somam {svm.formatar_pontuacao(cinco)}; "
            f"os outros vetores de suporte e o termo fixo somam {svm.formatar_pontuacao(total - cinco)}")


def mostrar_resultado(modelo, item):
    """Etapa 11: sempre a classe com o grau, a caixa da faixa e as fotos das 5 imagens do bairro.

    As fotos aparecem nas três faixas, não só na cinzenta: são elas que explicam de onde saiu
    o grau, tanto numa resposta folgada quanto numa apertada. Nenhuma das caixas muda a
    classe: elas só explicam a decisão que o modelo já tomou.

    O título e a caixa saem do mesmo número em duas casas que decidiu a classe e a faixa,
    então a cor da caixa nunca desmente o número que ela mostra.
    """
    resultado = item["resultado"]
    col_imagem, col_texto = st.columns([1, 2])
    with col_imagem:
        st.image(item["conteudo"], caption=item["nome"], width=300)
    with col_texto:
        st.markdown(f"## {resultado['mensagem']}")
        caixa = getattr(st, faixas(modelo)[faixa_do_resultado(resultado)][2])
        caixa(texto_caixa(modelo, resultado))
    quantas = len(resultado["fotos"])
    if modelo == "fuzzyknn":
        st.markdown(f"**As {quantas} imagens do bairro mais próximas** "
                    "(da mais próxima para a mais distante)")
    else:
        lado = "cachorro" if resultado["classe"] == 1 else "não cachorro"
        st.markdown(f"**As {quantas} imagens do bairro que mais puxaram para {lado}** "
                    "(da que mais pesou para a que menos pesou; os outros vetores de suporte também contam)")
    mostrar_fotos(resultado["fotos"])


def selecionar(modelo, ident):
    st.session_state[chave(modelo, "selecionada")] = ident


def mostrar_imagens_anteriores(modelo):
    historico = st.session_state[chave(modelo, "historico")]
    if not historico:
        return
    st.divider()
    st.subheader("Imagens anteriores")
    st.caption("Clique para rever o resultado. Elas ficam só nesta sessão: somem ao fechar ou "
               "recarregar a página e não entram na base.")
    recentes = list(reversed(historico))
    selecionada = st.session_state[chave(modelo, "selecionada")]
    por_linha = 6
    for inicio in range(0, len(recentes), por_linha):
        for coluna, item in zip(st.columns(por_linha), recentes[inicio:inicio + por_linha]):
            with coluna:
                st.image(item["conteudo"], width=120)
                st.button(item["resultado"]["mensagem"], key=chave(modelo, "rever", item["id"]),
                          help=item["nome"], on_click=selecionar, args=(modelo, item["id"]),
                          disabled=item["id"] == selecionada)


# ---------------------------------------------------------------- homes
def mostrar_grau_e_classe(modelo):
    """Grau × classe, nas palavras de cada modelo."""
    with st.container(border=True):
        if modelo == "fuzzyknn":
            # Por que a classe sai da maior pertinência, e não de um Alfa-Corte escolhido à mão.
            st.markdown(
                "O sistema calcula a **pertinência**: o quanto a imagem pertence ao conjunto “cachorro”, "
                "de 0 a 1. A pertinência a “não cachorro” é o que falta para 1.\n\n"
                "A **classe** é uma decisão. Fica a classe de maior pertinência: se a pertinência a "
                "cachorro passa de 0,50, a resposta é “É cachorro”; se fica em 0,50 ou menos, "
                "“Não é cachorro”.\n\n"
                "Para a classe, escolhi seguir a regra do próprio Fuzzy k-NN, a de maior pertinência, "
                "em vez de um Alfa-Corte (parâmetro numérico) escolhido à mão, como 0,60. Assim, a "
                "classe é sempre aquela a que a imagem mais pertence, e a dúvida aparece na área "
                "cinzenta, sem mudar a classe.\n\n"
                "Por isso uma imagem pode ser “É cachorro” e estar na área cinzenta ao mesmo tempo "
                "(nuances da dúvida): ela pertence mais ao conjunto cachorro do que ao outro, mas "
                "por pouco.")
        else:
            st.markdown(
                "O SVM calcula a **probabilidade**: a chance de a imagem ser cachorro, de 0% a 100%. "
                "A probabilidade de “não cachorro” é o que falta para 100%.\n\n"
                "A **classe** é uma decisão. Fica a classe de maior probabilidade: se a probabilidade "
                "de cachorro passa de 50%, a resposta é “É cachorro”; se fica em 50% ou menos, "
                "“Não é cachorro”.\n\n"
                "A dúvida aparece na área cinzenta, sem mudar a classe. Por isso uma imagem pode ser "
                "“É cachorro” e estar na área cinzenta ao mesmo tempo: a probabilidade de cachorro é a "
                "maior, mas por pouco.\n\n"
                "Probabilidade não é pertinência: a probabilidade é a chance de um evento (a imagem ser "
                "cachorro); a pertinência, na home do Fuzzy k-NN, é o quanto a imagem pertence ao "
                "conjunto “cachorro”.")


def home(modelo):
    st.title(f"Cachorro × não cachorro · {NOMES[modelo]}")
    if modelo == "fuzzyknn":
        st.caption(f"O EfficientNetB0 descreve a imagem em 1.280 números, e o Fuzzy k-NN "
                   f"(k = {config.FUZZYKNN_K}) compara esses números com o bairro. "
                   "A classe é a de maior pertinência.")
    else:
        st.caption("O EfficientNetB0 descreve a imagem em 1.280 números, e o SVM, treinado no bairro, "
                   "calcula a probabilidade de ser cachorro. A classe é a de maior probabilidade.")
    mostrar_grau_e_classe(modelo)
    if not config.ARQ_BAIRRO.exists():
        st.error("O bairro ainda não foi montado. Rode no terminal: python montar_bairro.py")
        return
    marca = config.ARQ_BAIRRO.stat().st_mtime
    bairro = carregar_bairro(marca)
    try:
        aviso = rodadas.aviso_da_home(modelo, bairro)      # proteção 7
    except rodadas.Trava as erro:
        aviso = str(erro)
    if aviso:
        st.warning(aviso)
    if modelo == "svm":
        carregar_svm(marca)                                # treina já, com o aviso de espera

    for nome, inicial in (("historico", []), ("selecionada", None), ("ultimo_envio", None)):
        if chave(modelo, nome) not in st.session_state:          # imagens desta home, só nesta sessão
            st.session_state[chave(modelo, nome)] = inicial
    historico = st.session_state[chave(modelo, "historico")]

    enviado = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png", "webp"],
                               key=chave(modelo, "envio"))
    if enviado is None:
        st.session_state[chave(modelo, "ultimo_envio")] = None
    else:
        conteudo = enviado.getvalue()
        ident = hashlib.sha1(conteudo).hexdigest()
        if (enviado.name, ident) != st.session_state[chave(modelo, "ultimo_envio")]:   # envio novo
            st.session_state[chave(modelo, "ultimo_envio")] = (enviado.name, ident)
            if ident not in {item["id"] for item in historico}:
                try:
                    with st.spinner("Analisando a imagem..."):
                        resultado = analisar(modelo, conteudo, bairro, marca)
                    historico.append({"id": ident, "nome": enviado.name,
                                      "conteudo": conteudo, "resultado": resultado})
                except Exception as erro:   # arquivo corrompido ou formato que o Pillow não abre
                    st.error(f"Não consegui analisar essa imagem: {erro}")
            if ident in {item["id"] for item in historico}:
                st.session_state[chave(modelo, "selecionada")] = ident

    selecionada = st.session_state[chave(modelo, "selecionada")]
    selecionado = next((item for item in historico if item["id"] == selecionada), None)
    if selecionado is not None:
        mostrar_resultado(modelo, selecionado)
    mostrar_imagens_anteriores(modelo)


# ---------------------------------------------------------------- visão admin: aba de cada modelo
def linha_da_tabela(modelo, rodada):
    linha = {"Rodada": rodada["rodada"], "O que mudou": rodada["o_que_mudou"], "Extrator": rodada["extrator"]}
    for campo in rodadas.CAMPOS_PROPRIOS_RODADAS[modelo]:     # k no Fuzzy k-NN; C e gamma no SVM
        linha[campo] = rodada[campo]
    linha.update({
        "Bairro": rodada["bairro"],
        "Assinatura do bairro": rodada["assinatura_bairro"],
        "VP": rodada["VP"],
        "FN": rodada["FN"],
        "FP": rodada["FP"],
        "VN": rodada["VN"],
        "Acurácia": percentual(float(rodada["acuracia"])),
        "Área cinzenta": rodada["area_cinzenta"],
        "Meta": "✓" if rodada["meta"] == "sim" else "✗",
    })
    return linha


def texto_legenda(situacao, com_marca=False):
    """A sigla da matriz por extenso: "FP · falso positivo — rótulo real não cachorro e classe cachorro"."""
    nome, explicacao, acertou = avaliacao.LEGENDA_CASAS[situacao]
    marca = f" {'✓' if acertou else '✗'}" if com_marca else ""
    return f"**{situacao} · {nome}** — {explicacao}{marca}"


def mostrar_matriz(rodada):
    total = sum(int(rodada[situacao]) for situacao in avaliacao.CASAS)
    st.markdown("**Matriz de confusão**")
    st.caption(f"Conta as {total} imagens do teste fixo, inclusive as da área cinzenta. "
               "A galeria mais abaixo mostra só as que caíram na faixa de dúvida, "
               "então os números dela são menores que os daqui.")
    st.markdown(
        "|  | disse cachorro | disse não |\n"
        "|---|---|---|\n"
        f"| **era cachorro** | VP {rodada['VP']} | FN {rodada['FN']} |\n"
        f"| **não era** | FP {rodada['FP']} | VN {rodada['VN']} |")
    # Na ordem de leitura da matriz, não na do PRD.
    st.caption("  \n".join(texto_legenda(situacao, com_marca=True)
                           for situacao in ("VP", "FN", "FP", "VN")))


def mostrar_acuracia_e_meta(rodada):
    vp, fn, fp, vn = (int(rodada[casa]) for casa in ("VP", "FN", "FP", "VN"))
    acuracia = float(rodada["acuracia"])
    st.metric("Acurácia", percentual(acuracia), help="(VP + VN) ÷ total")
    st.caption(f"({vp} + {vn}) ÷ {vp + fn + fp + vn}")
    atingida, criterios = avaliacao.meta_atingida(acuracia, fn, fp)
    st.markdown("**Meta:** " + " · ".join(f"{texto} {'✓' if ok else '✗'}" for ok, texto in criterios))
    if atingida:
        st.success("Meta atingida")
    else:
        st.warning("Meta não atingida")


def legenda_teste(modelo, item):
    """O grau sai da coluna de duas casas, a mesma que decidiu a classe e a faixa."""
    categoria = dados.nome_categoria(item["dataset"], item["categoria_original"])
    valor = float(item.get("p_cachorro_tela") or item["p_cachorro"])
    if modelo == "fuzzyknn":
        return f"{categoria} · pertinência a cachorro {formatar_grau(valor)}"
    return f"{categoria} · probabilidade de cachorro {svm.formatar_porcentagem(valor)}"


def mostrar_galeria(modelo, resultados):
    galeria = avaliacao.galeria_area_cinzenta(resultados)
    na_faixa = sum(len(lista) for lista in galeria.values())
    minimo, maximo = LIMITES[modelo]
    st.subheader(f"Galeria da área cinzenta · {na_faixa} de {len(resultados)} imagens")
    if modelo == "fuzzyknn":
        st.caption(f"Imagens do teste fixo com pertinência a cachorro de {minimo} a "
                   f"{maximo}. Os erros daqui também contam na matriz; as vizinhas mostram "
                   "por que o modelo ficou em dúvida.")
        st.caption("Em cada bloco abaixo, três coisas diferentes: **rótulo real** é o gabarito da base, "
                   "o que a imagem é de fato; **pertinência** é o número que o Fuzzy k-NN calculou, de 0 a 1; "
                   "**classe** é a decisão tirada desse número — acima de 0,50, cachorro. "
                   "A sigla (VP, FN, VN, FP) compara só o rótulo real com a classe.")
        formatar, nome_do_grau = formatar_grau, "pertinência a cachorro"
        coluna_fotos, coluna_classes = "vizinhas", "classes_vizinhas"
    else:
        st.caption(f"Imagens do teste fixo com probabilidade de cachorro de {minimo} a "
                   f"{maximo}. Os erros daqui também contam na matriz; as imagens que mais pesaram "
                   "mostram por que o modelo ficou em dúvida.")
        st.caption("Em cada bloco abaixo, três coisas diferentes: **rótulo real** é o gabarito da base, "
                   "o que a imagem é de fato; **probabilidade** é o número que o SVM calculou, de 0% a 100%; "
                   "**classe** é a decisão tirada desse número — acima de 50%, cachorro. "
                   "A sigla (VP, FN, VN, FP) compara só o rótulo real com a classe.")
        formatar, nome_do_grau = svm.formatar_porcentagem, "probabilidade de cachorro"
        coluna_fotos, coluna_classes = "mais_pesaram", "classes_mais_pesaram"
    situacoes = avaliacao.situacoes(formatar)
    for situacao in avaliacao.CASAS:
        faixa, resultado = situacoes[situacao][1], situacoes[situacao][3]
        lista = galeria[situacao]
        st.markdown(f"#### {resultado} · {len(lista)}")
        st.caption(f"{texto_legenda(situacao)}  \n"
                   f"{avaliacao.LEITURA_AREA_CINZENTA[situacao]}  \n"
                   f"{nome_do_grau} nesta galeria: {faixa}")
        if not lista:
            st.write("Nenhuma imagem nesta situação.")
            continue
        for item in lista:
            with st.container(border=True):
                col_teste, col_fotos = st.columns([1, 4])
                with col_teste:
                    mostrar_imagem(dados.localizar_imagem(item["arquivo"]), legenda_teste(modelo, item), 170)
                with col_fotos:
                    rotulos = [dados.rotulo(classe) for classe in item[coluna_classes].split(";")]
                    mostrar_fotos(list(zip(item[coluna_fotos].split(";"), rotulos)), largura=110)


COMANDO_DA_RODADA_1 = 'python avaliar.py --rodada 1 --mudanca "ponto de partida: bairro com 320 (160 + 160)"'


def aba_do_modelo(modelo):
    try:
        lista = rodadas.ler_rodadas(modelo)
    except rodadas.Trava as erro:
        st.error(str(erro))
        return
    if not lista:
        st.info(f"Nenhuma rodada do {NOMES[modelo]} registrada ainda. Rode no terminal:\n\n"
                f"{COMANDO_DA_RODADA_1}")
        return

    st.subheader("Rodadas")
    st.dataframe([linha_da_tabela(modelo, rodada) for rodada in lista], hide_index=True)

    por_numero = {int(rodada["rodada"]): rodada for rodada in lista}
    numeros = list(por_numero)
    numero = st.selectbox("Rodada", numeros, index=len(numeros) - 1, key=chave(modelo, "rodada"),
                          format_func=lambda n: f"Rodada {n} · {por_numero[n]['o_que_mudou']}")
    rodada = por_numero[numero]

    col_matriz, col_metricas = st.columns(2)
    with col_matriz:
        mostrar_matriz(rodada)
    with col_metricas:
        mostrar_acuracia_e_meta(rodada)
    try:
        resultados = rodadas.ler_resultados(modelo, numero)
    except rodadas.Trava as erro:
        st.error(str(erro))
        return
    mostrar_galeria(modelo, resultados)


# ---------------------------------------------------------------- visão admin: comparativo
def acertos(linha):
    return int(linha["VP"]) + int(linha["VN"])


def texto_diferenca(diferenca):
    """A diferença em número de imagens, do SVM em relação ao Fuzzy k-NN."""
    if diferenca is None:
        return "—"
    if diferenca == 0:
        return "igual"
    a_mais = NOMES["svm"] if diferenca > 0 else NOMES["fuzzyknn"]
    return f"{plural(abs(diferenca), 'imagem', 'imagens')} a mais no {a_mais}"


def mostrar_lado_a_lado(linha_fuzzyknn, linha_svm, total):
    medidas = [("Acurácia",
                f"{percentual(float(linha_fuzzyknn['acuracia']))} ({acertos(linha_fuzzyknn)} de {total})",
                f"{percentual(float(linha_svm['acuracia']))} ({acertos(linha_svm)} de {total})",
                acertos(linha_svm) - acertos(linha_fuzzyknn))]
    for casa in ("VP", "FN", "FP", "VN"):
        medidas.append((casa, linha_fuzzyknn[casa], linha_svm[casa], int(linha_svm[casa]) - int(linha_fuzzyknn[casa])))
    medidas.append(("Meta", "✓" if linha_fuzzyknn["meta"] == "sim" else "✗",
                    "✓" if linha_svm["meta"] == "sim" else "✗", None))
    medidas.append(("Área cinzenta", linha_fuzzyknn["area_cinzenta"], linha_svm["area_cinzenta"],
                    int(linha_svm["area_cinzenta"]) - int(linha_fuzzyknn["area_cinzenta"])))
    st.markdown("**Lado a lado**")
    st.dataframe([{"Medida": nome, NOMES["fuzzyknn"]: str(valor_f), NOMES["svm"]: str(valor_s),
                   "Diferença": texto_diferenca(diferenca)}
                  for nome, valor_f, valor_s, diferenca in medidas], hide_index=True)


def mostrar_quem_foi_melhor(linha_fuzzyknn, linha_svm, total):
    cinzenta = {"fuzzyknn": int(linha_fuzzyknn["area_cinzenta"]), "svm": int(linha_svm["area_cinzenta"])}
    veredito = avaliacao.quem_foi_melhor(acertos(linha_fuzzyknn), acertos(linha_svm),
                                         cinzenta["fuzzyknn"], cinzenta["svm"])
    if veredito["vencedor"] is not None:
        vencedor = MODELOS[veredito["vencedor"] - 1]
        st.success(f"Quem foi melhor: o {NOMES[vencedor]}, que acertou "
                   f"{plural(veredito['diferenca'], 'imagem', 'imagens')} a mais nas mesmas {total}.")
        return
    st.info(f"Empate: os dois acertaram {acertos(linha_fuzzyknn)} de {total}.")
    if veredito["menos_area_cinzenta"] is not None:
        menos = MODELOS[veredito["menos_area_cinzenta"] - 1]
        outro = MODELOS[1 - MODELOS.index(menos)]
        nota = (f"Nota: o {NOMES[menos]} teve menos imagens na área cinzenta ({cinzenta[menos]}, contra "
                f"{cinzenta[outro]} do {NOMES[outro]}).")
    else:
        nota = f"Nota: a área cinzenta também empatou ({cinzenta['fuzzyknn']} imagens em cada modelo)."
    st.caption(f"{nota} A nota não muda o resultado: a área cinzenta mostra dúvida, não acerto, e a "
               "faixa lê pertinência no Fuzzy k-NN e probabilidade no SVM.")


def resposta_do_modelo(modelo, linha):
    """"É cachorro · pertinência 0,62 · acertou (VP)", com o grau de duas casas da rodada."""
    classe, grau = dados.rotulo(linha["resposta"]), float(linha["p_cachorro_tela"])
    texto = fuzzyknn.mensagem(classe, grau) if modelo == "fuzzyknn" else svm.mensagem(classe, grau)
    return f"{texto} · {'acertou' if avaliacao.acertou(linha) else 'errou'} ({linha['casa']})"


def mostrar_imagem_a_imagem(resultados_fuzzyknn, resultados_svm):
    grupos = avaliacao.comparar_imagens(resultados_fuzzyknn, resultados_svm)
    st.markdown("**Imagem a imagem**")
    st.markdown(f"os dois acertaram: {len(grupos['os_dois_acertaram'])} · "
                f"só o Fuzzy k-NN acertou: {len(grupos['so_o_primeiro'])} · "
                f"só o SVM acertou: {len(grupos['so_o_segundo'])} · "
                f"os dois erraram: {len(grupos['os_dois_erraram'])}")
    discordam = len(grupos["so_o_primeiro"]) + len(grupos["so_o_segundo"])
    st.subheader(f"Galeria das que discordam · {plural(discordam, 'imagem', 'imagens')}")
    st.caption("Imagens em que só um dos modelos acertou, com o rótulo real e a resposta de cada "
               "modelo, no grau dele: pertinência no Fuzzy k-NN, probabilidade no SVM.")
    for titulo, grupo in (("Só o Fuzzy k-NN acertou", "so_o_primeiro"), ("Só o SVM acertou", "so_o_segundo")):
        st.markdown(f"#### {titulo} · {len(grupos[grupo])}")
        if not grupos[grupo]:
            st.write("Nenhuma imagem nesta situação.")
            continue
        for linha_fuzzyknn, linha_svm in grupos[grupo]:
            with st.container(border=True):
                col_imagem, col_texto = st.columns([1, 4])
                with col_imagem:
                    categoria = dados.nome_categoria(linha_fuzzyknn["dataset"], linha_fuzzyknn["categoria_original"])
                    mostrar_imagem(dados.localizar_imagem(linha_fuzzyknn["arquivo"]), categoria, 170)
                with col_texto:
                    st.markdown(f"rótulo real: {dados.texto_classe(linha_fuzzyknn['classe_real'])}  \n"
                                f"**{NOMES['fuzzyknn']}:** {resposta_do_modelo('fuzzyknn', linha_fuzzyknn)}  \n"
                                f"**{NOMES['svm']}:** {resposta_do_modelo('svm', linha_svm)}")


def aba_comparativo():
    try:
        numeros = rodadas.rodadas_nas_duas_tabelas()      # proteção 13: só as que existem nas duas
        o_que_mudou = {int(linha["rodada"]): linha["o_que_mudou"] for linha in rodadas.ler_rodadas("fuzzyknn")}
    except rodadas.Trava as erro:
        st.error(str(erro))
        return
    if not numeros:
        st.info("Nenhuma rodada existe ainda nas duas tabelas. O comparativo mostra a mesma rodada dos "
                f"dois modelos, depois que ela for registrada:\n\n{COMANDO_DA_RODADA_1}")
        return
    numero = st.selectbox("Rodada", numeros, index=len(numeros) - 1, key=chave("comparativo", "rodada"),
                          format_func=lambda n: f"Rodada {n} · {o_que_mudou.get(n, '')}")
    try:
        linhas, resultados = rodadas.carregar_comparacao(numero)
    except rodadas.Trava as erro:
        st.error(str(erro))
        return
    linha_fuzzyknn, linha_svm = linhas["fuzzyknn"], linhas["svm"]
    total = len(resultados["fuzzyknn"])
    st.caption(f"A rodada {numero} dos dois modelos, nas mesmas {total} imagens do teste fixo e com o mesmo "
               f"bairro (assinatura {linha_fuzzyknn['assinatura_bairro']}). Ganha quem acerta mais imagens.")
    mostrar_lado_a_lado(linha_fuzzyknn, linha_svm, total)
    mostrar_quem_foi_melhor(linha_fuzzyknn, linha_svm, total)
    mostrar_imagem_a_imagem(resultados["fuzzyknn"], resultados["svm"])


def visao_admin():
    st.title("Visão admin · resultados no teste fixo")
    aba_fuzzyknn, aba_svm, aba_comparacao = st.tabs([NOMES["fuzzyknn"], NOMES["svm"], "Comparativo"])
    with aba_fuzzyknn:
        aba_do_modelo("fuzzyknn")
    with aba_svm:
        aba_do_modelo("svm")
    with aba_comparacao:
        aba_comparativo()


# ---------------------------------------------------------------- navegação
VISOES = {"Home · Fuzzy k-NN": "fuzzyknn", "Home · SVM": "svm", "Visão admin": None}


def legenda_da_barra(modelo):
    minimo, maximo = LIMITES[modelo]
    if modelo == "fuzzyknn":
        return f"k = {config.FUZZYKNN_K} · área cinzenta: pertinência a cachorro de {minimo} a {maximo}"
    return (f"C = {config.SVM_C:g} · gamma = {config.SVM_GAMMA} · área cinzenta: probabilidade de "
            f"cachorro de {minimo} a {maximo}")


def main():
    st.set_page_config(page_title="Cachorro × não cachorro", page_icon="🐶", layout="wide")
    visao = st.sidebar.radio("Visão", list(VISOES))
    modelo = VISOES[visao]
    if modelo is not None:
        st.sidebar.caption(legenda_da_barra(modelo))
        home(modelo)
    else:
        st.sidebar.caption("  \n".join(f"{NOMES[m]}: {legenda_da_barra(m)}" for m in MODELOS))
        visao_admin()


if __name__ == "__main__":
    main()
