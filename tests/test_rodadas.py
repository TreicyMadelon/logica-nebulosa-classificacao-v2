"""Testes das rodadas e das travas contra salvar errado, um por proteção, com números
inventados e pastas temporárias (nada é gravado nas pastas de resultados do projeto).
"""
import copy

import numpy as np
import pytest

import avaliacao
import config
import dados
import fuzzyknn
import rodadas
import svm
from rodadas import Trava


# ---------------------------------------------------------------- dados inventados
def inventar(semente=0):
    """Um bairro de 60 (30 + 30) e um teste de 20, com números inventados em 6 dimensões."""
    rng = np.random.default_rng(semente)
    n = 30
    X = np.empty((2 * n, 6))
    X[0::2] = rng.normal(0.6, 1.0, (n, 6))
    X[1::2] = rng.normal(-0.6, 1.0, (n, 6))
    y = np.array([1, 0] * n)
    bairro = {"X": X.astype(np.float32), "y": y, "extrator": "efficientnetb0",
              "arquivos": [f"b{i:03d}.jpg" for i in range(2 * n)]}
    y_teste = np.array([1, 0] * 10)
    X_teste = rng.normal(np.where(y_teste == 1, 0.3, -0.3)[:, None], 1.0, (20, 6)).astype(np.float32)
    teste = {"X": X_teste, "y": y_teste, "arquivos": [f"t{i:02d}.jpg" for i in range(20)],
             "datasets": ["voc"] * 20, "categorias": ["dog" if r == 1 else "cat" for r in y_teste]}
    return bairro, teste


def avaliar(bairro, teste):
    return {"fuzzyknn": fuzzyknn.avaliar_teste_fixo(bairro, teste),
            "svm": svm.avaliar_teste_fixo(bairro, teste)}


def trocar_uma_resposta(medidas):
    """As mesmas medidas, com a resposta de uma imagem trocada (para mudar os números)."""
    linhas = copy.deepcopy(medidas["resultados"])
    linha = linhas[0]
    disse_cachorro = linha["resposta"] == "cachorro"
    linha["resposta"] = "nao_cachorro" if disse_cachorro else "cachorro"
    linha["casa"] = avaliacao.casa(dados.rotulo(linha["classe_real"]), 0 if disse_cachorro else 1)
    return avaliacao.medir(linhas)


@pytest.fixture
def pastas(tmp_path, monkeypatch):
    """Pastas de resultados e imagens da base em tmp_path."""
    monkeypatch.setattr(config, "FUZZYKNN_PASTA_RESULTADOS", tmp_path / "resultados_fuzzyknn")
    monkeypatch.setattr(config, "SVM_PASTA_RESULTADOS", tmp_path / "resultados_svm")
    base = tmp_path / "base_imagens"
    monkeypatch.setattr(config, "PASTAS_CLASSE", {"cachorro": base / "cachorro", "nao_cachorro": base / "nao_cachorro"})
    monkeypatch.setattr(config, "PASTA_DESCARTADAS", tmp_path / "triagem" / "descartadas")
    _, teste = inventar()
    for arquivo, rotulo in zip(teste["arquivos"], teste["y"]):
        pasta = config.PASTAS_CLASSE[dados.classe_do_rotulo(rotulo)]
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / arquivo).write_bytes(b"imagem inventada")
    return tmp_path


def arquivos_gravados(raiz):
    """Todo arquivo dentro das duas pastas de resultados, com o conteúdo."""
    gravados = {}
    for pasta in ("resultados_fuzzyknn", "resultados_svm"):
        for caminho in sorted((raiz / pasta).rglob("*")):
            if caminho.is_file():
                gravados[caminho.relative_to(raiz).as_posix()] = caminho.read_bytes()
    return gravados


# ---------------------------------------------------------------- 1. caminhos de uma função só
def test_1_caminhos_de_uma_funcao_so(pastas, monkeypatch):
    for modelo in rodadas.MODELOS:
        c = rodadas.caminhos(modelo)
        assert c.pasta == pastas / f"resultados_{modelo}"
        assert c.tabela == c.pasta / f"rodadas_{modelo}.csv"
        assert c.resultados(1) == c.pasta / f"rodada_1_{modelo}" / f"resultados_{modelo}.csv"
        assert c.copia_imagem(1, "FP", "0.55", "imagenet_ILSVRC2012_val_00000001.JPEG") == (
            c.pasta / f"rodada_1_{modelo}" / f"FP_{modelo}" / f"p0.55_imagenet_ILSVRC2012_val_00000001_{modelo}.JPEG")
    assert not hasattr(avaliacao, "ARQ_RODADAS")              # nenhum caminho fixado na hora do import
    with pytest.raises(Trava):
        rodadas.caminhos("knn")                               # modelo desconhecido
    monkeypatch.setattr(config, "SVM_PASTA_RESULTADOS", pastas / "resultados")
    with pytest.raises(Trava):
        rodadas.caminhos("svm")                               # a pasta precisa levar o nome do modelo
    monkeypatch.setattr(config, "SVM_PASTA_RESULTADOS", pastas / "resultados_fuzzyknn" / "resultados_svm")
    with pytest.raises(Trava):
        rodadas.caminhos("svm")                               # uma pasta dentro da outra


# ---------------------------------------------------------------- 2. trava de pasta
def test_2_trava_de_pasta(pastas):
    fuzzy, svm_ = rodadas.caminhos("fuzzyknn"), rodadas.caminhos("svm")
    assert rodadas.conferir_caminho("fuzzyknn", fuzzy.tabela)
    assert rodadas.conferir_caminho("svm", svm_.copia_imagem(2, "FN", "0.45", "voc_2008_000001.jpg"))
    proibidos = [
        ("svm", fuzzy.tabela),                                             # na pasta do outro modelo
        ("fuzzyknn", pastas / "resultados" / "rodadas.csv"),               # fora das pastas da V2
        ("fuzzyknn", fuzzy.pasta / "rodada_1_svm" / "resultados_svm.csv"),  # com o nome do outro
        ("fuzzyknn", fuzzy.pasta / "rodada_1_fuzzyknn" / "VP_fuzzyknn" / "p0.55_x_fuzzyknn.jpg"),
        ("svm", svm_.pasta / ".." / "resultados_fuzzyknn" / "rodadas_fuzzyknn.csv"),
    ]
    for modelo, caminho in proibidos:
        with pytest.raises(Trava):
            rodadas.conferir_caminho(modelo, caminho)
    with pytest.raises(Trava):
        rodadas._gravar_csv("svm", fuzzy.tabela, [], rodadas.CAMPOS_RODADAS["svm"])
    assert not fuzzy.tabela.exists()                                       # nada foi gravado


# ---------------------------------------------------------------- 3. trava de tabela
def test_3_trava_de_tabela(pastas):
    bairro, teste = inventar()
    medidas = avaliar(bairro, teste)
    # Os resultados do Fuzzy k-NN no lugar dos do SVM: recusado, nada gravado.
    with pytest.raises(Trava):
        rodadas.registrar_rodada(1, "ponto de partida", {"fuzzyknn": medidas["fuzzyknn"],
                                                         "svm": medidas["fuzzyknn"]}, bairro)
    assert arquivos_gravados(pastas) == {}
    # Uma linha do Fuzzy k-NN (com k, sem C e gamma) não entra na tabela do SVM.
    linha = rodadas.linha_da_rodada("fuzzyknn", 1, "x", medidas["fuzzyknn"], bairro)
    with pytest.raises(Trava):
        rodadas._conferir_campos(linha, rodadas.CAMPOS_RODADAS["svm"], "rodadas do SVM")
    # Uma tabela do Fuzzy k-NN copiada para a pasta do SVM: a leitura e a gravação recusam.
    dados.gravar_csv(rodadas.caminhos("svm").tabela, [linha], rodadas.CAMPOS_RODADAS["fuzzyknn"])
    with pytest.raises(Trava):
        rodadas.ler_rodadas("svm")
    antes = arquivos_gravados(pastas)
    with pytest.raises(Trava):
        rodadas.registrar_rodada(1, "ponto de partida", medidas, bairro)
    assert arquivos_gravados(pastas) == antes


# ---------------------------------------------------------------- 4. trava de rodada
def test_4_trava_de_rodada(pastas):
    bairro, teste = inventar()
    medidas = avaliar(bairro, teste)
    rodadas.registrar_rodada(1, "ponto de partida", medidas, bairro)
    antes = arquivos_gravados(pastas)
    with pytest.raises(Trava):
        rodadas.registrar_rodada(1, "de novo, sem pedir", medidas, bairro)          # sem pedido explícito
    assert arquivos_gravados(pastas) == antes
    with pytest.raises(Trava):
        rodadas.registrar_rodada(2, "nada para substituir", medidas, bairro, substituir=True)
    rodadas.registrar_rodada(1, "de novo, com pedido", medidas, bairro, substituir=True)
    for modelo in rodadas.MODELOS:                                                   # nas duas tabelas
        tabela = rodadas.ler_rodadas(modelo)
        assert [(l["rodada"], l["o_que_mudou"]) for l in tabela] == [("1", "de novo, com pedido")]
    rodadas.registrar_rodada(3, "pulando a 2", medidas, bairro)
    with pytest.raises(Trava):
        rodadas.registrar_rodada(2, "antes da última", medidas, bairro)              # viria antes da 3
    # Uma pasta de rodada sem linha na tabela (gravação interrompida) também pede --substituir.
    rodadas.caminhos("svm").pasta_rodada(4).mkdir()
    with pytest.raises(Trava):
        rodadas.registrar_rodada(4, "pasta já existe", medidas, bairro)


# ---------------------------------------------------------------- 5. grava no fim
def test_5_grava_no_fim_e_as_duas_tabelas_tem_as_mesmas_rodadas(pastas):
    bairro, teste = inventar()
    medidas = avaliar(bairro, teste)
    rodadas.registrar_rodada(1, "ponto de partida", medidas, bairro)
    assert [l["rodada"] for l in rodadas.ler_rodadas("fuzzyknn")] == [l["rodada"] for l in rodadas.ler_rodadas("svm")]
    # A tabela do SVM some: as duas deixam de ter as mesmas rodadas, e nada é gravado.
    rodadas.caminhos("svm").tabela.unlink()
    antes = arquivos_gravados(pastas)
    with pytest.raises(Trava):
        rodadas.registrar_rodada(2, "segunda", medidas, bairro)
    assert arquivos_gravados(pastas) == antes
    assert not rodadas.caminhos("fuzzyknn").pasta_rodada(2).exists()


def test_5_uma_trava_no_fim_nao_deixa_nada_gravado(pastas):
    bairro, teste = inventar()
    medidas = avaliar(bairro, teste)
    rodadas.registrar_rodada(1, "ponto de partida", medidas, bairro)
    antes = arquivos_gravados(pastas)
    mudado = {"fuzzyknn": medidas["fuzzyknn"], "svm": trocar_uma_resposta(medidas["svm"])}
    with pytest.raises(Trava):                     # a regra de repetir os números barra no fim
        rodadas.registrar_rodada(2, "nada mudou", mudado, bairro)
    assert arquivos_gravados(pastas) == antes


# ---------------------------------------------------------------- 6. as mesmas 80 nos dois
def test_6_as_mesmas_imagens_nos_dois(pastas):
    bairro, teste = inventar()
    medidas = avaliar(bairro, teste)
    com_uma_a_menos = {"fuzzyknn": medidas["fuzzyknn"],
                       "svm": avaliacao.medir(medidas["svm"]["resultados"][1:])}
    fora_de_ordem = {"fuzzyknn": medidas["fuzzyknn"],
                     "svm": avaliacao.medir(list(reversed(medidas["svm"]["resultados"])))}
    for medidas_erradas in (com_uma_a_menos, fora_de_ordem):
        with pytest.raises(Trava):
            rodadas.registrar_rodada(1, "ponto de partida", medidas_erradas, bairro)
    assert arquivos_gravados(pastas) == {}


# ---------------------------------------------------------------- 7. aviso da home
def test_7_a_home_avisa_se_o_modelo_em_uso_nao_foi_avaliado(pastas, monkeypatch):
    bairro, teste = inventar()
    for modelo in rodadas.MODELOS:
        assert "ainda não foi avaliado" in rodadas.aviso_da_home(modelo, bairro)
    rodadas.registrar_rodada(1, "ponto de partida", avaliar(bairro, teste), bairro)
    for modelo in rodadas.MODELOS:
        assert rodadas.aviso_da_home(modelo, bairro) is None
    rotulo_corrigido = {**bairro, "y": bairro["y"].copy()}
    rotulo_corrigido["y"][0] = 0                                  # mesmo total de imagens
    for modelo in rodadas.MODELOS:
        assert "não é o da última rodada" in rodadas.aviso_da_home(modelo, rotulo_corrigido)
    monkeypatch.setattr(config, "FUZZYKNN_K", 7)
    assert "parâmetros" in rodadas.aviso_da_home("fuzzyknn", bairro)
    assert rodadas.aviso_da_home("svm", bairro) is None


# ---------------------------------------------------------------- 12. ordem do bairro
def test_12_a_assinatura_acusa_outro_bairro_ou_outra_ordem():
    bairro, _ = inventar()
    assinatura = rodadas.assinatura_bairro(bairro)
    assert len(assinatura) == 12
    assert rodadas.assinatura_bairro(copy.deepcopy(bairro)) == assinatura          # o mesmo bairro
    ordem = np.arange(len(bairro["y"]))[::-1]
    invertido = {**bairro, "X": bairro["X"][ordem], "y": bairro["y"][ordem],
                 "arquivos": [bairro["arquivos"][i] for i in ordem]}
    rotulo = {**bairro, "y": np.where(np.arange(len(bairro["y"])) == 0, 1 - bairro["y"], bairro["y"])}
    trocada = {**bairro, "X": bairro["X"].copy(), "arquivos": ["outra.jpg", *bairro["arquivos"][1:]]}
    trocada["X"][0] += 0.5                                                          # imagem trocada, mesmo total
    extrator = {**bairro, "extrator": "mobilenetv2"}
    for outro in (invertido, rotulo, trocada, extrator):
        assert rodadas.assinatura_bairro(outro) != assinatura


# ---------------------------------------------------------------- 13. comparativo
def test_13_o_comparativo_so_compara_a_mesma_rodada(pastas):
    bairro, teste = inventar()
    assert rodadas.rodadas_nas_duas_tabelas() == []
    rodadas.registrar_rodada(1, "ponto de partida", avaliar(bairro, teste), bairro)
    assert rodadas.rodadas_nas_duas_tabelas() == [1]
    linhas, resultados = rodadas.carregar_comparacao(1)
    assert linhas["fuzzyknn"]["rodada"] == linhas["svm"]["rodada"] == "1"
    assert [r["arquivo"] for r in resultados["fuzzyknn"]] == [r["arquivo"] for r in resultados["svm"]]
    with pytest.raises(Trava):
        rodadas.carregar_comparacao(2)                         # não existe nas duas
    # Resultados do SVM com uma imagem a menos: a comparação não vale.
    arquivo = rodadas.caminhos("svm").resultados(1)
    linhas_svm = dados.ler_csv(arquivo)
    dados.gravar_csv(arquivo, linhas_svm[1:], rodadas.CAMPOS_RESULTADOS["svm"])
    with pytest.raises(Trava):
        rodadas.carregar_comparacao(1)
    dados.gravar_csv(arquivo, linhas_svm, rodadas.CAMPOS_RESULTADOS["svm"])
    # Assinaturas diferentes nas duas linhas da rodada: a comparação não vale.
    tabela = rodadas.caminhos("svm").tabela
    linha = dados.ler_csv(tabela)
    linha[0]["assinatura_bairro"] = "000000000000"
    dados.gravar_csv(tabela, linha, rodadas.CAMPOS_RODADAS["svm"])
    with pytest.raises(Trava):
        rodadas.carregar_comparacao(1)


# ---------------------------------------------------------------- regra: o modelo que não mudou repete
def test_o_modelo_que_nao_mudou_repete_os_numeros(pastas):
    bairro, teste = inventar()
    medidas = avaliar(bairro, teste)
    rodadas.registrar_rodada(1, "ponto de partida", medidas, bairro)
    rodadas.registrar_rodada(2, "conferência: nada mudou", medidas, bairro)       # os dois repetem
    so_o_svm_mudou = {"fuzzyknn": medidas["fuzzyknn"], "svm": trocar_uma_resposta(medidas["svm"])}
    with pytest.raises(Trava) as erro:
        rodadas.registrar_rodada(3, "código do SVM", so_o_svm_mudou, bairro)
    assert "SVM" in str(erro.value) and "--mudou svm" in str(erro.value)
    rodadas.registrar_rodada(3, "código do SVM", so_o_svm_mudou, bairro, mudou=("svm",))
    # O Fuzzy k-NN não pode mudar junto sem ser declarado.
    os_dois = {"fuzzyknn": trocar_uma_resposta(medidas["fuzzyknn"]), "svm": medidas["svm"]}
    with pytest.raises(Trava):
        rodadas.registrar_rodada(4, "código do SVM de novo", os_dois, bairro, mudou=("svm",))
    # Com outro bairro (rodada de dados), os dois mudaram: nenhum precisa repetir.
    outro_bairro, _ = inventar(semente=5)
    rodadas.registrar_rodada(4, "outro bairro", avaliar(outro_bairro, teste), outro_bairro)
    assert [l["rodada"] for l in rodadas.ler_rodadas("svm")] == ["1", "2", "3", "4"]


# ---------------------------------------------------------------- o que a rodada grava
def test_a_rodada_grava_tabela_resultados_e_copias_com_o_nome_do_modelo(pastas):
    bairro, teste = inventar()
    medidas = avaliar(bairro, teste)
    rodadas.registrar_rodada(1, "ponto de partida", medidas, bairro)
    for modelo in rodadas.MODELOS:
        c = rodadas.caminhos(modelo)
        linha = rodadas.ler_rodadas(modelo)[0]
        assert linha["assinatura_bairro"] == rodadas.assinatura_bairro(bairro)
        assert linha["bairro"] == str(len(bairro["arquivos"]))
        assert {campo: linha[campo] for campo in rodadas.CAMPOS_PROPRIOS_RODADAS[modelo]} == {
            campo: str(valor) for campo, valor in rodadas.parametros(modelo).items()}
        assert len(rodadas.ler_resultados(modelo, 1)) == len(teste["arquivos"])
        erros = [r for r in medidas[modelo]["resultados"] if r["casa"] in ("FP", "FN")]
        copias = sorted(p.name for casa in ("FP", "FN") if c.pasta_erros(1, casa).exists()
                        for p in c.pasta_erros(1, casa).iterdir())
        assert copias == sorted(f"p{r['p_cachorro_tela']}_{r['arquivo'][:-4]}_{modelo}.jpg" for r in erros)
