"""Testes do split 80/20, do teste fixo e da distribuição (só metadados, sem imagens)."""
from collections import Counter

from dados import (categoria_planejada, contar_distribuicao, definir_teste_fixo,
                   diferencas_distribuicao, sortear_teste_fixo)


def base_falsa():
    """400 linhas de metadados: 100 de cada dataset de origem em cada classe."""
    linhas = []
    for classe in ("cachorro", "nao_cachorro"):
        for dataset in ("imagenet", "voc"):
            for i in range(100):
                linhas.append({"arquivo": f"{dataset}_{classe}_{i:03d}.jpg", "classe": classe,
                               "dataset": dataset, "categoria_original": "x"})
    return linhas


def test_split_80_20_estratificado_por_classe_e_dataset():
    linhas = base_falsa()
    teste = sortear_teste_fixo(linhas)
    assert len(teste) == 80
    no_teste = set(teste)
    por_grupo = Counter((l["classe"], l["dataset"]) for l in linhas if l["arquivo"] in no_teste)
    assert sorted(por_grupo.values()) == [20, 20, 20, 20]    # 20 de cada dataset em cada classe
    assert sortear_teste_fixo(linhas) == teste               # random_state=42: sempre o mesmo sorteio


def test_teste_fixo_nao_muda_quando_entram_imagens_novas(tmp_path):
    linhas = base_falsa()
    caminho = tmp_path / "teste_fixo.txt"
    teste, criado, avisos = definir_teste_fixo(linhas, caminho)
    assert criado and caminho.exists() and not avisos
    novas = [{"arquivo": f"nova_{i}.jpg", "classe": "cachorro", "dataset": "voc", "categoria_original": "dog"}
             for i in range(20)]
    teste_depois, criado_depois, _ = definir_teste_fixo(linhas + novas, caminho)
    assert not criado_depois and teste_depois == teste        # as novas vão só para o bairro


def test_avisa_imagem_do_teste_removida(tmp_path):
    linhas = base_falsa()
    caminho = tmp_path / "teste_fixo.txt"
    teste, _, _ = definir_teste_fixo(linhas, caminho)
    sem_uma = [l for l in linhas if l["arquivo"] != teste[0]]
    teste_depois, _, avisos = definir_teste_fixo(sem_uma, caminho)
    assert len(teste_depois) == 79 and avisos


def test_categorias_reais_da_tabela_de_distribuicao():
    assert categoria_planejada("cachorro", "imagenet", "n02099601") == "raças domésticas"      # golden retriever
    assert categoria_planejada("nao_cachorro", "imagenet", "n02114367") == "outros animais"    # lobo-cinzento
    assert categoria_planejada("nao_cachorro", "imagenet", "n02342885") == "outros animais"    # hamster
    assert categoria_planejada("nao_cachorro", "imagenet", "n02690373") == "objetos e cenários"  # avião
    assert categoria_planejada("cachorro", "voc", "dog") == "cachorro"
    assert categoria_planejada("nao_cachorro", "voc", "cat") == "outros animais"
    assert categoria_planejada("nao_cachorro", "voc", "person") == "pessoas"
    assert categoria_planejada("nao_cachorro", "imagenet", "n02099601") == "fora do plano"     # raça em não cachorro


def test_correcao_de_rotulo_aparece_na_distribuicao():
    linhas = [{"arquivo": "a.jpg", "classe": "nao_cachorro", "dataset": "voc", "categoria_original": "cat"},
              {"arquivo": "b.jpg", "classe": "nao_cachorro", "dataset": "voc", "categoria_original": "person"}]
    planejada = contar_distribuicao(linhas)
    assert diferencas_distribuicao(contar_distribuicao(linhas), planejada) == []
    corrigida = [linhas[0], {**linhas[1], "classe": "cachorro"}]
    assert len(diferencas_distribuicao(contar_distribuicao(corrigida), planejada)) == 2
