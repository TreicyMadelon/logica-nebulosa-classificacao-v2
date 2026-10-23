"""Testes da leitura das anotações (XML) e do sorteio, com arquivos inventados."""
from collections import Counter

import pytest

from fontes import candidatas_do_codigo, ler_anotacao, sortear, sortear_em_rodizio

XML_VOC = """<annotation>
  <filename>2008_000001.jpg</filename>
  <object>
    <name>person</name><difficult>0</difficult>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>210</xmax><ymax>310</ymax></bndbox>
    <part><name>head</name><bndbox><xmin>50</xmin><ymin>10</ymin><xmax>90</xmax><ymax>50</ymax></bndbox></part>
  </object>
  <object>
    <name>dog</name><difficult>0</difficult>
    <bndbox><xmin>220</xmin><ymin>100</ymin><xmax>260.5</xmax><ymax>140</ymax></bndbox>
  </object>
</annotation>"""

XML_IMAGENET = """<annotation>
  <filename>ILSVRC2012_val_00000001</filename>
  <object><name>n01751748</name><difficult>0</difficult>
    <bndbox><xmin>111</xmin><ymin>108</ymin><xmax>441</xmax><ymax>332</ymax></bndbox></object>
</annotation>"""


def test_le_so_os_objetos_e_ignora_as_partes(tmp_path):
    caminho = tmp_path / "voc.xml"
    caminho.write_text(XML_VOC, encoding="utf-8")
    anotacao = ler_anotacao(caminho)
    assert anotacao["arquivo"] == "2008_000001.jpg"
    assert [objeto["nome"] for objeto in anotacao["objetos"]] == ["person", "dog"]   # "head" é parte
    assert anotacao["objetos"][0]["area"] == 200 * 300
    assert anotacao["objetos"][1]["area"] == pytest.approx(40.5 * 40)


def test_le_o_codigo_do_imagenet(tmp_path):
    caminho = tmp_path / "imagenet.xml"
    caminho.write_text(XML_IMAGENET, encoding="utf-8")
    assert [objeto["nome"] for objeto in ler_anotacao(caminho)["objetos"]] == ["n01751748"]


def test_rodizio_divide_igualmente_entre_as_categorias():
    grupos = {categoria: [{"arquivo": f"{categoria}_{i:02d}"} for i in range(30)]
              for categoria in ("bird", "cat", "cow", "horse", "sheep")}
    escolhidas = sortear_em_rodizio(grupos, 50, semente="42-teste")
    assert len(escolhidas) == 50
    assert set(Counter(foto["arquivo"].split("_")[0] for foto in escolhidas).values()) == {10}
    assert sortear_em_rodizio(grupos, 50, semente="42-teste") == escolhidas   # semente fixa: mesmo sorteio


def foto_voc(arquivo, categoria, tem_cachorro=False, dificil=False):
    return {"arquivo": arquivo, "dataset": "voc", "categoria_original": categoria,
            "tem_cachorro": tem_cachorro, "principal_dificil": dificil}


def test_regra_de_inclusao_e_fotos_ja_usadas():
    indice = {
        ("voc", "cat"): [foto_voc(f"voc_gato_{i}.jpg", "cat") for i in range(4)]
                        + [foto_voc("voc_gato_com_cao.jpg", "cat", tem_cachorro=True)],
        ("voc", "dog"): [foto_voc("voc_cao.jpg", "dog"), foto_voc("voc_cao_dificil.jpg", "dog", dificil=True)],
    }
    # não cachorro: nenhuma foto com cachorro anotado
    gatos = candidatas_do_codigo(indice, "nao_cachorro", "voc", "cat")
    assert "voc_gato_com_cao.jpg" not in {foto["arquivo"] for foto in gatos}
    # cachorro: o cão é o principal e não está marcado como difícil
    assert [foto["arquivo"] for foto in candidatas_do_codigo(indice, "cachorro", "voc", "dog")] == ["voc_cao.jpg"]
    # fotos já usadas não voltam
    escolhidas = sortear(indice, "nao_cachorro", "voc", ["cat"], 3, "s", {"voc_gato_0.jpg"})
    assert {foto["arquivo"] for foto in escolhidas} == {"voc_gato_1.jpg", "voc_gato_2.jpg", "voc_gato_3.jpg"}
