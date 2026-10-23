"""Teste do .gitignore da V2 (proteção 11), com o próprio Git: as pastas de imagens de FP e
FN dos dois modelos ficam fora, e os CSVs das rodadas sobem.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

import rodadas

RAIZ = Path(__file__).resolve().parent.parent


def ignorado(repositorio, caminho):
    """True se o Git ignora o arquivo (git check-ignore devolve 0), False se ele sobe (1)."""
    saida = subprocess.run(["git", "-C", str(repositorio), "check-ignore", "-q", caminho.as_posix()],
                           capture_output=True, text=True)
    assert saida.returncode in (0, 1), saida.stderr
    return saida.returncode == 0


@pytest.mark.skipif(shutil.which("git") is None, reason="precisa do git instalado")
def test_gitignore_deixa_as_imagens_de_fora_e_sobe_os_csvs(tmp_path):
    repositorio = tmp_path / "repositorio"
    subprocess.run(["git", "init", "-q", str(repositorio)], check=True)
    shutil.copy(RAIZ / ".gitignore", repositorio / ".gitignore")

    sobem, ficam_fora = [], []
    for modelo in rodadas.MODELOS:
        # Os nomes de verdade, tirados da mesma classe que o caminhos(modelo) usa.
        pasta = rodadas.Caminhos(modelo, Path(f"resultados_{modelo}"))
        sobem += [pasta.tabela, pasta.resultados(1), pasta.resultados(12)]
        ficam_fora += [pasta.copia_imagem(1, "FP", "0.55", "imagenet_ILSVRC2012_val_00000001.JPEG"),
                       pasta.copia_imagem(1, "FN", "0.45", "voc_2008_000001.jpg"),
                       pasta.copia_imagem(12, "FP", "0.61", "voc_2009_000002.jpg")]
    sobem += [Path("base_imagens/metadados.csv"), Path("base_imagens/teste_fixo.txt"), Path("triagem/relatorio.csv")]
    ficam_fora += [Path("base_imagens/cachorro/x.jpg"), Path("base_imagens/nao_cachorro/y.jpg"),
                   Path("vetores/bairro.npz"), Path("bases_origem/imagenet/imagens/z.JPEG"),
                   Path("resultados/rodada_1/FP/p0.55_x.jpg")]           # a linha antiga continua

    for caminho in sobem + ficam_fora:
        (repositorio / caminho).parent.mkdir(parents=True, exist_ok=True)
        (repositorio / caminho).write_bytes(b"x")
    assert [c.as_posix() for c in sobem if ignorado(repositorio, c)] == []
    assert [c.as_posix() for c in ficam_fora if not ignorado(repositorio, c)] == []
