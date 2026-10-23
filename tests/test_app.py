"""Proteção 10: as duas homes não dividem estado.

Lê o app.py como texto (ast), sem abrir o Streamlit: toda chave de widget e todo nome
guardado em st.session_state saem da função chave(dono, nome), que põe o modelo no nome
(historico_fuzzyknn, envio_svm, rever_fuzzyknn_<id>).
"""
import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARVORE = ast.parse((RAIZ / "app.py").read_text(encoding="utf-8"))


def e_session_state(no):
    return (isinstance(no, ast.Attribute) and no.attr == "session_state"
            and isinstance(no.value, ast.Name) and no.value.id == "st")


def e_chamada_de_chave(no):
    return isinstance(no, ast.Call) and isinstance(no.func, ast.Name) and no.func.id == "chave"


def funcao_chave():
    """A função chave do app, executada sozinha."""
    definicao = next(no for no in ARVORE.body if isinstance(no, ast.FunctionDef) and no.name == "chave")
    espaco = {}
    exec(compile(ast.Module(body=[definicao], type_ignores=[]), "app.py", "exec"), espaco)
    return espaco["chave"]


def test_chave_poe_o_modelo_no_nome():
    chave = funcao_chave()
    assert chave("fuzzyknn", "historico") == "historico_fuzzyknn"
    assert chave("svm", "envio") == "envio_svm"
    assert chave("fuzzyknn", "rever", "abc") == "rever_fuzzyknn_abc"
    for nome in ("historico", "selecionada", "ultimo_envio", "envio", "rodada"):
        assert chave("fuzzyknn", nome) != chave("svm", nome)


def test_toda_chave_de_widget_sai_da_funcao_chave():
    chaves = [palavra.value for no in ast.walk(ARVORE) if isinstance(no, ast.Call)
              for palavra in no.keywords if palavra.arg == "key"]
    assert chaves, "o app tem widgets com chave"
    assert all(e_chamada_de_chave(valor) for valor in chaves)


def test_session_state_so_com_nomes_da_funcao_chave():
    usos = 0
    for no in ast.walk(ARVORE):
        if isinstance(no, ast.Attribute) and e_session_state(no.value):
            raise AssertionError(f"st.session_state.{no.attr} (linha {no.lineno}): use st.session_state[chave(...)]")
        if isinstance(no, ast.Assign) and e_session_state(no.value):
            raise AssertionError(f"linha {no.lineno}: st.session_state não vai para outra variável")
        if isinstance(no, ast.Subscript) and e_session_state(no.value):
            assert e_chamada_de_chave(no.slice), f"linha {no.lineno}"
            usos += 1
        if isinstance(no, ast.Compare) and any(e_session_state(c) for c in no.comparators):
            assert e_chamada_de_chave(no.left), f"linha {no.lineno}"
    assert usos > 0
