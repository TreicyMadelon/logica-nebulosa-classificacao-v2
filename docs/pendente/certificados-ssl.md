# Certificados SSL do Python 3.12 não estão instalados

**Situação:** pendente · **anotado em:** 13/10/2026 · **impacto no projeto hoje:** nenhum

## O que acontece

Qualquer download feito de dentro do Python com a biblioteca `urllib` falha com:

```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

Apareceu ao carregar o extrator pela primeira vez, quando o Keras tenta baixar os pesos do
EfficientNetB0 de `storage.googleapis.com`.

## A causa

O `python3.12` desta máquina é o instalador oficial do python.org:

```
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
```

Ele procura os certificados em
`/Library/Frameworks/Python.framework/Versions/3.12/etc/openssl/cert.pem`,
e esse arquivo **não existe**. O instalador do python.org não traz os certificados prontos:
ele deixa um script para você rodar depois, o `Install Certificates.command`, e esse passo
nunca foi feito.

Não é problema do `.venv` nem do código do projeto. Um venv herda a configuração de SSL do
Python que o criou, então todo venv feito com esse Python 3.12 nasce com a mesma falha.

## Por que o YOLO baixou e o Keras não

| biblioteca | como baixa | resultado |
|---|---|---|
| `ultralytics` (yolov8n.pt) | usa `requests`, que carrega o pacote `certifi` embutido | funcionou |
| `keras` (pesos do EfficientNetB0) | usa `urllib` direto, que cai no caminho vazio do sistema | falhou |
| `pip` | traz o próprio `certifi` | funcionou |

## Por que não bloqueia o projeto

Os três arquivos que precisavam de download já estão em cache no disco:

- `~/.keras/models/efficientnetb0_notop.h5` (16 MB)
- `~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5` (9 MB)
- `yolov8n.pt`, na raiz do projeto (6,2 MB)

Foram baixados com `curl`, que usa os certificados do próprio macOS e não passa pelo Python.
Nenhuma etapa restante do projeto (`triagem.py --aplicar`, `montar_bairro.py`, `avaliar.py`,
`streamlit run app.py`) baixa qualquer coisa.

## Quando voltaria a incomodar

- Rodar o projeto em outra máquina, ou recriar o `.venv` em um computador sem esse cache.
- Apagar `~/.keras/models/` ou o `yolov8n.pt`.
- Trocar o extrator por um terceiro modelo, numa rodada de melhoria (o `config.py` já prevê
  EfficientNetB0 e MobileNetV2, e os dois estão em cache; um modelo novo precisaria baixar).
- Qualquer biblioteca nova que baixe com `urllib`.

## Como resolver

**Recomendado — conserta o Python 3.12 inteiro e todo venv futuro:**

```
"/Applications/Python 3.12/Install Certificates.command"
```

É o passo que o próprio instalador do python.org pede. Ele copia o pacote de certificados do
`certifi` para o caminho que está faltando. Mexe só no framework do Python 3.12.

**Alternativa contida ao projeto**, se não quiser mexer no Python do sistema: criar o arquivo
`.venv/lib/python3.12/site-packages/sitecustomize.py` com

```python
import os
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
```

Vale só para este venv, e some se o `.venv` for recriado.

**Contorno pontual**, para um download específico: baixar o arquivo com `curl` e colocá-lo na
pasta de cache que a biblioteca espera. Foi o que fizemos com os pesos do Keras e do YOLO.

## Como conferir se já foi resolvido

```
.venv/bin/python -c "
import os, ssl
caminho = ssl.get_default_verify_paths().openssl_cafile
print(caminho, '| existe:', os.path.exists(caminho))
"
```

Resolvido quando a saída terminar em `existe: True`.
