"""Extrator de características: cada imagem vira 1.280 números.

O EfficientNetB0 (Keras/TensorFlow) é usado sem a parte final (include_top=False),
com pooling="avg" e os pesos do ImageNet. Ele espera a imagem crua (pixels de 0 a
255): a normalização já está dentro do modelo, por isso o código só redimensiona
para 224 × 224. Se você dividir por 255 antes, a imagem é preparada duas vezes: o
código roda sem erro, mas o resultado piora (bug silencioso, que só aparece na matriz).

Bairro, teste fixo e upload passam sempre pelo mesmo carregar_imagem e pelo mesmo
extrator. Numa rodada de troca para o MobileNetV2, a preparação troca junto
(o preprocess_input dele).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensagens do TensorFlow no terminal

import numpy as np
from PIL import Image, ImageOps

import config

EXTRATORES = ("efficientnetb0", "mobilenetv2")
_FILTRO = getattr(Image, "Resampling", Image).BILINEAR


def carregar_imagem(origem):
    """Abre a imagem (caminho ou arquivo enviado) e só redimensiona para 224 × 224."""
    with Image.open(origem) as original:
        em_pe = ImageOps.exif_transpose(original)   # foto de celular deitada fica em pé
        imagem = em_pe.convert("RGB").resize(config.TAMANHO_IMAGEM, _FILTRO)
    return np.asarray(imagem, dtype=np.float32)     # pixels crus, de 0 a 255


def sem_preparo(lote):
    """EfficientNetB0: nada a fazer, a preparação já está embutida no modelo."""
    return lote


def carregar_extrator(nome=config.EXTRATOR_PADRAO):
    """Devolve (extrator, preparar). Na primeira vez, o Keras baixa os pesos do ImageNet."""
    formato = (*config.TAMANHO_IMAGEM, 3)
    if nome == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        extrator = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=formato)
        return extrator, sem_preparo
    if nome == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        extrator = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=formato)
        return extrator, preprocess_input
    raise ValueError(f"Extrator desconhecido: {nome}. Use um destes: {', '.join(EXTRATORES)}")


def extrair(extrator, preparar, origens, tamanho_lote=32, mostrar_progresso=False):
    """Transforma cada imagem em 1.280 números: devolve uma linha por imagem."""
    vetores = []
    for inicio in range(0, len(origens), tamanho_lote):
        lote = np.stack([carregar_imagem(origem) for origem in origens[inicio:inicio + tamanho_lote]])
        vetores.append(extrator.predict(preparar(lote.copy()), verbose=0))
        if mostrar_progresso:
            print(f"  extraídas {min(inicio + tamanho_lote, len(origens))}/{len(origens)}")
    return np.concatenate(vetores).astype(np.float32)
