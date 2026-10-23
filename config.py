"""Configurações do projeto V2: Fuzzy k-NN e SVM.

O arquivo tem três partes, cada uma com um comentário de início e outro de fim:
- Comum: a régua da comparação, igual para os dois modelos (pastas, bases de origem,
  extrator, split, faixa da área cinzenta, meta, composição da base e YOLO).
- Só do Fuzzy k-NN: todo nome começa com FUZZYKNN_.
- Só do SVM: todo nome começa com SVM_.

Nenhuma configuração sobrescreve outra: um teste falha se um nome aparecer duas vezes,
e o código só lê o config, nunca escreve nele.
Regra das rodadas: mude uma coisa por vez e teste sempre nas mesmas 80 imagens.
"""
from pathlib import Path

from codigos_imagenet import OBJETOS_E_CENARIOS_IMAGENET, OUTROS_ANIMAIS_IMAGENET

# ================================================================== COMUM · início
# A régua da comparação: vale igual para os dois modelos.

# ------------------------------------------------------------------ pastas
RAIZ = Path(__file__).resolve().parent

PASTA_BASE = RAIZ / "base_imagens"            # a base única (400 imagens)
PASTAS_CLASSE = {
    "cachorro": PASTA_BASE / "cachorro",          # rótulo 1
    "nao_cachorro": PASTA_BASE / "nao_cachorro",  # rótulo 0
}
ARQ_METADADOS = PASTA_BASE / "metadados.csv"  # arquivo, classe, dataset, categoria_original
ARQ_TESTE_FIXO = PASTA_BASE / "teste_fixo.txt"  # as 80 imagens do teste fixo
ARQ_DISTRIBUICAO = PASTA_BASE / "distribuicao_planejada.csv"  # quantas por classe, dataset e categoria

PASTA_TRIAGEM = RAIZ / "triagem"              # relatório e decisões da triagem (YOLO)
PASTA_DESCARTADAS = PASTA_TRIAGEM / "descartadas"  # imagens substituídas (não são apagadas)
ARQ_DESCARTADAS = PASTA_TRIAGEM / "descartadas.csv"

PASTA_VETORES = RAIZ / "vetores"              # os 1.280 números de cada imagem
ARQ_BAIRRO = PASTA_VETORES / "bairro.npz"     # 320 imagens: números + rótulos
ARQ_TESTE = PASTA_VETORES / "teste.npz"       # 80 imagens do teste fixo

# Bases de origem: os datasets ImageNet (validação) e Pascal VOC 2012.
PASTA_IMAGENET_IMAGENS = RAIZ / "bases_origem" / "imagenet" / "imagens"      # ILSVRC2012_img_val.tar
PASTA_IMAGENET_ANOTACOES = RAIZ / "bases_origem" / "imagenet" / "anotacoes"  # ILSVRC2012_bbox_val_v3.tgz
PASTA_VOC = RAIZ / "bases_origem" / "VOCdevkit" / "VOC2012"                  # VOCtrainval_11-May-2012.tar

# ------------------------------------------------------------------ área cinzenta
# Grau de cachorro (a pertinência no Fuzzy k-NN, a probabilidade no SVM), limites inclusive
# (grau da classe vencedora de 0,50 a 0,60, com duas casas). A faixa é decidida pelo mesmo
# número de duas casas que decide a classe. No SVM, a tela escreve a faixa em porcentagem.
FAIXA_CINZENTA = (0.40, 0.60)

# ------------------------------------------------------------------ extrator
TAMANHO_IMAGEM = (224, 224)     # só redimensiona; a normalização já está dentro do modelo
EXTRATOR_PADRAO = "efficientnetb0"   # rodada de troca do extrator: "mobilenetv2"

# ------------------------------------------------------------------ split 80/20
SEMENTE = 42                    # random_state: o sorteio sai sempre igual
FRACAO_TESTE = 0.2              # test_size: 80 das 400 imagens

# ------------------------------------------------------------------ meta no teste fixo
META_ACURACIA = 0.90            # pelo menos 72 das 80
META_MAX_FN = 6
META_MAX_FP = 6

# ------------------------------------------------------------------ composição da base
# Nomes das categorias reais (a "categoria" da tabela de distribuição do PRD).
CAT_RACAS = "raças domésticas"
CAT_CACHORRO = "cachorro"
CAT_OUTROS_ANIMAIS = "outros animais"
CAT_OBJETOS = "objetos e cenários"
CAT_PESSOAS = "pessoas"
CAT_FORA_DO_PLANO = "fora do plano"   # aparece só se uma correção de rótulo tirar a imagem do plano

# (classe, dataset de origem, categoria real, quantidade) — a tabela "Dados" do PRD.
PLANO_BASE = [
    ("cachorro", "imagenet", CAT_RACAS, 100),
    ("cachorro", "voc", CAT_CACHORRO, 100),
    ("nao_cachorro", "imagenet", CAT_OUTROS_ANIMAIS, 50),   # com hamster e canídeos selvagens
    ("nao_cachorro", "voc", CAT_OUTROS_ANIMAIS, 50),        # pássaro, gato, vaca, cavalo, ovelha
    ("nao_cachorro", "imagenet", CAT_OBJETOS, 50),
    ("nao_cachorro", "voc", CAT_PESSOAS, 50),
]

# Códigos sorteados em cada categoria do ImageNet (a tabela fica em codigos_imagenet.py).
# Os cachorros do ImageNet são todas as raças domésticas (regra em codigos_imagenet.py).
CODIGOS_IMAGENET = {
    CAT_OUTROS_ANIMAIS: OUTROS_ANIMAIS_IMAGENET,
    CAT_OBJETOS: OBJETOS_E_CENARIOS_IMAGENET,
}

# Pascal VOC: a categoria vem escrita no XML.
VOC_CACHORRO = "dog"
VOC_PESSOAS = "person"
VOC_OUTROS_ANIMAIS = {
    "bird": "pássaro",
    "cat": "gato",
    "cow": "vaca",
    "horse": "cavalo",
    "sheep": "ovelha",
}
NOMES_VOC = {VOC_CACHORRO: "cachorro", VOC_PESSOAS: "pessoa", **VOC_OUTROS_ANIMAIS}

# ------------------------------------------------------------------ triagem (YOLO)
YOLO_MODELO = "yolov8n.pt"      # Ultralytics, treinado no COCO (tem a classe "dog")
YOLO_CONFIANCA_MINIMA = 0.25    # detecções abaixo disso são ignoradas (padrão da Ultralytics)
YOLO_CONFIANCA_INCERTA = 0.50   # cachorro detectado abaixo disso conta como detecção incerta
ANIMAIS_COCO = {"bird", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
FRACAO_AMOSTRA = 0.05           # parte das imagens não sinalizadas que você confere à mão
# ================================================================== COMUM · fim

# ================================================================== SÓ DO FUZZY K-NN · início
FUZZYKNN_K = 5                  # quantas vizinhas votam; fixo durante toda a V2
FUZZYKNN_PASTA_RESULTADOS = RAIZ / "resultados_fuzzyknn"   # rodadas do Fuzzy k-NN no teste fixo
# ================================================================== SÓ DO FUZZY K-NN · fim

# ================================================================== SÓ DO SVM · início
SVM_C = 1.0                     # padrão do scikit-learn; fixo durante toda a V2
SVM_GAMMA = "scale"             # padrão do scikit-learn; fixo durante toda a V2
SVM_PASTA_RESULTADOS = RAIZ / "resultados_svm"             # rodadas do SVM no teste fixo
# ================================================================== SÓ DO SVM · fim
