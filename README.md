# Cachorro × não cachorro · V2: Fuzzy k-NN e SVM

Trabalho final de Lógica Nebulosa (prof. Mardonio). O sistema recebe uma imagem e responde se é cachorro ou não, com o grau da resposta, em dois modelos que aprendem do mesmo bairro e são comparados nas mesmas 80 imagens. O EfficientNetB0 descreve a imagem em 1.280 números. O **Fuzzy k-NN** (k = 5, peso 1/d²) compara esses números com um bairro de 320 imagens rotuladas e dá a pertinência da classe vencedora, um número de 0 a 1. O **SVM**, treinado nesse mesmo bairro, dá a probabilidade da classe vencedora, mostrada em porcentagem. Nos dois, a classe é a de maior grau; com o grau de cachorro entre 0,40 e 0,60 (40% e 60% no SVM), a tela também mostra o aviso de área cinzenta.

## A lógica do projeto

O raciocínio segue o mapa do projeto, de cima para baixo: requisitos → classes e tipo de aprendizado → dados → treino → grau (pertinência no Fuzzy k-NN, probabilidade no SVM) → classe pelo maior grau e área cinzenta → avaliação e comparação dos dois modelos → melhoria (futuro). Cada passo existe por causa do anterior.

### 1. Os requisitos definem o problema

Projeto fuzzy com classificação binária.

A classificação binária define as **duas classes**: cachorro (rótulo 1) e não cachorro (rótulo 0). "Não cachorro" é tudo o que não é cachorro: gatos, outros animais, pessoas, objetos e cenários.

### 2. As classes pedem aprendizado supervisionado

As imagens já vêm marcadas como cachorro ou não, então o problema é **supervisionado**: cada exemplo traz a resposta certa (o rótulo). Fuzzy k-NN na versão clássica de classificação, sem rótulo, as vizinhas não diriam nada.

O SVM também usa os rótulos: no fit, eles traçam a fronteira entre as classes.

### 3. As bases de origem fornecem as imagens rotuladas

As 400 imagens vêm de duas bases de origem: o conjunto de validação do **ImageNet** (objeto em destaque, muita variedade de animais, como o hamster) e o **Pascal VOC** (cenas do dia a dia, mais parecidas com as fotos que um usuário envia). Cada classe tem 100 imagens de cada base. Essa mistura simétrica impede que o estilo de uma base vire pista para separar as classes (o viés de dataset, de Torralba e Efros). Do ImageNet entram só fotos de validação, que o EfficientNetB0 nunca viu no treino.

As imagens rotuladas precisam de duas coisas:

- **Um "não X" variado**: se "não cachorro" tivesse só gatos, o sistema aprenderia "cachorro × gato". Por isso entram outros animais, canídeos selvagens (lobo, coiote, raposa, dingo, mabeco, bons candidatos à área cinzenta), pessoas, objetos e cenários.
- **Balanceamento**: 200 + 200. Com muito mais exemplos de uma classe, o sistema tenderia a responder sempre ela, e a acurácia enganaria.

A base equilibrada passa pela **triagem com YOLO**, que confere se cada foto bate com a pasta (por exemplo, um cachorro ao fundo de uma foto em `nao_cachorro/`). Rótulo errado ensina errado. A triagem vem antes do split.

### 4. O split separa o treino da prova

O **split 80/20** divide as 400 imagens: 80% vão para o treino (o bairro, 320 imagens) e 20% para o teste (80 imagens que o sistema nunca viu, como uma prova com questões inéditas).

- O **stratify** equilibra o split: sorteia dentro de cada grupo classe + dataset de origem, então o teste fica com 20 imagens de cada base em cada classe, a mesma simetria da base.
- O **random_state=42** fixa o sorteio: rodar de novo dá a mesma divisão. Se o resultado mudar, foi por algo que você alterou.
- O arquivo `base_imagens/teste_fixo.txt` guarda as 80 sorteadas. Assim, quando uma melhoria acrescenta imagens, o teste continua o mesmo e as novas vão só para o bairro.
- Os dois modelos usam o mesmo split: o mesmo bairro e as mesmas 80.

### 5. Treinar é montar o bairro (e fazer o fit do SVM)

No Fuzzy k-NN, treinar não ajusta pesos nem calcula loss: é **montar a base de referência**. O EfficientNetB0 (pré-treinado, usado como extrator de características) transforma cada uma das 320 imagens em 1.280 números, que ficam guardados com os rótulos. Esse é o bairro. Ele só é montado de novo quando mudam os dados ou o extrator.

No **SVM**, treinar é o *fit*: sobre os mesmos números e rótulos, um pipeline do scikit-learn padroniza os números (`StandardScaler`, com a escala aprendida só no bairro), traça a fronteira (`SVC`, com C = 1 e gamma "scale", os padrões, fixos na V2) e calibra a probabilidade (`CalibratedClassifierCV`, com `ensemble=False`). Não usa `class_weight`, porque o bairro já é 160 + 160. O fit é refeito a partir do bairro sempre que precisa, sem arquivo guardado, e não sorteia nada: o mesmo bairro, na mesma ordem, dá sempre a mesma probabilidade. Por isso o código nunca reordena o bairro.

### 6. O grau fuzzy: o Fuzzy k-NN dá a pertinência

Quando chega uma imagem nova, o mesmo EfficientNetB0 transforma a imagem em 1.280 números, e o **Fuzzy k-NN** compara esses números com o bairro:

1. mede a distância da imagem nova até cada imagem do bairro;
2. pega as 5 mais próximas (as vizinhas);
3. dá a cada vizinha o peso 1/d²: quem está mais perto pesa mais;
4. calcula a pertinência: pesos das vizinhas cachorro ÷ soma de todos os pesos.

Pela vizinhança: o KNN comum só conta as casas vizinhas ("4 cachorros e 1 gato, então é cachorro"). O Fuzzy k-NN é o vizinho detalhista, que mede a distância de cada casa e responde em grau: "sua casa pertence 0,65 à rua dos cachorros e 0,35 à rua dos gatos".

Atenção: Pertinência diz o quanto a imagem pertence ao conjunto "cachorro";

**No SVM**, o grau é a **probabilidade** (`predict_proba`). O SVM traça uma fronteira entre cachorro e não cachorro com o bairro e guarda as imagens que ficam perto dela (os vetores de suporte). Uma imagem nova cai de um lado da fronteira, a uma distância: longe, do lado do cachorro, a probabilidade chega perto de 1; perto da fronteira, fica no meio. No lugar das vizinhas, a home do SVM mostra as **5 imagens do bairro que mais puxaram para a classe vencedora** na conta daquela imagem, com o rótulo de cada uma. O SVM não vota: a pontuação é a soma de todos os vetores de suporte, e a caixa mostra quanto as 5 somam e quanto vem do resto. Probabilidade é a chance de um evento, não pertinência: cada home usa a palavra do seu modelo, e os dois são comparados pelas métricas.

### 7. O que define as faixas e a área cinzenta

Cada modelo dá um grau de cachorro para a imagem: a pertinência no Fuzzy k-NN e a probabilidade no SVM. É desse grau que saem a classe e a faixa. As regras são as mesmas nos dois modelos; muda o nome do grau (pertinência ou porcentagem), a forma de escrever o número e o que a caixa usa para explicar a resposta.

**O que vale para os dois**

A classe é a de **maior grau**, decidida por um `if` comum. 

O grau de cachorro cria três situações:

| grau de cachorro | o que a tela mostra |
|---|---|
| abaixo de 0,40 | "Não é cachorro", com o grau, na caixa vermelha "❌ Imagem na área não é cachorro" |
| de 0,40 a 0,60 | a classe com o grau, na caixa amarela "⚠ Imagem na área cinzenta" |
| acima de 0,60 | "É cachorro", com o grau, na caixa verde "🐶 Imagem na área é cachorro" |

Os limites ficam no `config.py` (`FAIXA_CINZENTA`), um só para os dois modelos.

A caixa aparece sempre, nas três faixas: ela traz a regra da faixa e os dois graus, o da classe vencedora primeiro. Abaixo dela vêm as fotos de 5 imagens do bairro, também nas três faixas — são elas que explicam de onde saiu o grau, tanto numa resposta folgada quanto numa apertada. O que só a área cinzenta tem é o aviso de que a decisão foi apertada.

A área cinzenta não muda a classe: só marca que a decisão foi apertada.

**No Fuzzy k-NN**

O grau é a pertinência a cachorro, de 0 a 1, e as faixas são abaixo de 0,40, de 0,40 a 0,60 e acima de 0,60. A caixa traz a linha "pertinência da imagem:", com as duas pertinências, e a linha "parecida com:", que conta quantas das 5 vizinhas são cachorro e quantas não são. As fotos são as 5 vizinhas, da mais próxima para a mais distante; na área cinzenta, são elas que explicam a dúvida.

**No SVM**

O grau é a probabilidade de cachorro, escrita em porcentagem, e as faixas são abaixo de 40%, de 40% a 60% e acima de 60%, com as mesmas caixas. A caixa traz a linha "probabilidade da imagem:", com as duas probabilidades em porcentagem, a pontuação do SVM (positivo puxa para cachorro, negativo para não cachorro) e quanto dessa pontuação vem das 5 imagens que mais puxaram para a classe vencedora e quanto vem do resto (os outros vetores de suporte e o termo fixo). As fotos são essas imagens, até 5 (nas 80 do teste, sempre 5), da que mais pesou para a que menos pesou. O SVM não vota: as 5 não são uma contagem, por isso a caixa não tem a linha "parecida com:".

### 8. O que vira a mensagem na tela

A tela junta o binário e o resultado ao mesmo tempo: a classe responde sim ou não, e o grau diz o quanto. Os dois modelos montam a mensagem, cada um a partir do seu grau de cachorro com duas casas; muda a palavra do grau e a forma de escrever o valor.

**O que vale para os dois**

A mensagem principal segue a regra `<classe> · <grau> <valor>`: a classe é "É cachorro" ou "Não é cachorro", iguais nas duas homes, e o valor é o grau da classe vencedora calculado para aquela imagem, escrito do jeito de cada modelo. Abaixo dela vem a caixa colorida da faixa (passo 7), que repete a vencedora e mostra a outra classe.

Arredondamento: o grau de cachorro é arredondado para **duas casas**, pelo arredondamento convencional (o meio sobe), e **é desse único número que sai tudo** — a classe (acima de 0,50 é cachorro; 0,50 fica não cachorro), a faixa, a cor da caixa e os dois graus que a tela mostra. O grau de não cachorro é 1 menos esse valor, então os dois sempre somam 1,00 (100% no SVM).

Por isso a tela nunca se contradiz. Um grau de 0,3965 aparece como 0,40 **e** entra na área cinzenta, em vez de aparecer como 0,40 numa caixa que diz "abaixo de 0,40". E um de 0,5049 aparece como 0,50 **e** é classificado como não cachorro, em vez de dizer "É cachorro" com 0,50.

As imagens enviadas ficam no bloco "Imagens anteriores" da home de cada modelo só durante a sessão e nunca entram na base.

**No Fuzzy k-NN**

A mensagem segue a regra `<classe> · pertinência <valor>`, com o valor de 0 a 1 e duas casas decimais. A caixa mostra as duas pertinências na linha "pertinência da imagem:". 

**No SVM**

A mensagem segue a regra `<classe> · probabilidade <valor>%`, com o valor em porcentagem. Ali a porcentagem cabe, porque o número é uma probabilidade. A porcentagem é o número de duas casas vezes 100, sem arredondar de novo, então ela nunca discorda da classe nem da faixa, e o empate aparece como 50%, "Não é cachorro". A pontuação do SVM aparece só na caixa, com duas casas e sinal: ela ajuda a explicar a probabilidade, mas não decide a classe nem a faixa.

### 9. A matriz de confusão avalia o grau fuzzy (No teste)

As 80 imagens do teste fixo passam pelos dois modelos, num comando só, e a **matriz de confusão** de cada um compara as respostas com os rótulos guardados: VP, FN, FP e VN. Da matriz saem:

- a **acurácia**: (VP + VN) ÷ total. A meta é ≥ 90% (72 das 80), com FN ≤ 6 e FP ≤ 6, a mesma para os dois modelos;
- a pergunta **FP × FN: qual pesa?** O sistema erra mais chamando não cachorro de cachorro (FP) ou deixando cachorro passar (FN)? Isso diz onde olhar.

A área cinzenta também vira uma **galeria** no teste fixo, com as 4 situações (VP, FN, VN, FP) e as vizinhas de cada imagem (no SVM, as imagens que mais pesaram). Os erros dela mostram as lacunas do bairro.

O **comparativo** põe a mesma rodada dos dois modelos lado a lado: acurácia, VP, FN, FP, VN, meta e área cinzenta, com a diferença em número de imagens, e a lista imagem a imagem (os dois acertaram, só o Fuzzy k-NN, só o SVM, os dois erraram). Ganha quem acerta mais imagens nas mesmas 80. Com o mesmo número de acertos, é empate, e uma nota mostra qual modelo teve menos imagens na área cinzenta, sem mudar o resultado: a área cinzenta mostra dúvida, não acerto, e a faixa lê pertinência num modelo e probabilidade no outro.

### 10. Se o modelo erra muito, a matriz pede a melhoria

A melhoria monta o bairro de novo (ou ajusta o código) e testa os dois modelos nas **mesmas 80 imagens**, uma mudança por vez:

- **dados**: acrescentar exemplos parecidos com os erros (ex.: lobos em não cachorro), sempre na mesma quantidade em cada classe e em cada base de origem, passando pela triagem;
- **extrator**: trocar o EfficientNetB0 pelo MobileNetV2, o que refaz os dois modelos na mesma rodada;
- **código**: corrigir um erro silencioso. O k (5) do Fuzzy k-NN e o C e o gamma do SVM ficam fixos na V2, para nenhum modelo ser ajustado olhando as 80.

Cada volta é uma rodada, com o mesmo número nos dois modelos e registrada na tabela de cada um. Lida de cima para baixo, a tabela mostra o que cada melhoria fez com os acertos e com cada tipo de erro. E o ciclo volta ao passo 5.

## Arquivos

| arquivo | o que faz | etapa do Fluxo completo |
|---|---|---|
| `config.py` | em três partes: comum (pastas, extrator, split, faixa da área cinzenta, meta, composição da base e YOLO), só do Fuzzy k-NN (k e pasta de resultados) e só do SVM (C, gamma e pasta de resultados) | — |
| `codigos_imagenet.py` | tabela de códigos do ImageNet (ex.: n02342885 = hamster) | 2 |
| `fontes.py` | lê os XML das duas bases e sorteia as fotos | 1 a 3 |
| `montar_base.py` | monta a base de 400 imagens ou acrescenta imagens numa melhoria | 1 a 3 |
| `triagem.py` | YOLO, relatório, decisões e critério de parada | 4 |
| `extrator.py` | imagem → 1.280 números (EfficientNetB0; MobileNetV2 na rodada de troca) | 5 e 8 |
| `montar_bairro.py` | extrai os números e separa o bairro (320) do teste fixo (80) | 5 e 6 |
| `dados.py` | metadados, distribuição, split 80/20, teste fixo e vetores | 3 e 6 |
| `regua.py` | duas casas, decisão pelo maior grau e área cinzenta: a mesma régua para os dois modelos | 10 |
| `fuzzyknn.py` | `fuzzy_knn`: pertinências, vizinhas, avaliação nas 80 e textos com "pertinência" | 9 e 10 |
| `svm.py` | o pipeline do SVM, a probabilidade, as imagens que mais pesaram, a avaliação nas 80 e textos com "probabilidade" | 9 e 10 |
| `avaliacao.py` | matriz de confusão, acurácia, meta e galeria, para os dois modelos | 12 |
| `rodadas.py` | os caminhos de cada modelo (`caminhos(modelo)`), gravar e ler as rodadas e as travas contra salvar errado | 12 |
| `avaliar.py` | o comando da rodada: avalia os dois modelos e grava uma linha em cada tabela | 12 |
| `app.py` | Streamlit: as duas homes (Fuzzy k-NN e SVM) e a visão admin com três abas (Fuzzy k-NN, SVM e Comparativo) | 7 a 11 |
| `tests/` | testes com números inventados | — |

Pastas criadas pelos scripts:

```
base_imagens/
  cachorro/                    rótulo 1
  nao_cachorro/                rótulo 0
  metadados.csv                arquivo, classe, dataset, categoria_original
  teste_fixo.txt               as 80 do teste fixo (gravado na primeira montagem do bairro)
  distribuicao_planejada.csv   quantas por classe, dataset e categoria (a triagem confere)
triagem/                       relatório e decisões; descartadas/ guarda as imagens substituídas
vetores/                       bairro.npz (320 imagens) e teste.npz (80)
resultados_fuzzyknn/           só do Fuzzy k-NN
  rodadas_fuzzyknn.csv         uma linha por rodada
  rodada_N_fuzzyknn/           resultados_fuzzyknn.csv, FP_fuzzyknn/ e FN_fuzzyknn/
resultados_svm/                só do SVM
  rodadas_svm.csv              uma linha por rodada
  rodada_N_svm/                resultados_svm.csv, FP_svm/ e FN_svm/
```

O que é de um modelo leva o nome dele; o que é dos dois é comum. As cópias das imagens de FP e FN ficam fora do Git; os CSVs sobem.

## 1. Instalar

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-triagem.txt     # só para a triagem: traz o PyTorch
```

Use um Python que o TensorFlow suporte (3.10 a 3.12 é o caminho mais seguro). Na primeira vez, o Keras baixa os pesos do EfficientNetB0 (~16 MB) e a Ultralytics baixa o `yolov8n.pt` (~6 MB). Depois, tudo roda localmente.

O `requirements.txt` da V2 fixa a versão do scikit-learn: o SVM depende dela, e trocar a versão vira uma rodada dos dois modelos.

## 2. Baixar as bases de origem

Os dois datasets de origem:

- **ImageNet (validação)**: image-net.org, com cadastro. Arquivos `ILSVRC2012_img_val.tar` (as 50.000 fotos) e `ILSVRC2012_bbox_val_v3.tgz` (as anotações em XML).
- **Pascal VOC 2012**: host.robots.ox.ac.uk/pascal/VOC/voc2012, arquivo `VOCtrainval_11-May-2012.tar`.

Coloque assim (ou mude os caminhos em `config.py`):

```
bases_origem/
  imagenet/
    imagens/        ← as fotos .JPEG de ILSVRC2012_img_val.tar
    anotacoes/      ← os .xml de ILSVRC2012_bbox_val_v3.tgz (pasta val/)
  VOCdevkit/
    VOC2012/        ← de VOCtrainval_11-May-2012.tar (Annotations/ e JPEGImages/)
```

A pasta `bases_origem/` fica fora do Git.

## 3. Parte 1 · construção da base (uma vez)

1. **Montar a base**: `python montar_base.py`
   Copia as 400 fotos para `base_imagens/` e grava os metadados.

   Se você clonou este repositório, use `python montar_base.py --remontar`. As imagens não entram no Git (o ImageNet e o Pascal VOC não permitem redistribuição), mas o `metadados.csv` entra: o `--remontar` copia das bases de origem exatamente as 400 que ele lista, sem sortear de novo. A base sai idêntica à que gerou os resultados registrados, com a substituição da triagem já aplicada — então você pode pular direto para o passo 3.

   | classe | categoria real | dataset de origem | imagens |
   |---|---|---|---|
   | cachorro | raças domésticas | ImageNet (validação) | 100 |
   | cachorro | cachorro | Pascal VOC | 100 |
   | não cachorro | outros animais (com hamster e canídeos selvagens) | ImageNet (validação) | 50 |
   | não cachorro | outros animais (pássaro, gato, vaca, cavalo, ovelha) | Pascal VOC | 50 |
   | não cachorro | objetos e cenários | ImageNet (validação) | 50 |
   | não cachorro | pessoas | Pascal VOC | 50 |

2. **Triagem com YOLO** (repita até aparecer "Triagem concluída"):
   - `python triagem.py` roda o YOLO e gera o relatório com os casos suspeitos e uma amostra das imagens não sinalizadas.
   - Abra `triagem/relatorio.html` para ver as fotos. Em `triagem/relatorio.csv`, escreva na coluna `decisao` de cada linha: `manter`, `corrigir` (o rótulo), `substituir` (mesma categoria e mesmo dataset) ou `revisar` (fica pendente). Nas linhas da amostra, `manter` quer dizer "conferi e está certa".
   - `python triagem.py --aplicar` aplica as decisões.
   - Se houve substituição, rode `python triagem.py` de novo: as imagens novas também passam pela triagem.

   A triagem termina quando não há imagens sem triagem nem casos pendentes, a distribuição planejada está preservada e a amostra foi conferida. Não é preciso zerar os alertas do YOLO: uma pessoa ao lado de um gato não é erro, e um lobo detectado como "dog" continua não cachorro.

3. **Montar o bairro**: `python montar_bairro.py`
   Extrai os 1.280 números de cada imagem. Na primeira vez, o split 80/20 (estratificado por classe + dataset de origem, `random_state=42`) sorteia as 80 do teste fixo e grava a lista em `base_imagens/teste_fixo.txt`; nas próximas, o código só lê essa lista e avisa se uma imagem do teste foi removida ou renomeada. Não apague esse arquivo: a prova mudaria, e as rodadas deixariam de ser comparáveis.

4. **Avaliar a rodada 1**: `python avaliar.py --rodada 1 --mudanca "ponto de partida: bairro com 320 (160 + 160)"`
   Mostra, para os dois modelos, a matriz, a acurácia, a meta e a área cinzenta. Registra a rodada 1 em `resultados_fuzzyknn/rodadas_fuzzyknn.csv` e em `resultados_svm/rodadas_svm.csv` e copia as imagens de FP e FN de cada modelo para `resultados_fuzzyknn/rodada_1_fuzzyknn/` e `resultados_svm/rodada_1_svm/`. O SVM é treinado no bairro nessa hora; nada dele fica guardado em arquivo.

## 4. Parte 2 · aplicação

`streamlit run app.py`. Na barra lateral, escolha entre "Home · Fuzzy k-NN", "Home · SVM" e "Visão admin":

- **Home · Fuzzy k-NN**: abre com um texto curto que separa **pertinência** (o número de 0 a 1 que o Fuzzy k-NN calcula) de **classe** (a decisão tirada dele), e explica por que o projeto segue a regra da maior pertinência em vez de um alfa-corte escolhido à mão. Envie uma imagem e veja a classe com o grau. Abaixo vem sempre a caixa da faixa em que a pertinência caiu — verde "🐶 Imagem na área é cachorro", amarela "⚠ Imagem na área cinzenta" ou vermelha "❌ Imagem na área não é cachorro" — com a regra da faixa, as duas pertinências e a contagem das vizinhas. Depois da caixa, também sempre, as fotos das 5 vizinhas com a classe de cada uma. O bloco "Imagens anteriores" guarda as imagens enviadas só durante a sessão: somem ao fechar ou recarregar a página e não entram na base. Cada home tem o seu.
- **Home · SVM**: o mesmo formato, com o SVM. A mensagem é `<classe> · probabilidade <valor>%`; a caixa da faixa tem as mesmas cores, com as faixas em porcentagem e a linha "probabilidade da imagem:"; a caixa mostra a pontuação do SVM, quanto as 5 imagens somam e quanto vem do resto; abaixo dela vêm as 5 imagens do bairro que mais puxaram para a classe vencedora, com a classe de cada uma. Os textos explicam probabilidade × classe; "É cachorro" e "Não é cachorro" são os mesmos da outra home. Enquanto o SVM treina no bairro, aparece um aviso de espera.
- **Visão admin**, com três abas: **Fuzzy k-NN** e **SVM** (a tabela das rodadas do modelo e, para a rodada escolhida, a matriz de confusão, a acurácia, a meta e a galeria da área cinzenta nas 4 situações, cada imagem com as suas 5 vizinhas ou as 5 que mais pesaram) e **Comparativo** (a mesma rodada dos dois lado a lado; só aparecem as rodadas que existem nas duas tabelas).

Cada home avisa quando o modelo dela ainda não foi avaliado com o bairro em uso: ela compara a assinatura do bairro com a da última rodada registrada do modelo.

Meta no teste fixo, para os dois modelos: acurácia ≥ 90% (72 das 80), com FN ≤ 6 e FP ≤ 6.

## 5. Rodadas de melhoria

Uma mudança por rodada (dados, extrator ou código), sempre testada nas mesmas 80 imagens, nos dois modelos. O `avaliar.py` avalia os dois e grava uma linha em cada tabela, com o mesmo número e o mesmo "o que mudou"; o modelo que não mudou precisa repetir os números da rodada anterior, senão o comando para e mostra a diferença. Numa rodada de dados ou de extrator, o bairro muda e os dois modelos contam como mudados. Numa rodada de código que mexeu num modelo só, diga qual com `--mudou` (por exemplo, `--mudou svm`): o outro precisa repetir os números. O bairro só é montado de novo quando mudam os dados ou o extrator, e o SVM é treinado de novo junto. O k, o C e o gamma ficam fixos na V2.

| rodada | o que muda | comandos |
|---|---|---|
| 2 · exemplo | + 80 imagens (20 de cada dataset em cada classe), após a triagem | `python montar_base.py --acrescentar 20` → triagem (passo 2) → `python montar_bairro.py` → `python avaliar.py --rodada 2 --mudanca "+ 80 imagens (20 de cada dataset em cada classe)"` |
| 3 · exemplo | troca do extrator: EfficientNetB0 → MobileNetV2 (com o `preprocess_input` dele) | `python montar_bairro.py --extrator mobilenetv2` → `python avaliar.py --rodada 3 --mudanca "MobileNetV2"` |
| 4 · exemplo | ajuste de código | mude o código → `python montar_bairro.py` (só se a mudança mexer na extração ou nos dados) → `python avaliar.py --rodada 4 --mudanca "o que mudou" --mudou svm` (o modelo cujo código mudou; os dois: `--mudou fuzzyknn svm`) |

- Para direcionar o acréscimo ao que o modelo erra, use `--categorias` com códigos do ImageNet ou nomes do Pascal VOC: `python montar_base.py --acrescentar 20 --categorias n02114367,n02114548` sorteia os não cachorros do ImageNet só entre lobos; as outras partes seguem o sorteio normal, mantendo a simetria.
- O app usa sempre o extrator com que o bairro foi montado. Para voltar ao EfficientNetB0 depois da rodada 3, rode `python montar_bairro.py` sem `--extrator`.
- As rodadas da V2 começam do zero: a rodada 1 é o ponto de partida dos dois modelos. Registrar de novo uma rodada que já existe só acontece com pedido explícito (`--substituir`), e nas duas tabelas.

## Onde cada requisito está

| requisito | onde |
|---|---|
| RF1 · upload | `app.py`, em cada home: a imagem é analisada sozinha, sem remontar o bairro |
| RF2 · mesmo EfficientNetB0 | `extrator.py`: só redimensiona para 224 × 224, sem normalizar à mão; o app usa o extrator gravado com o bairro |
| RF3 · pertinências | `fuzzyknn.py` (`fuzzy_knn`): 5 vizinhas, peso 1/d², as duas pertinências somam 1 |
| RF4 · decisão | `regua.py` (`decidir`): maior grau por um `if` comum, com o grau em duas casas; empate em 0,50 → não cachorro; a mesma regra para os dois modelos |
| RF5 · resultado | `fuzzyknn.py` (`mensagem`): classe + pertinência da classe vencedora; `svm.py`: classe + probabilidade da classe vencedora, em porcentagem |
| RF6 · área cinzenta | `regua.py` (`na_area_cinzenta`) e o bloco amarelo do `app.py`: o aviso de que a decisão foi apertada; ele não muda a classe |
| RF7 · imagens anteriores | `app.py`, em `st.session_state`, com nomes separados por home |
| RF8 · avaliar no teste fixo | `avaliar.py` e `avaliacao.py`; abas Fuzzy k-NN e SVM da visão admin |
| RF9 · rodadas | `rodadas.py`; `resultados_fuzzyknn/rodadas_fuzzyknn.csv` e `resultados_svm/rodadas_svm.csv`; visão admin |
| RF10 · faixa do grau | `app.py` (`FAIXAS`, `faixa_do_resultado`, `texto_caixa`): toda imagem recebe a caixa da sua faixa — verde acima de 0,60, amarela de 0,40 a 0,60, vermelha abaixo de 0,40 — com a linha "pertinência da imagem:", a vencedora primeiro, e as fotos das 5 vizinhas abaixo; na home do SVM, as mesmas faixas em porcentagem, com as 5 imagens que mais pesaram |
| RF11 · probabilidade do SVM | `svm.py`: `StandardScaler` + `CalibratedClassifierCV(SVC(), ensemble=False)`, treinado no bairro; a coluna 1 do `predict_proba` é a de cachorro, e o `predict` não é usado |
| RF12 · imagens que mais pesaram | `svm.py` e a home do SVM: as 5 imagens do bairro que mais puxaram para a classe vencedora, com quanto elas somam na pontuação e quanto vem do resto |
| RF13 · comparativo | aba Comparativo da visão admin: a mesma rodada dos dois, lado a lado |

## Escolhas que o PRD não fixa

Estão em `config.py` e `codigos_imagenet.py` e podem ser trocadas.

1. **Sorteio das fotos**: em rodízio entre as categorias, com semente fixa (42). Os 50 outros animais do Pascal VOC saem 10 de cada (pássaro, gato, vaca, cavalo, ovelha), e os 100 cachorros do ImageNet saem de raças diferentes.
2. **Categorias do ImageNet**: o PRD fixa o hamster e os canídeos selvagens (lobos, coiote, dingo, mabeco e raposas). As outras espécies e os objetos e cenários (25 categorias em cada grupo, 2 fotos por categoria) estão listados em `codigos_imagenet.py`.
3. **Regra de inclusão no Pascal VOC**: já no sorteio, a foto de cachorro precisa ter o cão como maior objeto anotado (e não marcado como difícil); a de não cachorro não pode ter nenhum cachorro anotado. A "categoria original" de uma foto do Pascal VOC é o maior objeto anotado.
4. **Triagem**: detecções do YOLO abaixo de 0,25 de confiança são ignoradas (padrão da Ultralytics); cachorro abaixo de 0,50 conta como detecção incerta. A amostra das não sinalizadas é 5% de cada lote triado.
5. **Imagem substituída** não é apagada: vai para `triagem/descartadas/` e não volta a ser sorteada.
6. **Acréscimo**: N fotos de cada dataset em cada classe; em não cachorro, divididas igualmente entre as duas categorias reais de cada dataset (ou só nas categorias de `--categorias`).
7. **Tabelas de rodadas**: uma por modelo. Além das colunas do PRD (o que mudou, VP, FN, FP, VN, acurácia, área cinzenta, meta e assinatura do bairro), as duas guardam o número da rodada, o extrator, o tamanho do bairro e a data; a do Fuzzy k-NN guarda o k, e a do SVM, o C e o gamma.
8. **Imagem**: redimensionamento bilinear para 224 × 224, a mesma função para bairro, teste e upload, com correção da orientação de foto de celular. O upload aceita jpg, png e webp; foto HEIC do iPhone precisa ser exportada como JPG.
9. **Visão admin** sem senha, escolhida na barra lateral.
10. **`--ignorar-triagem`** em `montar_bairro.py` monta o bairro mesmo com a triagem incompleta. Existe só como saída de emergência; o PRD pede a triagem antes da extração.

## Testes

`python -m pytest tests`

Usam números inventados: não precisam das imagens, do TensorFlow nem do YOLO. Cobrem o `fuzzy_knn` (incluindo o exemplo de 0,65 do card Grau fuzzy), a decisão, a área cinzenta, a mensagem, o split estratificado, o teste fixo gravado, a leitura dos XML, o sorteio e a matriz.

Na V2, também cobrem a régua comum (as bordas de sempre, sem mudar nenhum valor esperado), o SVM (fit, classe, faixa, a porcentagem nas bordas e a conta das imagens que mais pesaram), as travas das rodadas, os nomes do `config.py`, o `.gitignore` das pastas de FP e FN e a memória separada das duas homes.
