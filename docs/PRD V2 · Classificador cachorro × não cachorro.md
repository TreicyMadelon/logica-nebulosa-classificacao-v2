# PRD V2 · Classificador cachorro × não cachorro

Oct 22, 2026 · @treicy

O sistema recebe uma imagem e responde se é cachorro ou não, com o grau da resposta, em dois modelos: o Fuzzy k-NN e o SVM. O EfficientNetB0 descreve a imagem em 1.280 números; o Fuzzy k-NN (k = 5) compara esses números com um bairro de 320 imagens rotuladas, e o SVM é treinado nesse bairro. O Fuzzy k-NN mostra “\<classe> · pertinência \<valor>”, de 0 a 1, e o SVM mostra “\<classe> · probabilidade \<valor>%”. Nos dois, a classe é a de maior grau, e um grau de cachorro entre 0,40 e 0,60 acende o aviso de área cinzenta. Os dois são comparados nas mesmas 80 imagens.

## Contexto e objetivo

Trabalho final individual da disciplina Lógica Nebulosa (prof. Mardonio), na pós em IA Generativa: um classificador binário que mostra, além da classe, o grau fuzzy da resposta. A V2 mantém o plano A (o Fuzzy k-NN, sem corte e sem inferência fuzzy) e acrescenta um SVM como segundo modelo, porque o professor pediu, na aula de 17/10, pelo menos 2 modelos para comparar as métricas. A V2 é uma cópia do projeto; o original não muda.

O objetivo é provar três coisas: que o sistema separa cachorro de não cachorro em imagens que nunca viu, que o grau mostra onde ele está em dúvida e qual dos dois modelos acerta mais nas mesmas 80 imagens.

**Quem usa**

- **Pessoa que sobe uma imagem:** escolhe a home de um dos modelos e recebe a classe e o grau na hora, com a caixa colorida da faixa e as 5 imagens do bairro que explicam o resultado (as vizinhas, no Fuzzy k-NN; as que mais pesaram, no SVM); na área cinzenta, a caixa amarela avisa a dúvida.
- **A autora, na apresentação:** mostra os resultados no teste fixo na visão admin, com uma aba por modelo e a aba Comparativo (matrizes, rodadas e galerias da área cinzenta), e faz a demonstração ao vivo nas duas homes.

## Requisitos do professor

São quatro requisitos para o classificador e uma tarefa extra separada, que fica fora da V2. A V2 atende os quatro com os dois modelos: a pertinência no Fuzzy k-NN e a probabilidade no SVM.

| Requisito | O que o professor pediu | Como o projeto atende |
| --- | --- | --- |
| Classificação binária | “um projeto de classificação binária, que a gente consiga colocar essa ideia aí da lógica fuzzy” | Cachorro × não cachorro (rótulos 1 e 0); nos dois modelos, a classe é a de maior grau |
| Mostrar o grau (%) | “um resultado com o grau, o percentual”; é nesse percentual que está o nebuloso | Fuzzy k-NN: a pertinência da classe vencedora, de 0 a 1 (“\<classe> · pertinência \<valor>”). SVM: a probabilidade da classe vencedora, em porcentagem (“\<classe> · probabilidade \<valor>%”) |
| Casos no limite (aula 3) | os casos “lá no limite da incerteza, tá saindo entre 51 e 60” | Área cinzenta: grau de cachorro de 0,40 a 0,60 (de 40% a 60% no SVM), com aviso na tela e galeria no teste fixo, nos dois modelos |
| Comparar modelos (aula de 17/10) | “escolham pelo menos 2 para comparar as métricas” | SVM como segundo modelo, nas mesmas 80 imagens, e a aba Comparativo: a mesma rodada dos dois, lado a lado |

## Escopo

A V2 cobre da montagem da base à comparação dos dois modelos no teste fixo; tudo o que depende de inferência fuzzy, corte ou fotos de usuários fica de fora.

**Dentro do escopo**

- Base única de 400 imagens (ImageNet validação + Pascal VOC), conferida por triagem com YOLO.
- Extração de características com o EfficientNetB0 pré-treinado, a mesma para os dois modelos.
- Classificação pelo Fuzzy k-NN (k = 5, peso 1/d²) e decisão pela maior pertinência.
- Segundo modelo: SVM do scikit-learn (padronização + SVC calibrado), treinado no mesmo bairro; o grau é a probabilidade, e a classe é a de maior probabilidade, com duas casas.
- Área cinzenta (grau de cachorro de 0,40 a 0,60) na tela e na avaliação, nos dois modelos.
- Aplicação em Streamlit com duas homes (Fuzzy k-NN e SVM) e a visão admin com três abas (Fuzzy k-NN, SVM e Comparativo).
- Em cada home: upload, resultado com a caixa colorida da faixa (verde, amarela ou vermelha) e as 5 imagens do bairro que explicam o resultado, texto que explica o grau × a classe e bloco “Imagens anteriores”.
- Avaliação dos dois modelos no teste fixo de 80 imagens, rodadas de melhoria registradas por modelo e comparativo da mesma rodada.

**Fora do escopo**

- Fotos enviadas pelos usuários na base: ficam só na sessão.
- Treinar ou ajustar os pesos de uma rede neural: o EfficientNetB0 só extrai os números.
- Ajustar o k, o C e o gamma: ficam fixos na V2 (k = 5; C e gamma nos padrões do scikit-learn).
- Terceiro modelo (Random Forest), base de validação, precisão/revocação/F1, aumento de dados, incremental e pseudo-labeling: ver Evoluções futuras.
- A tarefa extra de reforço, separada do classificador.

## Arquitetura

&#91;embedded content: arquitetura V2 · Parte 1, Parte 2 e avaliação dos dois modelos\]

A Parte 1 roda uma vez e separa o bairro (320 imagens) do teste fixo (80); o SVM é treinado nesse mesmo bairro sempre que precisa, sem arquivo guardado. A Parte 2 roda a cada upload, na home de cada modelo. Na avaliação, as 80 passam pelos dois modelos num comando só; a melhoria só remonta o bairro, e refaz o fit do SVM, quando mudam os dados ou o extrator.

## Dados

A base é uma só: 400 imagens, 200 por classe e 100 de cada dataset de origem em cada classe. A mistura simétrica evita que o estilo de uma base vire pista para separar as classes (viés de dataset, Torralba e Efros, 2011). Os dois modelos usam a mesma base, o mesmo bairro e o mesmo teste fixo.

| Classe | Categoria real | Dataset de origem | Imagens |
| --- | --- | --- | --- |
| Cachorro | Raças domésticas | ImageNet (validação) | 100 |
| Cachorro | Cachorro | Pascal VOC | 100 |
| Não cachorro | Outros animais (com hamster e canídeos selvagens) | ImageNet (validação) | 50 |
| Não cachorro | Outros animais (pássaro, gato, vaca, cavalo, ovelha) | Pascal VOC | 50 |
| Não cachorro | Objetos e cenários | ImageNet (validação) | 50 |
| Não cachorro | Pessoas diversas | Pascal VOC | 50 |

1. **Canídeos selvagens:** lobo, coiote, raposa, dingo e mabeco são não cachorro (bons candidatos à área cinzenta).
2. **ImageNet só do conjunto de validação:** o EfficientNetB0 não viu essas fotos no treino. As duas bases vêm dos sites oficiais (o ImageNet exige cadastro), sem o Hugging Face; as anotações são XML nas duas.
3. **Pastas e metadados:** `base_imagens/cachorro/` e `base_imagens/nao_cachorro/`, mais `base_imagens/metadados.csv` (arquivo, classe, dataset de origem, categoria original).
4. **Regra de inclusão:** em cachorro, o cão claramente visível e principal; em não cachorro, nenhum cachorro na imagem.
5. **Triagem com YOLO, antes da extração:** cada caso suspeito recebe manter, corrigir o rótulo, substituir (mesma categoria e mesmo dataset) ou revisar. Termina quando não há pendências, a distribuição está preservada e uma amostra não sinalizada foi conferida.
6. **Split 80/20:** estratificado por classe + dataset de origem, `random_state=42`. Ficam 320 imagens no bairro (160 + 160) e 80 no teste fixo (40 + 40; 20 de cada dataset por classe).
7. **Teste fixo gravado:** na primeira execução, a lista das 80 vai para `base_imagens/teste_fixo.txt`; depois, o código lê essa lista. Imagem do teste removida ou renomeada → o código avisa.
8. **O que nunca entra no bairro:** as imagens do teste fixo e as fotos enviadas pelos usuários.
9. **Acréscimos numa melhoria:** vêm do ImageNet e do Pascal VOC, na mesma quantidade por classe e por dataset, e passam pela triagem.

## Requisitos funcionais

A aplicação tem uma home por modelo: cada uma classifica a imagem enviada sem remontar a base e sempre mostra a classe com o grau; a área cinzenta acrescenta um aviso, mas não muda a classe.

**Regra de arredondamento:** a classe e a faixa saem do grau de cachorro (a pertinência no Fuzzy k-NN, a probabilidade no SVM) com duas casas decimais, pelo arredondamento convencional (o meio sobe). O grau de não cachorro é 1 menos esse valor. A home do Fuzzy k-NN mostra esse número com duas casas; a do SVM mostra o mesmo número vezes 100, em porcentagem sem casas decimais, sem arredondar de novo. A régua é uma só, para os dois modelos.

| ID | Requisito | Critério de aceite |
| --- | --- | --- |
| RF1 | Receber uma imagem por upload | A imagem é analisada sozinha, sem remontar o bairro; cada home tem o seu campo de envio |
| RF2 | Extrair as características com o mesmo EfficientNetB0 | Redimensiona para 224 × 224, sem normalizar à mão; sai um vetor de 1.280 números |
| RF3 | Calcular as pertinências com o Fuzzy k-NN | 5 vizinhas do bairro, peso 1/d²; pertinência de cada classe = pesos da classe ÷ soma dos pesos; as duas somam 1 |
| RF4 | Decidir a classe | Maior grau (pertinência ou probabilidade), por um `if` comum, sem corte, com o grau em duas casas; empate em 0,50 → não cachorro; a mesma regra nos dois modelos |
| RF5 | Mostrar o resultado | Sempre a classe + o grau da classe vencedora. Fuzzy k-NN: a pertinência, de 0 a 1 com duas casas (“\<classe> · pertinência \<valor>”). SVM: a probabilidade, em porcentagem sem casas decimais (“\<classe> · probabilidade \<valor>%”) |
| RF6 | Avisar a área cinzenta | Grau de cachorro de 0,40 a 0,60 (de 40% a 60% no SVM) → bloco amarelo “⚠ Imagem na área cinzenta”; o aviso não muda a classe |
| RF7 | Listar as imagens anteriores | Bloco “Imagens anteriores” em `st.session_state`: clicar revê o resultado; some ao fechar ou recarregar; não entra na base; um bloco por home, com nomes separados na memória da sessão |
| RF8 | Avaliar no teste fixo | Matriz de confusão, acurácia e galeria da área cinzenta nas 80 imagens, para os dois modelos, num comando só |
| RF9 | Registrar as rodadas | Uma tabela por modelo (`rodadas_fuzzyknn.csv` e `rodadas_svm.csv`), com o mesmo número de rodada e o mesmo “o que mudou” nas duas: VP, FN, FP, VN, acurácia, área cinzenta, meta e a assinatura do bairro |
| RF10 | Mostrar a faixa do grau | Toda imagem recebe um bloco colorido pela faixa: acima de 0,60, verde “🐶 Imagem na área é cachorro”; de 0,40 a 0,60, amarelo “⚠ Imagem na área cinzenta”; abaixo de 0,40, vermelho “❌ Imagem na área não é cachorro”. Os três trazem a linha “pertinência da imagem:” com as duas pertinências, a vencedora primeiro, e as 5 vizinhas do bairro (fotos e classes), da mais próxima para a mais distante. Na home do SVM, as mesmas faixas em porcentagem (acima de 60%, de 40% a 60%, abaixo de 40%), a linha “probabilidade da imagem:” com os dois valores em porcentagem e as 5 imagens do bairro que mais pesaram (RF12). |
| RF11 | Calcular a probabilidade com o SVM | Pipeline `make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), ensemble=False))`, treinado no bairro, com C = 1 e gamma “scale” fixos; a probabilidade de cachorro é a coluna 1 do `predict_proba`; o `predict` não é usado |
| RF12 | Mostrar as imagens que mais pesaram (SVM) | As 5 imagens do bairro que mais pesaram na conta do SVM, da que mais pesou para a que menos pesou, com o rótulo de cada uma e o aviso de que os outros vetores de suporte também contam; um teste confere que a soma das contribuições bate com a pontuação do SVM |
| RF13 | Comparar os dois modelos | Aba Comparativo da visão admin: só a mesma rodada nos dois; lado a lado acurácia, VP, FN, FP, VN, meta e área cinzenta, com a diferença em número de imagens; imagem a imagem: os dois acertaram, só um acertou, os dois erraram; ganha quem acerta mais nas 80, e no empate uma nota mostra a área cinzenta de cada um |

Tela de uma imagem na área cinzenta (um gato parecido com cachorro), na home do Fuzzy k-NN:

```
<classe> · pertinência <valor>

⚠ Imagem na área cinzenta
   Toda imagem enviada pelo usuário que cair entre 0,40 e 0,60
   pertinência da imagem: <classe vencedora> <valor> · <outra classe> <1 − valor>
   parecida com: <n> <classe vencedora> e <m> <outra classe> entre as 5 imagens mais próximas
```

Na home do SVM, o mesmo caso:

```
<classe> · probabilidade <valor>%

⚠ Imagem na área cinzenta
   Toda imagem enviada pelo usuário que cair entre 40% e 60%
   probabilidade da imagem: <classe vencedora> <valor>% · <outra classe> <100 − valor>%
   entre as 5 imagens que mais pesaram: <n> <classe vencedora> e <m> <outra classe>
   (os outros vetores de suporte também contam)
```

Na home do SVM, os textos que falam do Fuzzy k-NN, da pertinência ou das vizinhas ganham uma versão do SVM: o título com o nome do modelo, a legenda, a explicação, as linhas da caixa, o título das fotos e a barra lateral (C, gamma e a faixa em porcentagem). “É cachorro” e “Não é cachorro” ficam iguais nas duas homes.

## Avaliação e melhoria

Os dois modelos são avaliados só nas 80 imagens do teste fixo, que nunca entram no bairro: matriz de confusão, acurácia ((VP + VN) ÷ total) e a galeria da área cinzenta de cada um. O equilíbrio entre FP e FN orienta a melhoria, não uma régua.

Meta de acurácia no teste fixo: pelo menos 90% (72 das 80 imagens), com no máximo 6 erros de cada tipo na matriz (FN ≤ 6 e FP ≤ 6). A meta é a mesma para os dois modelos.

Por que esses números: numa base 40 + 40, quem chuta sempre a mesma classe acerta 50%. O EfficientNetB0 já conhece cerca de 120 raças de cachorro, então 90% é alcançável, mas a base tem casos difíceis de propósito (canídeos selvagens e cenas do Pascal VOC). O limite por tipo de erro garante que cada classe acerte pelo menos 34 de 40, para uma não esconder a outra.

A galeria separa as imagens de teste que caíram na faixa em 4 situações; os erros dela também contam na matriz. No SVM, a mesma galeria, com a probabilidade em porcentagem (51% a 60% e 40% a 50%) e as imagens que mais pesaram no lugar das vizinhas.

| Rótulo real | Pertinência a cachorro | O modelo disse | Resultado |
| --- | --- | --- | --- |
| Cachorro | 0,51 a 0,60 | é cachorro | acertou (VP) |
| Cachorro | 0,40 a 0,50 | não é cachorro | errou (FN) |
| Não cachorro | 0,40 a 0,50 | não é cachorro | acertou (VN) |
| Não cachorro | 0,51 a 0,60 | é cachorro | errou (FP) |

**Comparativo dos dois modelos**

- Só a mesma rodada: rodada N do Fuzzy k-NN × rodada N do SVM, com as mesmas 80 e a mesma assinatura do bairro; o comparativo não deixa escolher números diferentes.
- Lado a lado: acurácia, VP, FN, FP, VN, meta e área cinzenta, com a diferença em número de imagens. Imagem a imagem: os dois acertaram, só o Fuzzy k-NN, só o SVM, os dois erraram, com a galeria das que discordam.
- Quem foi melhor: quem acerta mais imagens nas mesmas 80. O projeto não define qual erro pesa mais, então não há desempate por FP ou FN.
- Empate: a tela diz que houve empate e traz uma nota com qual modelo teve menos imagens na área cinzenta, com os dois números. A nota não muda o resultado; ela avisa que a área cinzenta mostra dúvida, não acerto, e que a faixa lê pertinência num modelo e probabilidade no outro. Se a área cinzenta também empatar, a nota diz isso.

**Regras das rodadas de melhoria**

- Uma mudança por rodada (dados, extrator ou código), sempre testada nas mesmas 80 imagens, nos dois modelos.
- Um comando avalia os dois modelos e grava uma linha em cada tabela, com o mesmo número de rodada e o mesmo “o que mudou”. As rodadas da V2 começam do zero.
- O modelo que não mudou repete os números da rodada anterior; se não repetir, o comando para e mostra a diferença.
- O k (5) e a forma do peso do Fuzzy k-NN, e o C (1) e o gamma (“scale”) do SVM, ficam fixos durante toda a V2: nenhum modelo é ajustado olhando as 80.
- A base é montada de novo, e o fit do SVM refeito, só quando mudam os dados ou o extrator. Trocar a versão do scikit-learn também vira uma rodada dos dois.
- Imagens acrescentadas seguem a regra de acréscimos (Dados, item 9).
- Conferência da rodada 1: o Fuzzy k-NN repete os números da rodada 2 do projeto original, que só é lido.

| Rodada | O que muda |
| --- | --- |
| 1 · ponto de partida | Bairro com 320 imagens (160 + 160), nos dois modelos |
| 2 · exemplo | + 80 imagens (20 de cada dataset em cada classe), após a triagem |
| 3 · exemplo | Troca do extrator: EfficientNetB0 → MobileNetV2 (com o `preprocess_input` dele); refaz os dois modelos |
| 4 · exemplo | Ajuste de código |

Na apresentação, primeiro os resultados no teste fixo (as duas matrizes, as rodadas, o comparativo e as galerias); depois, a demonstração ao vivo nas duas homes, com uma imagem que cai na área cinzenta.

## Requisitos técnicos

Tudo roda localmente em Python, num MacBook Neo (A18 Pro, 8 GB), com modelos gratuitos.

| Componente | Escolha | Observação |
| --- | --- | --- |
| Extrator | EfficientNetB0 (Keras/TensorFlow), `include_top=False`, `pooling="avg"`, pesos do ImageNet | \~5,3 milhões de parâmetros; pesos de \~16 MB baixados na primeira vez; espera imagens cruas (0 a 255); os mesmos números servem aos dois modelos |
| Classificador 1 · Fuzzy k-NN | Função `fuzzy_knn` própria, em NumPy (\~10 linhas), no `fuzzyknn.py` | Devolve as pertinências e os índices das 5 vizinhas |
| Classificador 2 · SVM | scikit-learn: `make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), ensemble=False))`, no `svm.py` | C = 1 e gamma “scale” (os padrões), fixos; sem `class_weight` (o bairro é 160 + 160) e sem validação cruzada para escolher C e gamma; treinado no bairro quando precisa, sem arquivo guardado e sem `random_state` (o fit não sorteia) |
| Régua comum | `regua.py`: duas casas (`formatar_grau`, `grau_na_tela`), decisão (`decidir`) e faixa (`na_area_cinzenta`, `limites_area_cinzenta`) | Definida uma vez só, para os dois modelos; os testes de borda de hoje passam sem mudar nenhum valor esperado |
| Triagem | YOLOv8n (Ultralytics), treinado no COCO | Só na preparação da base; traz o PyTorch; licença AGPL-3.0, gratuita para uso acadêmico |
| Split e métricas | scikit-learn: `train_test_split`, `confusion_matrix`, `accuracy_score` | No Fuzzy k-NN, o scikit-learn só divide a base e conta os acertos; no SVM, ele também classifica |
| Configurações | `config.py` em três partes: comum, só do Fuzzy k-NN (`FUZZYKNN_K`, `FUZZYKNN_PASTA_RESULTADOS`) e só do SVM (`SVM_C`, `SVM_GAMMA`, `SVM_PASTA_RESULTADOS`) | Nenhuma configuração sobrescreve outra: um teste falha se um nome aparecer duas vezes |
| Resultados e rodadas | `resultados_fuzzyknn/` e `resultados_svm/`, cada uma com `rodadas_<modelo>.csv` e `rodada_N_<modelo>/` (`resultados_<modelo>.csv`, `FP_<modelo>/`, `FN_<modelo>/`) | Caminhos de uma função só, `caminhos(modelo)`, no `rodadas.py`; 13 proteções contra salvar errado, cada uma com teste |
| Interface | Streamlit: duas homes (Fuzzy k-NN e SVM) e a visão admin com três abas (Fuzzy k-NN, SVM e Comparativo) | Estado da sessão em `st.session_state`, com nomes separados por home |
| Reprodutibilidade | `random_state=42`, teste fixo em `base_imagens/teste_fixo.txt`, leitura na ordem dos metadados, versão do scikit-learn fixa no `requirements.txt` e assinatura do bairro em cada rodada | Rodar de novo dá a mesma matriz, também num clone: a base é remontada com as imagens listadas no metadados.csv; o SVM treina com o bairro na ordem do arquivo, e o código nunca reordena as linhas |
| Repositório | Cópia do projeto ligada ao repositório novo logica-nebulosa-classificacao-v2, com histórico novo | O original não muda; commits só com a identidade da autora |

**Git: o que sobe e o que não sobe**

- Sobe: código, testes, README, requirements e os CSVs (metadados, teste fixo, distribuição, triagem, tabelas de rodadas e resultados imagem por imagem dos dois modelos).
- Não sobe: as imagens (base, descartadas e cópias de FP e FN dos dois modelos), as bases de origem, `vetores/`, `.venv`, o relatório HTML da triagem e os pesos do YOLO.
- As bases de origem vão na cópia, mas continuam fora do Git: o `.gitignore` já tem a linha `bases_origem/`.
- O `.gitignore` da V2 ganha `resultados_*/rodada_*/FP_*/` e `resultados_*/rodada_*/FN_*/`, com as linhas antigas mantidas; um teste confere que as pastas de imagens ficam fora e os CSVs sobem.

## Riscos e limitações

A principal limitação é não ter base de validação: as rodadas de dados são avaliadas no próprio teste fixo, como no split 80/20 da aula; com k, C e gamma fixos, nenhum modelo é ajustado olhando as 80.

| Risco ou limitação | Efeito | Como o projeto trata |
| --- | --- | --- |
| Sem base de validação | Escolhendo imagens novas olhando os erros das 80, a autora pode “decorar” a prova | k, C e gamma fixos na V2, então o risco fica só nas rodadas de dados; registrado como limitação; base de validação como evolução |
| Viés de dataset e desvio de domínio | Errar mais com fotos reais, diferentes das da base | Base híbrida simétrica (ImageNet + Pascal VOC) |
| O YOLO também erra | Imagem errada passa, ou imagem boa é sinalizada | Decisão humana por caso + amostra das não sinalizadas |
| Pertinência × probabilidade | Ler a probabilidade do SVM como pertinência, ou o contrário | Pertinência = o quanto pertence a um conjunto vago; probabilidade = a chance de um evento. Cada home usa a palavra do seu modelo, e os dois são comparados pelas métricas |
| Pixels normalizados duas vezes | Bug silencioso que só aparece na matriz | Imagens cruas: o EfficientNetB0 já faz a preparação |
| Classe errada nos metadados | O sistema aprende o contrário | Triagem e conferência dos rótulos |
| Diferença pequena entre os modelos | Com 80 imagens, uma diferença de poucas imagens pode ser acaso | Ganha quem acerta mais (a regra não muda), e o comparativo mostra a diferença em número de imagens. Uma diferença de menos de 6 imagens nunca passa no teste exato de McNemar a 5%; mesmo 6 só passa se as 6 discordâncias forem a favor do mesmo modelo |
| Salvar a rodada no lugar errado | Misturar as rodadas dos dois modelos ou sobrescrever uma rodada | 13 proteções no código, cada uma com teste: caminhos de uma função só, travas de pasta, tabela e rodada, as mesmas 80 e a assinatura do bairro |
| Ordem do bairro | Outra ordem muda um pouco as probabilidades do SVM, sem nenhuma imagem nova | O código nunca reordena o bairro; a assinatura do bairro acusa uma ordem diferente |
| Versão do scikit-learn | No 1.9, o `SVC(probability=True)` está descontinuado e sai no 1.11; outra versão pode mudar os números | `CalibratedClassifierCV(SVC(), ensemble=False)`, a troca que o próprio scikit-learn indica, e a versão fixa no `requirements.txt` |

Não aparecem nas transcrições das aulas e devem ser apresentados como escolhas do projeto: modelo pré-treinado e transferência de aprendizado, Fuzzy k-NN (o KNN comum foi visto), a calibração da probabilidade do SVM (`CalibratedClassifierCV`; o SVM e o `predict_proba` foram citados na aula de 17/10), YOLO e `stratify`.

## Evoluções futuras

Nada disto entra na V2; cada item exige mudança no código ou uma regra nova.

- **Base de validação:** uma terceira parte da base só para os ajustes das rodadas, deixando o teste fixo para a avaliação final; com ela, dá para ajustar o k, o C e o gamma sem olhar as 80.
- **Precisão, revocação e F1:** calculadas da mesma matriz de confusão, para os dois modelos (o professor destacou o F1 na aula de 17/10).
- **Terceiro modelo:** Random Forest, no mesmo bairro e nas mesmas 80.
- **Aumento de dados:** versões espelhadas, giradas ou recortadas, geradas na leitura e só para as imagens do bairro (nunca para as do teste).
- **Incremental:** acrescentar imagens novas ao bairro sem montá-lo de novo (os números entram em `X_treino_num` e os rótulos em `y_treino`); no SVM, o fit é refeito inteiro.
- **Pseudo-labeling:** só imagens com pertinência a cachorro ≥ 0,95 ou ≤ 0,05 virariam exemplo; a área cinzenta iria para revisão humana. Exigiria uma regra para as fotos dos usuários, que seriam um terceiro dataset de origem.

## Glossário

| Termo | Significado no projeto |
| --- | --- |
| Bairro (base de referência, base de treino) | As 320 imagens rotuladas, guardadas como números, onde o Fuzzy k-NN procura as vizinhas e o SVM faz o fit |
| Treinar | No Fuzzy k-NN, montar o bairro, sem pesos ajustados nem loss; no SVM, o fit: padronizar os números, traçar a fronteira e calibrar a probabilidade |
| Teste fixo | As 80 imagens separadas no split, sempre as mesmas e fora do bairro |
| Vizinhas | As 5 imagens do bairro mais próximas da imagem analisada |
| Pertinência | O quanto a imagem pertence a cada classe, de 0 a 1; as duas somam 1 |
| Classe vencedora | A classe de maior grau; a tela mostra o grau dela (a pertinência ou a probabilidade) |
| Área cinzenta | Pertinência a cachorro de 0,40 a 0,60 (pertinência da classe vencedora de 0,50 a 0,60, com duas casas); no SVM, a probabilidade de cachorro de 40% a 60% |
| Dataset de origem | De onde veio a imagem: ImageNet (validação) ou Pascal VOC |
| Rodada | Uma mudança (dados, extrator ou código) seguida de novo teste no teste fixo, com o mesmo número nos dois modelos |
| SVM (máquina de vetores de suporte) | O segundo modelo: com o bairro, traça uma fronteira entre as classes; a imagem nova cai de um lado, a uma distância dela |
| Vetores de suporte | As imagens do bairro que o SVM guarda para traçar a fronteira |
| Probabilidade | O grau do SVM (`predict_proba`): a chance de a imagem ser cachorro, de 0 a 1, mostrada em porcentagem; não é pertinência |
| Imagens que mais pesaram | Os 5 vetores de suporte que mais pesaram na conta do SVM para aquela imagem, para qualquer um dos lados |
| Assinatura do bairro | Um código curto, tirado dos números, dos rótulos e dos nomes das 320 imagens, na ordem do arquivo, e do extrator; muda se o bairro mudar |
| Comparativo | A aba da visão admin que põe a mesma rodada dos dois modelos lado a lado |

## Referências

- Transcrições das aulas 1 a 5 e da aula de 17/10, e anotações do quadro do professor Mardonio.
- Keller, J. M.; Gray, M. R.; Givens, J. A. *A Fuzzy K-Nearest Neighbor Algorithm*. IEEE Transactions on Systems, Man, and Cybernetics, 1985.
- Torralba, A.; Efros, A. A. *Unbiased Look at Dataset Bias*. CVPR, 2011.
- scikit-learn, [Support Vector Machines](https://scikit-learn.org/dev/modules/svm.html): “Support Vector Machine algorithms are not scale invariant, so it is highly recommended to scale your data”.
- Aviso do scikit-learn 1.9.1 ao usar `SVC(probability=True)`: o parâmetro foi descontinuado no 1.9 e sai no 1.11, e a troca indicada é `CalibratedClassifierCV(SVC(), ensemble=False)` (testado com números inventados).
