# Metodologia e resultados: visão geral

Com base nos dados disponibilizados pela Assembleia da República em
formato XML são criadas _dataframes_ (tabelas de duas dimensões) com
base no processamento e selecção de informação relativa aos padrões de
votação de cada partido (e/ou deputados não-inscritos).

São fundamentalmente feitas as seguintes análises:

1. **Vista geral das votações de cada partido**, visualizado através de um
   mapa térmico que compara os votos de todos eles.
2. **Matriz de distância euclidiana** entre todos os partidos e
   visualização de ** *clustering* hierárquico ** através de um dendograma.
3. **Identificação de grupos** (*clustering*) por DBSCAN e *Spectral
   Clustering*, com criação de matriz de afinidade
4. Redução das dimensões e visualização das distâncias e agrupamentos
   num espaço cartesiano a duas e três dimensões através de
   ** *Multidimensional Scaling* (MDS) **

A utilidade deste tipo de análise em ciência política é reconhecida
{cite}`figueiredofilhoClusterAnalysisPolitical2014` e tem sido
aplicada a vários registos de votações; a análise presente tem como
principal diferença o ser efectuado sobre as votações de partidos e
não, como é mais comum na bibliografia consultada, a deputados
individuais.

A informação é obtida a partir das listas publicadas de votações relativas a

- Actividades
- Iniciativas

Os dados utilizados são um subconjuntos dos disponibilizados, sendo
que qualquer erro ou omissão nos dados originais irá ter imediato
reflexo nos resultados das análises.



```{warning} NB
O processo de tratamento de dados não é indiferente para o
resultado final: *são feitas escolhas a vários níveis (desde a
selecção dos dados considerados importantes aos algoritmos escolhidos)
que têm impacto nos resultados, nem que seja por omissão*. Mais do que
evitá-lo (o que não seria possível), optámos por identificar de forma
clara as escolhas feitas e explicar as razões que levaram à sua
escolha: cada leitor poderá assim determinar a razoabilidade de cada
uma e, sobretudo, ensaiar novas formas que considere mais adequadas.*
```
