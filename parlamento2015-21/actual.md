# O que é novo, e o que se mantém

Este trabalho é uma actualização do anteriormente realizado, para um
período de tempo menor, e que introduziu uma abordagem específica para
a análise da proximidade, distância e suporte mútuo entre todos os
grupos parlamentares e deputados independentes presentes na Assembleia
da República.

O trabalho original descreve de forma bastante aprofundada a
metodologia utilizada e as escolhas que foram feitas, chegando ao
ponto de ser complexo de ler para quem deseja ter um acesso mais
directo aos dados finais. Esta actualização procede assim a algumas
modificações a esse nível:

* A utilização de **Jupyter Book** permite uma organização dos
  conteúdos de forma mais estruturada.
* Os **blocos de código estão escondidos**, podendo ser vistos se
  desejado.
* As explicações sobre os processos de tratamento de dados e
  fundamentação teórica das análises **foi eliminado**, reduzido ou
  movido para um apêndice ou nota de rodapé.

Para além destas alterações de forma, uma alteração de conteúdo: a
análise original cobria parte da 14ª legislatura (do início até cerca
de Fevereiro de 2021), com algumas actualizações posteriores a
cobrirem períodos mais alargados ou a incidirem sobre áreas
específicas. Este trabalho cobre todo o período da 13ª Legistatura em
2015 até Outubro de 2021, data na qual foi votada a não aprovação do
Orçamento de Estado de 2022: todo o período de 6 anos onde o Governo
minoritário do PS governou com apoio à sua esquerda, nomeadamento nas
aprovações dos Orçamentos de Estado.

## Ponto de partida e destino

Se o trabalho original foi pensado como resposta à entrada de novos
partidos no Parlamento (IL e Chega) e o debate que se fez sobre o seu
posicionamento geométrico no hemiciclo, esta actualização mantém
**exactamente** os mesmos princípios e visualizações (com a eventual
correcção de erros ou melhoramentos), sendo que no momento em que é
publicado a leitura será, inevitavelmente, mais ampla: a cada altura
específica, e decorrendo da forma como os dados podem ser mais ou
menos interessantes para o debate político, os dados que apresentámos
são alvo de picos de interesse.

O objectivo inicial do trabalho que introduziu a abordagem utilizada
era razoavelmente simples:

> O ponto de partida para esta análise foi precisamente tentar descobrir
> se exclusivamente com base na actividade parlamentar, e em concreto no
> registo de votações, é possível estabelecer relações de proximidade e
> distância que permitam um agrupamento que não dependa de
> classificações a priori, e se sim, de que forma estes agrupamentos
> confirmam ou divergem da percepção existente?


```{margin} Nota
Este trabalho está a ser publicado numa altura onde a marcação de eleições antecipadas é o cenário quase certo.
```

Essa simplicidade faz com que os mecanismos de análise utilizados
sejam genéricos e aplicáveis de forma mais vasta: se serviram
inicialmente para enquadrar a questão do local onde os novos partidos
se iriam sentar, servem agora para uma restrospectiva dos 6 anos que
permeiam a moção de censura que derrubou o recém-eleito governo do PSD
e CDS em 2015, até à não aprovação do Orçamento de Estado de 2022
