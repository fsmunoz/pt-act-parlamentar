#!/usr/bin/env python
# coding: utf-8

# # Início
# 
# ```{epigraph}
# E por isso o dizem na _Turba Philosophorum_, que qualquer que seja a cor que vem após o negro é louvável, pois é o início da Obra.
# 
# -- Michael Meiers, «Atalanta Fugiens»
# ```
# 

# :::{margin}
# ```{image} ./_images/atalanta-fugiens-emblem-12.jpg
# :alt: Atalanta Fugiens, 12
# :width: 200px
# :align: center
# ```
# :::

# Com quem vota cada um dos partidos? Que relações se podem extrair dos padrões de votação - e apenas das votações? Observar-se-á um resultado enquadrável nos conceitos de Esquerda e Direita ou, pelo contrário, emergem dimensões diferentes?
# 
# Nestas páginas iremos obter, processar e analisar todo o histórico de votações do período 2015-2021, aplicando um conjunto de métricas
# 
# * A obtenção dos dados das votações realizadas com base nos dados abertos disponibilizados pela página oficial do Parlamento.
# * O tratamento dos dados de forma à criação de uma tabela consistente e mais facilmente analisável
# * A criação de diagramas e tabelas de agrupamento, matrizes de distância e afinidade e mapas bi e tridimensionais de posicionamento relativos dos partidos.
# 
# Um factor que define a metodologia utilizada é a _utilização exclusiva dos dados das votações_, e apenas esses dados; não é feita nenhuma valorização "manual" das votações, sendo que o resultado final é uma resposta a uma questão simples: qual a relação dos vários partidos tendo em conta não o seu programa, não as suas afirmações públicas, mas o seu comportamento concreto naquilo que define a sua actividade parlamentar - as votações.
# 
# A utilidade deste tipo de análise em ciência política é reconhecida {cite:p}`filhoClusterAnalysisPolitical2014` e tem sido aplicada a vários registos de votações; a análise presente tem como principal diferença o ser efectuado sobre as votações de partidos e não, como é mais comum na bibliografia consultada, a deputados individuais.
# 
# ## Um novo olhar sobre um momento único
# 
# Esta análise serve, também, para um olhar adicional a uma experiência governativa que, independentemente das opiniões sobre o seu mérito, foi inovadora em vários aspectos: a não formação de governo pela coligação mais votada e, sobretudo, a formação de um governo PS sustentado com apoio (variável, e também isso será interessante analisar com base nos dados) dos partidos à sua Esquerda foi uma ruptura assinalável com o que tinha sido a 
# 
# Se a mera possibilidade desta solução (que passaria a ser comumente chamada de _geringonça_ {cite:p}`valenteGeringonca2015`) foi posta em causa por vários sectores da sociedade (incluíndo orgãos de soberania), mais ainda divergiram os prognósticos em relação à sua durabilidade: dias ou meses para alguns, alguns anos para outros, até visões de prolongamento indefinido pelas décadas após as primeiras aprovações de orçamentos.
# 
# Este trabalho é uma actualização do anteriormente realizado, para um período de tempo menor, e que introduziu uma abordagem específica para a análise da proximidade, distância e suporte mútuo entre todos os grupos parlamentares e deputados independentes presentes na Assembleia da República.

# In[ ]:




