#!/usr/bin/env python
# coding: utf-8

# # Orçamento de Estado de 2021

# A análise das votações e posicionamento relativo dos partidos tendo como base *exclusivamente* a forma como votam foi a base do trabalho anterior, análise essa que teve como fonte as votações das Iniciativas e Actividades.
# 
# O Orçamento de Estado para 2021 foi um acontecimento marcante da realidade política nacional do último trimestre de 2020 e que continuará a ter impactos na apreciação e debate político no futuro próximo: a posição dos vários partidos em termos de viabilização ou não do OE, as propostas que apresentaram e quem as aprovou ou recusou são matéria utilizada no debate para as Presidenciais, por exemplo.
# 
# Estas votações não estão incluídas na análise anterior: os dados utilizados relativs a votações no Parlamento não incluem, por exemplo, as votações feitas em Comissões. Esta actualização aborda um conjunto de votações independente, muito mais circunscrito no tempo e (dentro da grande variedade de temas abordados no OE) no conteúdo.

# ## Metodologia

# Com base nos dados disponibilizados pela Assembleia da República em formato XML [DadosAbertos] são criadas _dataframes_ (tabelas de duas dimensões) com base na selecção de informação relativa aos padrões de votação de cada partido (e/ou deputados não-inscritos)
# 
# São fundamentalmente feitas as seguintes análises:
# 
# 1. Quantidade e tipo de propostas feitas, e resultado das mesmas
# 2. Apoio para as propostas de cada partido
# 2. Matriz de distância entre todos os partidos e dendograma
# 3. Identificação de grupos (_spectral clustering_) e visualização das distâncias num espaço cartesiano (_multidimensional scaling_)

# In[1]:


get_ipython().system('pip3 install --user -q  matplotlib pandas seaborn sklearn ')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
import random
import seaborn as sns
from matplotlib.colors import ListedColormap
sns.set_theme(style="whitegrid", palette="pastel")


# In[3]:


from urllib.request import urlopen
import xml.etree.ElementTree as ET

oe_url = "http://app.parlamento.pt/webutils/docs/doc.xml?path=6148523063446f764c324679626d56304c3239775a57356b595852684c3052685a47397a51574a6c636e52766379395063734f6e5957316c626e52764a5449775a47386c4d6a4246633352685a47387657456c574a5449775447566e61584e7359585231636d457654305651636d397762334e3059584e426248526c636d466a595738794d4449785433497565473173&fich=OEPropostasAlteracao2021Or.xml&Inline=true"
oe_tree = ET.parse(urlopen(oe_url))


# In[4]:


import collections

counter=0
vc=0
## We will build a dataframe from a list of dicts
## Inspired by the approach of Chris Moffitt here https://pbpython.com/pandas-list-dict.html
oe_list = []

for alter in oe_tree.findall(".//PropostaDeAlteracao"):
    votep = alter.find('./Votacoes')
    if votep is not None:
        oe_dict = collections.OrderedDict()
        counter +=1
        oe_dict["ID"]=alter.find('./ID').text
        oe_dict["Nr"]=alter.find('./Numero').text
        oe_dict["Date"]=alter.find('./Data').text
        oe_dict["Domain"]=alter.find('./Tema').text
        oe_dict["Type"]=alter.find('./Tipo').text
        oe_dict["State"]=alter.find('./Estado').text
        oe_dict["GP"]=alter.find('./GrupoParlamentar_Partido').text
        init_title=alter.find('./Iniciativas_Artigos/Iniciativa_Artigo/Titulo')
        if init_title is not None:
            oe_dict["IniTitle"]=init_title.text
        #oe_list.append(oe_dict)
        for vote in alter.findall("./Votacoes/"):
            vc +=1
            for vote_el in vote:
                if vote_el.tag == "Data":
                    oe_dict["V_Date"] = vote_el.text
                    # print(oe_dict["ID"])
                if vote_el.tag == "Descricoes":
                    descp = vote_el.find('./Descricao')
                    if descp is not None: 
                        oe_dict["VoteDesc"] = vote_el.find('./Descricao').text
                if vote_el.tag == "Resultado":
                    oe_dict["Result"] = vote_el.text
                for gps in vote.findall("./GruposParlamentares/"):
                    if gps.tag == "GrupoParlamentar":
                        gp = gps.text
                    else:
                        oe_dict[gp] = gps.text

        oe_list.append(oe_dict)
    print('.', end='')
        
print("\nProposals:",counter)
print(vc)


# In[5]:


import pandas as pd

oe_df = pd.DataFrame(oe_list)
oe_df


# ```{margin}
# Em gráfico de barras:
# ```

# ## As propostas: quantidade, aprovações, rejeições
# 
# Após obtermos e processarmos o ficheiro com as Propostas de Alteração podemos ter uma primeira ideia sobre a origem das propostas:

# In[6]:


oe_df.groupby('GP')[['ID']].count().sort_values(by=['ID'], axis=0, ascending=False).plot(kind="bar",stacked=True,figsize=(6,6))
plt.show()


# In[7]:


oe_df.groupby('GP')[['ID']].count().sort_values("ID", ascending=False)


# ```{margin}
# «Os partidos que viabilizaram o OE 2021, sem contar com o PS, entregaram 58% das propostas de alteração, com um total de 898 medidas. O PCP, que é decisivo na votação final, é o que mais quer mudar OE», {cite}`eco58PropostasPara2020`)
# ```

# A quantidade total de propostas está em linha com o que se noticiou na altura relativamente à fonte maioritária das propostas de alteração.
# 
# Sabemos a quantidade total de propostas, mas quais foram aprovadas? Esta tabela indica os resultados das propostas de todos os partidos e deputados:

# In[8]:


pd.crosstab(oe_df.GP, oe_df.State).columns

ct = pd.crosstab(oe_df.GP, oe_df.State)[['Aprovado(a) por Unanimidade em Plenário',
                                         'Aprovado(a) por Unanimidade em Comissão',
                                         'Aprovado(a) em Plenário',
                                         'Aprovado(a) em Comissão',
                                         'Aprovado(a) Parcialmente em Plenário',
                                         'Aprovado(a) Parcialmente em Comissão',
                                         'Entrada (via IPA)',
                                         'Aguarda Voto em Comissão',
                                         'Retirado(a)',
                                         'Prejudicado(a)',
                                         'Rejeitado(a) em Plenário',
                                         'Rejeitado(a) em Comissão'
                                        ]]
ct


# A mesma informação em forma de gráfico de barras: o total de propostas de cada partido (ou deputados) com a distribuição do resultado das mesmas, ordenados pelo maior número de aprovações.

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
#    sp = sp.sort_values(by=['Favor','Abstenção','Contra'], ascending=False, axis=1)
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set()
sns.set_style("whitegrid")

## Sort by the "Approved in Comission" since this is >95% of the approved group and simplifies having to
## sort by a separate aggregate value.
ct.sort_values(by=['Aprovado(a) em Comissão', 'Aprovado(a) em Plenário','Aprovado(a) Parcialmente em Comissão','Aprovado(a) Parcialmente em Plenário','Aprovado(a) por Unanimidade em Comissão','Aprovado(a) por Unanimidade em Plenário'], ascending=False,axis=0).plot(kind="bar", stacked=True, colormap=ListedColormap(sns.color_palette("coolwarm").as_hex()),figsize=(10,10))
plt.show()


# ```{margin}
# «("PS, PCP, PAN e PEV viabilizam OE 2021. BE e direita votam contra", https://eco.sapo.pt/2020/10/28/ps-pcp-pan-e-pev-viabilizam-oe-2021-be-e-direita-votam-contra/», {cite}`ecoPSPCPPAN2020`.
# ```

# Algumas das leituras que se podem tirar desdes dados: 
# 
# * O maior número de propostas submetidas, rejeitadas mas também aprovadas é do PCP.
# * O PS é o que tem uma percentagem de aprovação maior.
# * CH não tem nenhuma. 
# * Os restantes partidos e deputados têm a maioria das suas propostas rejeitadas e, em diferente número (BE tem 1 alteração aprovada), algumas aprovadas.
# 
# Estes resultados podem ser relacionados com o que foi a viabilização do OE: de facto os 4 primeiros partidos por número de alterações aprovadas são os que participaram na viabilização , sendo legítimo considerar que esse facto está directamente relacionado.
# 
# ### Propostas aprovadas
# 
# E que propostas cada partido conseguiu aprovar? A utilização do título da iniciativa é aqui útil

# In[10]:


from IPython.display import display, HTML

approved_oe = oe_df[oe_df.State.str.contains("Aprovado")].fillna("")

for gp in approved_oe.GP.unique():
    gp_df = approved_oe[approved_oe["GP"]==gp][["GP","IniTitle", "VoteDesc", "State"]]
    print(gp + ":", len(gp_df.index), " aprovadas.")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):  # more options can be specified also
        display(gp_df)


# ### Propostas rejeitadas
# 
# Da mesma forma podemos obter a lista de propostas rejeitas, por partido.

# In[11]:


from IPython.display import display, HTML

approved_oe = oe_df[~oe_df.State.str.contains("Aprovado")].fillna("")

for gp in approved_oe.GP.unique():
    gp_df = approved_oe[approved_oe["GP"]==gp][["GP","IniTitle", "VoteDesc", "State"]]
    print(gp + ":", len(gp_df.index), " Rejeitadas.")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):  # more options can be specified also
        display(gp_df)


# ## As votações
# 
# Até agora contabilizámos as propostas de alteração e o seu resultado; se em geral existe uma votação por proposta de alteração isso nem sempre acontece: existem propostas de alteração que dão origem a mais que uma votação. As votações contêm informação adicional que é interessante para de determinar de forma directa o teor das propostas (nomeadamente o título) e também a forma como os diferentes partidos e deputados votaram: se ao nível das propostas temos o resultado final, com as votações podemos saber como atingiram esse fim.
# 
# Após processarmos as votações enriquecemos o _dataframe_ com informação adicional:

# In[12]:


import collections

counter=0
vc=0
## We will build a dataframe from a list of dicts
## Inspired by the approach of Chris Moffitt here https://pbpython.com/pandas-list-dict.html
oe_list = []

for alter in oe_tree.findall(".//PropostaDeAlteracao"):
    votep = alter.find('./Votacoes')
    if votep is not None:
        oe_dict = collections.OrderedDict()
        counter +=1
        oe_dict["ID"]=alter.find('./ID').text
        oe_dict["Nr"]=alter.find('./Numero').text
        oe_dict["Date"]=alter.find('./Data').text
        oe_dict["Domain"]=alter.find('./Tema').text
        oe_dict["Type"]=alter.find('./Tipo').text
        oe_dict["State"]=alter.find('./Estado').text
        oe_dict["GP"]=alter.find('./GrupoParlamentar_Partido').text
        init_title=alter.find('./Iniciativas_Artigos/Iniciativa_Artigo/Titulo')
        if init_title is not None:
            oe_dict["IniTitle"]=init_title.text
        #oe_list.append(oe_dict)
        for vote in alter.findall("./Votacoes/"):
            vc +=1
            for vote_el in vote:
                if vote_el.tag == "Data":
                    oe_dict["V_Date"] = vote_el.text
                    # print(oe_dict["ID"])
                if vote_el.tag == "Descricoes":
                    descp = vote_el.find('./Descricao')
                    if descp is not None: 
                        oe_dict["VoteDesc"] = vote_el.find('./Descricao').text
                if vote_el.tag == "SubDescricao":
                    oe_dict["SubDesc"] = vote_el.text
                if vote_el.tag == "Resultado":
                    oe_dict["Result"] = vote_el.text
                for gps in vote.findall("./GruposParlamentares/"):
                    if gps.tag == "GrupoParlamentar":
                        gp = gps.text
                    else:
                        oe_dict[gp] = gps.text

            oe_list.append(oe_dict)
    print('.', end='')
        
print("\nProposals:",counter)
print("Voting sessions:", vc)


# In[13]:


import pandas as pd

oe_df = pd.DataFrame(oe_list)
oe_df
oe_df["VoteDesc"] = oe_df["VoteDesc"] + ": " + oe_df["SubDesc"]
oe_df.drop(['SubDesc'],axis=1,inplace=True)
oe_df


# In[14]:


#df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
oe_dfr = oe_df.rename(columns={'Partido Socialista': 'PS', 
                               'Partido Social Democrata': 'PSD',
                               'Bloco de Esquerda': 'BE',
                               'Partido Comunista Português': 'PCP',
                               'Centro Democrático Social - Partido Popular': 'CDS-PP',
                               'Pessoas-Animais-Natureza': 'PAN',
                               'Chega': 'CH',
                               'Iniciativa Liberal':'IL',
                               'Partido Ecologista "Os Verdes"':'PEV',
                               'Livre':'L',
                               'CRISTINA RODRIGUES(Ninsc)': 'CR',
                               'JOACINE KATAR MOREIRA(Ninsc)': 'JKM'
                               
                              })
oe_dfr.replace("CRISTINA RODRIGUES","CR", inplace=True)
oe_dfr.replace("JOACINE KATAR MOREIRA","JKM", inplace=True)
oe_dfr.head()


# Para a análise das votações escolhemos um subconjunto alargado dos autores das propostas, isto porque nem todos os partidos estão presentes na Comissão de Orçamento e Finanças e, por outro lado, as propostas de deputados individuais ou em grupo necessitaria de um tratamento mais complexo.
# O resultado é uma tabela com a indicação do autor da proposta onde se integra a votação e os votos dos partidos.

# ```{margin} Votações, partidos e Comissões
# Devido à forma  de funcionamento das Comissões, existem partidos e deputados não-inscritos que não estão presentes em várias delas, resultando num número elevado de ausências. Optámos pela sua não inclusão.
# ```

# In[15]:


mycol  = ['GP', 'BE', 'PCP', 'PEV', 'JKM','PS', 'PAN', 'CR', 'PSD','IL','CDS-PP', 'CH' ]
parties   = ['BE', 'PCP', 'PEV', 'JKM','PS', 'PAN', 'CR','PSD','IL','CDS-PP', 'CH']
df=oe_dfr

submissions_ini = df[mycol]
submissions_ini.head()


# Com esta informação é possível determinar os padrões de votação; o diagrama seguinte mostra a relação entre cada par de partidos: no eixo horizontal quem propõe, e no vertical como votaram:

# In[16]:


parties   = ['BE', 'PCP','PS', 'PAN','PSD','IL','CDS-PP', 'CH']
gpsubs = submissions_ini

cmap=ListedColormap(sns.color_palette("pastel").as_hex())
colors=["#DFE38C","#F59B9B","black","#7FE7CC" ]
cmap = ListedColormap(colors)

spn = 0
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(20, 20))
axes = axes.ravel()
for party in parties:
    for p2 in parties:
        sns.set_style("white")
        subp = gpsubs[gpsubs['GP'] == p2][[party]]
        sp = subp.fillna("Ausência").apply(pd.Series.value_counts)
        d = pd.DataFrame(columns=["GP","Abstenção", "Contra", "Ausência","Favor"]).merge(sp.T, how="right").fillna(0)
        d["GP"] = party
        d = d.set_index("GP")
        d = d[["Abstenção", "Contra", "Ausência","Favor"]]
        if p2 != party:
            sns.despine(left=True, bottom=True)
            if spn < 8:
                d.plot(kind='barh', stacked=True,width=400,colormap=cmap, title=p2,use_index=False,ax=axes[spn])
            else:
                d.plot(kind='barh', stacked=True,width=400,colormap=cmap,use_index=False,ax=axes[spn])
            axes[spn].get_legend().remove()
            plt.ylim(-4.5, axes[spn].get_yticks()[-1] + 0.5)
        else:
            axes[spn].set_xticks([])
            #d.plot(kind='barh', stacked=True,width=400,colormap=cmap,use_index=False,ax=axes[spn])
            #axes[spn].get_legend().remove()
            if spn < 8:
                axes[spn].set_title(p2)
        axes[spn].set_yticks([])
        ## Why? Who knows? Certainly not me. This is likely a side-effect of using a single axis through .ravel
        if spn%8 == 0:
            if spn != 0:
                text = axes[spn].text(-4,0,party,rotation=90)
            else:
                text = axes[spn].text(-0.17,0.5,party,rotation=90)
        #print(party, p2)
        #print(d)
        #print("-------------------------_")
        spn += 1

#axes[11].set_axis_off()
text = axes[0].text(4,1.3,"Quem propôs",rotation=0,fontsize=22)
text = axes[0].text(-0.4,-4,"Como votaram",rotation=90,fontsize=22)

#fig.tight_layout()
plt.show()


# Uma outra visualização, menos condensada mas com maior clareza quantitativa: para cada partido é criado um gráfico de barras, ordenado pelos votos favoráveis, com  comportamento de votos dos restantes para com as suas propostas.

# In[17]:


from IPython.display import display
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import cm
parties   = ['BE', 'PCP', 'PEV', 'JKM','PS', 'PAN', 'CR','PSD','IL','CDS-PP', 'CH']

ndf = pd.DataFrame()
#submissions_ini_nu = submissions_ini.loc[submissions_ini['unanime'] != "unanime"]
gpsubs = submissions_ini
cmap=ListedColormap(sns.color_palette("pastel").as_hex())
colors=["#fdfd96",  "black","#ff6961","#77dd77", ]
cmap = ListedColormap(colors)

#spn = 0
#axes = axes.ravel()

for party in parties:
    sns.set_style("whitegrid")
    subp = gpsubs[gpsubs['GP'] == party]
    sp = subp[parties].apply(pd.Series.value_counts).fillna(0).drop([party],axis=1)
    sp = sp.sort_values(by=['Favor','Abstenção','Contra'], ascending=False, axis=1)
    d = sp.T
    f = plt.figure()
    plt.title(party)
    d.plot(kind='bar', ax=f.gca(), stacked=True, title=party, colormap=cmap,)
    plt.legend(loc='center left',  bbox_to_anchor=(0.7, 0.9),)
    plt.show()

plt.show()


# ## Dendograma e distância
# 
# Com base nas votações obtemos a distância euclideana entre todos os partidos (a distância entre todos os pares possíveis, considerado todas as votações), e com base nela um dendograma que indica a distância entre eles; note-se pelos diagramas acima que o número de votações de PEV. JKM e CR são várias ordens de magnitude inferiores aos restantes. A opção aqui foi removermos estes partidos da análise por considerarmos que o resultado seria muito pouco representativo (e possivelmente enganador) - em todo o caso, é uma opção subjectiva.

# In[18]:


from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import numpy as np
votes_hm = oe_dfr[['BE', 'PCP' , 'PS', 'PAN', 'PSD','IL','CDS-PP', 'CH']]
votes_hmn = votes_hm.replace(["Favor", "Contra", "Abstenção", "Ausente"], [1,-1,0,0]).fillna(0)

## Transpose the dataframe used for the heatmap
votes_t = votes_hmn.transpose()

## Determine the Eucledian pairwise distance
## ("euclidean" is actually the default option)
pwdist = pdist(votes_t, metric='euclidean')

## Create a square dataframe with the pairwise distances: the distance matrix
distmat = pd.DataFrame(
    squareform(pwdist), # pass a symmetric distance matrix
    columns = votes_t.index,
    index = votes_t.index
)

## Normalise by scaling between 0-1, using dataframe max value to keep the symmetry.
## This is essentially a cosmetic step to 

distmat_mm=((distmat-distmat.min().min())/(distmat.max().max()-distmat.min().min()))*1

## Affinity matrix
affinmat_mm = pd.DataFrame(1-distmat_mm, distmat.index, distmat.columns)

#pd.DataFrame(distmat_mm, distmat.index, distmat.columns)
## Perform hierarchical linkage on the distance matrix using Ward's method.
distmat_link = hc.linkage(pwdist, method="ward", optimal_ordering=True )

sns.clustermap(
    distmat,
    annot = True,
    cmap=sns.color_palette("Reds_r"),
    linewidth=1,
    #standard_scale=1,
    row_linkage=distmat_link,
    col_linkage=distmat_link,
    figsize=(8,8)).fig.suptitle('Votações do OE 2021: Clustermap')

plt.show()


# ## MDS: *Multidimensional scaling*
# 
# Tendo como base os votos no OE podemos utilizar a mesma técnica que empregámos na análise de toda a legislatura. Para identificar grupos usamos (mais uma vez, como no trabalho original, e presente nos Apêndices) *Spectral scaling*, definindo 4 grupos.

# ```{margin} Votações por tema
# 
# Com base na identificação do "Tema" (_Domain_, no _dataframe_) pode-se verificar a proximidade (e _clustering_) em termos de cada tema; é necessário ter em conta a diferença (por vezes substancial) do número de votações de cada tema: existem temas com menos de 10 votações, e até com 1, o que produziria um número elevado de diagramas, razão pela qual excluímos os temas com menos de 10 votações.
# ```

# In[19]:



for area in oe_df["Domain"].unique():
    varea=oe_dfr[oe_dfr["Domain"] == area]

    avotes_hm = varea[['BE', 'PCP' , 'PS', 'PAN', 'PSD','IL','CDS-PP','CH']]
    avotes_hmn = avotes_hm.replace(["Favor", "Contra", "Abstenção", "Ausente"], [1,-1,0,0]).fillna(0) 
    if avotes_hmn.shape[0] < 10:
        continue
    avotes_t = avotes_hmn.transpose()
    apwdist = pdist(avotes_t, metric='euclidean')
    
    adistmat = pd.DataFrame(
        squareform(apwdist), # pass a symmetric distance matrix
        columns = avotes_t.index,
        index = avotes_t.index)
    adistmat_mm=((adistmat-adistmat.min().min())/(adistmat.max().max()-adistmat.min().min()))*1
    
    aaffinmat_mm = pd.DataFrame(1-adistmat_mm, adistmat.index, adistmat.columns)

    asc = SpectralClustering(4, affinity="precomputed",random_state=2020).fit_predict(aaffinmat_mm)
    asc_dict = dict(zip(adistmat,asc))   
    
    amds = MDS(n_components=2, dissimilarity='precomputed',random_state=2020, n_init=100, max_iter=1000)
    aresults = amds.fit(adistmat_mm.values)
    acoords = aresults.embedding_
    
    sns.set()
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=(8,8))

    plt.title(area + "(n=" + str(avotes_hmn.shape[0]) +  ")", fontsize=14, fontweight="bold")

    for label, x, y in zip(adistmat_mm.columns, acoords[:, 0], acoords[:, 1]):
        ax.scatter(x, y, c = "C"+str(asc_dict[label]), s=250)
        #ax.scatter(x, y, s=250)
        ax.axis('equal')
        ax.annotate(label,xy = (x-0.02, y+0.025))
    plt.show()
    #print(asc_dict)


# In[20]:


sc = SpectralClustering(3, affinity="precomputed",random_state=2020).fit_predict(affinmat_mm)
sc_dict = dict(zip(distmat,sc))

pd.DataFrame.from_dict(sc_dict, orient='index', columns=["Group"]).T


# O resultado aparente resultar numa divisão "esquerda/centro/direita", com o PS numa posição individual; tal como referido não será indiferente o facto do PS ser o partido que suporta o Governo.
# 
# Esta separação pode ser vista também em termos de agrupamento e proximidade relativa:

# In[21]:


from sklearn.manifold import MDS
import random
mds = MDS(n_components=2, dissimilarity='precomputed',random_state=2020, n_init=100, max_iter=1000)

## We use the normalised distance matrix but results would
## be similar with the original one, just with a different scale/axis
results = mds.fit(distmat_mm.values)
coords = results.embedding_

sns.set()
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(8,8))
fig.suptitle('Portuguese Parliament Voting Records Analysis, 14th Legislature', fontsize=14)
ax.set_title('MDS with Spectrum Scaling clusters (2D)')


for label, x, y in zip(distmat_mm.columns, coords[:, 0], coords[:, 1]):
    ax.scatter(x, y, c = "C"+str(sc_dict[label]), s=250)
    ax.axis('equal')
    ax.annotate(label,xy = (x-0.02, y+0.025))

plt.show()


# O resultado aqui quase mapeia a viabilização do OE: o PS enquanto partido de suporte ao Governo, os partidos à sua direita e os à sua esquerda - e o "quase" é exactamente aqui, visto o BE ter votado de forma próxima, nas votações individuais, a partidos que viabilizaram o OE, sem no entanto o ter feito.
# 
# Uma visualização em 3D adiciona uma dimensão 

# In[22]:


mds = MDS(n_components=3, dissimilarity='precomputed',random_state=1234, n_init=100, max_iter=1000)
results = mds.fit(distmat.values)
parties = distmat.columns
coords = results.embedding_
import plotly.graph_objects as go
# Create figure
fig = go.Figure()

# Loop df columns and plot columns to the figure
for label, x, y, z in zip(parties, coords[:, 0], coords[:, 1], coords[:, 2]):
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z],
                        text=label,
                        textposition="top center",
                        mode='markers+text', # 'lines' or 'markers'
                        name=label))
fig.update_layout(
    width = 700,
    height = 700,
    title = "OE 2021: 3D MDS",
    template="plotly_white",
    showlegend=False
)
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
plot(fig, filename = 'oe_3d_mds.html')
display(HTML('oe_3d_mds.html'))


# ## Palavras finais
# 
# Esta análise demonstra como a utilização de dados abertos pode, uma vez mais, facilitar a análise da actividade parlamentar por parte dos eleitores; as ilações políticas que se podem tirar são variadas e, se é verdade que existem limitações várias a ter em conta, tal como no trabalho relativo à análise das votações parlamentares parecem emergir padrões que parecem reflectir tendências e agrupamentos que estão presentes no discurso político.

# In[ ]:




