#!/usr/bin/env python
# coding: utf-8

# # Os dados do Orçamento de Estado de 2021
# 
# ```{margin} _Caveat emptor!_
# A existência de erros e omissões é uma possibilidade a não desprezar: vários foram detectados durante as fases iniciais deste projecto, e alguns nas fases finais. Dada a importância que os dados têm para todas as outras análises o impacto pode ir desde irrelevante até significativo.
# ```
# A base de todos os dados é a informação disponibilizada pela Assembleia da República na secção [Dados Abertos](https://www.parlamento.pt/Cidadania/Paginas/DadosAbertos.aspx). Sobre estes são efectuadas várias transformações que resultam da selecção da informação pretendida; neste processo, e para além das decisões sobre o conteúdo a seleccionar, os dados originais são convertidos em formato tabular.
# 
# As análises anteriores mostram segmentos desta informação em forma de tabelas, sempre de tamanho limitado para se enquadrar no fluxo do texto; aqui colocamos um conjunto alargado da informação recolhida sobre as votações do Orçamento de Estado de 2021, essencialmente as Propostas de Alteração e como foram votadas.
# 

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

#oe_url = "http://app.parlamento.pt/webutils/docs/doc.xml?path=6148523063446f764c324679626d56304c3239775a57356b595852684c3052685a47397a51574a6c636e52766379395063734f6e5957316c626e52764a5449775a47386c4d6a4246633352685a47387657456c574a5449775447566e61584e7359585231636d457654305651636d397762334e3059584e426248526c636d466a595738794d4449785433497565473173&fich=OEPropostasAlteracao2021Or.xml&Inline=true"
#oe_tree = ET.parse(urlopen(oe_url))
oe_file = './OEPropostasAlteracao2021Or.xml'
oe_tree = ET.parse(oe_file)


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


# In[6]:


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


# In[7]:


import pandas as pd

oe_df = pd.DataFrame(oe_list)
oe_df
oe_df["VoteDesc"] = oe_df["VoteDesc"] + ": " + oe_df["SubDesc"]
oe_df.drop(['SubDesc'],axis=1,inplace=True)
oe_df = oe_df.fillna("")
oe_df


# In[8]:


oe_df.to_json("oe2021.json", orient="records")


# In[9]:


from IPython.display import display, HTML
js = '''
<div>
    <button id="download-csv">Download CSV</button>
    <button id="download-json">Download JSON</button>
</div>

<div id="example-table"></div>
<script>
var table = new Tabulator("#example-table", {
    pagination:"local",
    responsiveLayout:true,
    paginationSize:50,         //allow 20 rows per page of data
    movableColumns:true,      //allow column order to be changed
    resizableRows:true,       //allow row order to be changed
    autoColumns:true,
    layout:"fitColumns",
    ajaxURL:"./oe2021.json", //ajax URL
});

//trigger download of data.csv file
document.getElementById("download-csv").addEventListener("click", function(){
    table.download("csv", "oe2021.csv");
});

//trigger download of data.json file
document.getElementById("download-json").addEventListener("click", function(){
    table.download("json", "oe2021.json");
});

</script>
'''
display(HTML(js))

