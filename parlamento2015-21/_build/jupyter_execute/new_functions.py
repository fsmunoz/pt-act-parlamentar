#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from IPython.display import display
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import cm
ndf = pd.DataFrame()
#submissions_ini_nu = submissions_ini.loc[submissions_ini['unanime'] != "unanime"]
gpsubs = submissions_ini
cmap=ListedColormap(sns.color_palette("pastel").as_hex())
colors=["#fdfd96",  "black","#ff6961","#77dd77", ]
cmap = ListedColormap(colors)

spn = 0
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 30))
axes = axes.ravel()
for party in parties:
    sns.set_style("whitegrid")
    subp = gpsubs[gpsubs['GP'] == party]
    sp = subp[parties].apply(pd.Series.value_counts).fillna(0).drop([party],axis=1)
    sp = sp.sort_values(by=['Favor','Abstenção','Contra'], ascending=False, axis=1)
    d = sp.T
    #f = plt.figure()
    plt.title(party)
    #d.plot(kind='bar', ax=f.gca(), stacked=True, title=party, colormap=cmap, ax=spn)
    d.plot(kind='bar', stacked=True, title=party, colormap=cmap, ax=axes[spn])
    #plt.legend(loc='center left',  bbox_to_anchor=(0.7, 0.9),)
    axes[spn].get_legend().remove()
    spn += 1

axes[11].set_axis_off()
plt.show()


# In[ ]:


political_distance_matrix(l13_votes, parties)


# In[ ]:


political_distance_clustermap(l13_votes, parties,"Clustermap")


# In[ ]:


political_distance_clustermap(l14_votes, l14_parties,"Clustermap")


# In[ ]:


political_dendogram(l13_votes, parties,"Clustermap")


# In[ ]:


#df[(df['date'] > '2013-01-01') & (df['date'] < '2013-02-01')]

for year in [2015,2016,2017,2018,2019]:
    y_votes=l13_votes[(l13_votes['data'] > '{}-01-01'.format(2015)) & (l13_votes['data'] < '{}-01-01'.format(year+1))]
    political_distance_clustermap(y_votes, parties,"Clustermap for Year {}".format(year))


# In[ ]:


#df[(df['date'] > '2013-01-01') & (df['date'] < '2013-02-01')]

for year in [2019, 2020, 2021]:
    y_votes=l14_votes[(l14_votes['data'] > '{}-01-01'.format(year)) & (l14_votes['data'] < '{}-01-01'.format(year+1))]
    political_distance_clustermap(y_votes, l14_parties,"Clustermap for Year {}".format(year))


# In[ ]:


## Display the heatmap of the distance matrix

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()

sns.heatmap(
    l13_distmat,
    cmap=sns.color_palette("Reds_r"),
    linewidth=1,
    annot = True,
    square =True,
    cbar_kws={"shrink": 0.8})
plt.title('Portuguese Parliament 13th Legislature, Distance Matrix')

plt.show()


# In[ ]:


## Perform hierarchical linkage on the distance matrix using Ward's method.
l13_distmat_link = hc.linkage(l13_pwdist, method="ward", optimal_ordering=True )

sns.clustermap(
    l13_distmat,
    annot = True,
    cmap=sns.color_palette("Reds_r"),
    linewidth=1,
    #standard_scale=1,
    row_linkage=l13_distmat_link,
    col_linkage=l13_distmat_link,
    figsize=(8,8)).fig.suptitle('Portuguese Parliament 13th Legislature, Clustermap')

plt.show()


# In[ ]:


from scipy.cluster.hierarchy import dendrogram
fig = plt.figure(figsize=(8,5))
dendrogram(l13_distmat_link, labels=l13_votes_hmn.columns)

plt.title("Portuguese Parliament 14th Legislature, Dendogram")
plt.show()


# In[ ]:


l13_inis = ini_to_df_ini(l13_ini_tree)
l14_inis = ini_to_df_ini(l14_ini_tree)


# In[ ]:


l14_inis


# In[ ]:


init_ct = pd.crosstab(l13_inis.iniAutorGruposParlamentares,l13_inis.iniTipo)
init_ct14 = pd.crosstab(l14_inis.iniAutorGruposParlamentares,l14_inis.iniTipo)


# In[ ]:


import plotly.express as px

long_df = px.data.medals_long()

fig = px.bar(init_ct, template="plotly_white")
fig.show()


# In[ ]:


import seaborn as sns
#sns.set(font="EB Garamond")

#sns.barplot(init_ct, x="iniAutorGruposParlamentares", y="iniTipo")


# In[ ]:


init_ct14


# In[ ]:


import plotly.express as px

long_df = px.data.medals_long()

fig = px.bar(init_ct14, template="plotly_white")
fig.show()


# In[ ]:


import pandas as pd
from itables import show
import qgrid
#l13_ini_df = pd.DataFrame(l13_init_list)
#print(ini_df.shape)
l13_ini_df = ini_to_df(l13_ini_tree)

widget = qgrid.show_grid(l13_ini_df)
widget


# In[ ]:


l13_ini_df


# In[ ]:


l14_ini_df = ini_to_df(l14_ini_tree)
## Copy Livre voting record to new aggregate columns...
l14_ini_df["L/JKM"] = l14_ini_df["L"]
## ... and fill the NAs with JKM voting record.
l14_ini_df["L/JKM"] = l14_ini_df["L/JKM"].fillna(l14_ini_df["Joacine Katar Moreira (Ninsc)"])

#print(l14_ini_df[["descricao","L","Joacine Katar Moreira (Ninsc)","L/JKM"]])
## Copy PAN voting record to new aggregate columns...
l14_ini_df["PAN/CR"] = l14_ini_df["PAN"]
## ... and update/replace with CR voting where it exists
l14_ini_df["PAN/CR"].update(l14_ini_df["Cristina Rodrigues (Ninsc)"])
#l14_ini_df["PAN/CR"] = l14_ini_df["PAN/CR"].fillna(l14_ini_df["Cristina Rodrigues (Ninsc)"])
#print(l14_ini_df[["id","descricao","PAN","Cristina Rodrigues (Ninsc)","PAN/CR"]])
l14_votes = l14_ini_df.drop(["tipoReuniao"],axis=1)


# In[ ]:


from pivottablejs import pivot_ui

pivot_ui(votes[parties.insert(0,"resultado")], outfile_path="l13_pivot.html")
display(HTML("l13_pivot.html"))
pivot_ui(votes[parties.insert(0,"resultado")])

