import pandas as pd
import json
import re
import string
import unidecode
import nltk
from nltk.corpus import stopwords
import networkx as nx
import chardet
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Encontrar la codificación del archivo
with open("traficogt.txt", "rb") as f:
    rawdata = f.read(100000)  # leer un bloque
    print(chardet.detect(rawdata))
    
# Cargar tweets desde archivo JSONL (cada línea es un JSON)
tweets = []
with open("traficogt.txt", "r", encoding="utf-16") as f:
    for line_num, line in enumerate(f):
        try:
            tweets.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping line {line_num + 1} due to JSONDecodeError: {e}")
        except Exception as e:
            print(f"Skipping line {line_num + 1} due to unexpected error: {e}")


df = pd.json_normalize(tweets)  # Convertir a DataFrame
print(df.head(3))

print("Total tweets cargados:", len(df))

nltk.download("stopwords")
spanish_stopwords = set(stopwords.words("spanish"))

def clean_text(text):
    text = text.lower()                          
    text = re.sub(r"http\S+|www\S+", "", text)   
    text = re.sub(r"@\w+", "", text)             
    text = re.sub(r"#\w+", "", text)             
    text = re.sub(r"[0-9]+", "", text)           
    text = text.translate(str.maketrans("", "", string.punctuation)) 
    text = "".join(c for c in text if c.isalnum() or c.isspace())     
    text = unidecode.unidecode(text)           
    tokens = [t for t in text.split() if t not in spanish_stopwords] 
    return " ".join(tokens)

df["clean_text"] = df["rawContent"].astype(str).apply(clean_text)
df[["rawContent","clean_text"]].head(5)

edges = []

for _, row in df.iterrows():
    user = row["user.username"]

    # Menciones
    if isinstance(row.get("mentionedUsers"), list):
        for mention in row["mentionedUsers"]:
            edges.append((user, mention["username"], "mention"))

    # Retweets
    if row.get("retweetedTweet") is not None and isinstance(row["retweetedTweet"], dict):
        rt_user = row["retweetedTweet"]["user"]["username"]
        edges.append((user, rt_user, "retweet"))

    # Respuestas
    if row.get("inReplyToUser") is not None and isinstance(row["inReplyToUser"], dict):
        reply_user = row["inReplyToUser"]["username"]
        edges.append((user, reply_user, "reply"))

edges_df = pd.DataFrame(edges, columns=["source", "target", "type"])
print(edges_df.head(10))

edges_df.drop_duplicates(inplace=True)
edges_df["source"] = edges_df["source"].str.lower()
edges_df["target"] = edges_df["target"].str.lower()

G = nx.from_pandas_edgelist(
    edges_df, source="source", target="target", edge_attr="type", create_using=nx.DiGraph()
)

print("Nodos:", G.number_of_nodes())
print("Aristas:", G.number_of_edges())

edges_df.to_csv("red_interacciones.csv", index=False)

# 4) Análisis exploratorio

# Número de tweets cargados
total_tweets = len(df)

# Usuarios únicos
usuarios_unicos = df["user.username"].nunique()

# Número de menciones totales
total_menciones = edges_df[edges_df["type"] == "mention"].shape[0]

# Número de retweets
total_retweets = edges_df[edges_df["type"] == "retweet"].shape[0]

# Número de respuestas
total_respuestas = edges_df[edges_df["type"] == "reply"].shape[0]

print("Tweets totales:", total_tweets)
print("Usuarios únicos:", usuarios_unicos)
print("Menciones:", total_menciones)
print("Retweets:", total_retweets)
print("Respuestas:", total_respuestas)

# Hashtags más frecuentes
def extraer_hashtags(texto):
    return re.findall(r"#\w+", str(texto).lower())

hashtags = df["rawContent"].apply(extraer_hashtags)
hashtags = [tag for sublist in hashtags for tag in sublist]

top_hashtags = Counter(hashtags).most_common(10)
print(top_hashtags)

# Nube de palabras de tweets limpios
text = " ".join(df["clean_text"].dropna())

wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Usuarios más mencionados y más activos

# Más mencionados
top_mencionados = edges_df[edges_df["type"]=="mention"]["target"].value_counts().head(10)

# Más activos (quienes más interactúan)
top_activos = edges_df["source"].value_counts().head(10)

print("Usuarios más mencionados:\n", top_mencionados)
print("\nUsuarios más activos:\n", top_activos)

# 5.1) Construcción y visualización de grafos

# Grafo reducido (para visualizar, usamos solo los más activos y mencionados)
top_nodes = edges_df["target"].value_counts().head(30).index.tolist()
subG = G.subgraph(top_nodes)

plt.figure(figsize=(12,8))
pos = nx.spring_layout(subG, k=0.5, seed=42)

# Dibujar nodos
nx.draw_networkx_nodes(subG, pos, node_size=800, node_color="lightblue")

# Dibujar aristas
nx.draw_networkx_edges(subG, pos, alpha=0.3, arrows=True)

# Etiquetas de los nodos
nx.draw_networkx_labels(subG, pos, font_size=10)

plt.title("Red de interacciones (usuarios más mencionados)")
plt.axis("off")
plt.show()

# 5.2) Cálculo de métricas de red clave

densidad = nx.density(G)
print("Densidad de la red:", densidad)

if nx.is_strongly_connected(G):
    diametro = nx.diameter(G)
else:
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    subG = G.subgraph(largest_cc)
    diametro = nx.diameter(subG)

print("Diámetro de la red:", diametro)

clustering = nx.average_clustering(G.to_undirected())
print("Coeficiente de agrupamiento:", clustering)

# 6)

# Detección de comunidades
partition = community_louvain.best_partition(G.to_undirected())

# Guardar comunidad asignada a cada nodo
nx.set_node_attributes(G, partition, "comunidad")

# Número de comunidades
num_comunidades = len(set(partition.values()))
print("Número de comunidades detectadas:", num_comunidades)

# Posiciones de nodos
pos = nx.spring_layout(G, k=0.3, seed=42)

plt.figure(figsize=(12,10))

# Dibujar nodos coloreados por comunidad
nx.draw_networkx_nodes(G, pos,
                       node_size=40,
                       cmap=plt.cm.tab10,
                       node_color=list(partition.values()),
                       alpha=0.7)

nx.draw_networkx_edges(G, pos, alpha=0.2)

plt.title("Comunidades detectadas con Louvain")
plt.axis("off")
plt.show()

from collections import defaultdict

# Agrupar nodos por comunidad
comunidades_dict = defaultdict(list)
for nodo, com in partition.items():
    comunidades_dict[com].append(nodo)

# Mostrar los primeros nodos de la comunidad más grande
for com_id, nodos in comunidades_dict.items():
    print(f"Comunidad {com_id} ({len(nodos)} nodos): {nodos[:10]}")

print(" ")
print("Top Comunidades más grandes")

# Contar tamaño de cada comunidad
comunidades = pd.Series(list(partition.values())).value_counts()
print(comunidades.head(5))

# Agregar etiquetas de comunidad al dataframe de interacciones
edges_df["source_comunidad"] = edges_df["source"].map(partition)
edges_df["target_comunidad"] = edges_df["target"].map(partition)

# Top 3 comunidades más grandes
top3 = comunidades.head(3).index.tolist()

colors = {c: i for i, c in enumerate(top3)}

plt.figure(figsize=(12,8))
sub_nodes = [n for n, c in partition.items() if c in top3]
subG = G.subgraph(sub_nodes)

nx.draw_networkx(subG, pos=nx.spring_layout(subG, k=0.4, seed=42),
                 node_color=[colors[partition[n]] for n in subG.nodes()],
                 node_size=50, cmap=plt.cm.Set1, alpha=0.8, with_labels=False)

plt.title("Tres comunidades más grandes")
plt.show()

# 7.1) centralidades

# Centralidad de grado
grado = nx.degree_centrality(G)

# Centralidad de intermediación (betweenness)
betweenness = nx.betweenness_centrality(G)

# Centralidad de cercanía
closeness = nx.closeness_centrality(G)

# Top 10 usuarios en cada métrica
top_grado = sorted(grado.items(), key=lambda x: x[1], reverse=True)[:10]
top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top centralidad de grado:", top_grado)
print("\nTop centralidad de intermediación:", top_betweenness)
print("\nTop centralidad de cercanía:", top_closeness)

# 8.1) subredes y nodos aislados

# Convertir a no dirigido para análisis de conectividad
G_und = G.to_undirected()

# Detectar componentes conexas
components = list(nx.connected_components(G_und))
print("Número de subredes detectadas:", len(components))

# Ordenarlas por tamaño
components_sorted = sorted(components, key=len, reverse=True)

# Subredes pequeñas (aisladas)
isolated_groups = [c for c in components_sorted if len(c) < 10]
print("Ejemplo de grupos aislados (usuarios):", isolated_groups[:5])

df["sentiment"] = df["clean_text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

# Clasificación: positivo (>0.05), negativo (<-0.05), neutral (entre -0.05 y 0.05)
def classify_sentiment(score):
    if score > 0.05: return "positivo"
    elif score < -0.05: return "negativo"
    else: return "neutral"

df["sentiment_label"] = df["sentiment"].apply(classify_sentiment)

print(df["sentiment_label"].value_counts(normalize=True))

stopsp = stopwords.words("spanish")

vectorizer = CountVectorizer(max_df=0.9, min_df=10, stop_words=stopsp)
X = vectorizer.fit_transform(df["clean_text"])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Mostrar palabras clave por tema
for i, topic in enumerate(lda.components_):
    words = [vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-10:]]
    print(f"Tema {i+1}: {', '.join(words)}")





