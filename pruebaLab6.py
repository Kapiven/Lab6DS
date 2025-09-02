import pandas as pd
import json
import re
import string
import unidecode
import nltk
from nltk.corpus import stopwords
import networkx as nx
import chardet

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
