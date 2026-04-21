import pandas as pd
import numpy as np
import os
import zipfile
import re
import spacy
import seaborn as sns
import matplotlib.pyplot as plt 
import hashlib


from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ZIP_PATH = "data/legislatives.zip"

data = pd.read_csv("data/metadonnees.csv")
data["date"] = pd.to_datetime(data["date"])

data[["annee","mois","jour"]] = data["date"].apply(
    lambda x: pd.Series([x.year, x.month, x.day])
)


def classifier_parti(etiquette_brute):
    if pd.isna(etiquette_brute) or etiquette_brute == 'nan':
        return np.nan
    
    texte = str(etiquette_brute).lower()
    
    sous_partis = [p.strip() for p in texte.split(';')]
    
    # Drapeaux (Flags) pour détecter les familles présentes
    is_ps = False
    is_udf_rpr = False
    is_centre = False
    is_eco = False
    is_extreme_droite = False
    is_extreme_gauche = False
    is_regionaliste = False
    is_sans_etiquette = False
    
    # Mots-clés par famille
    keywords = {
        'ps': ['parti socialiste', 'ps', 'gauche', 'mouvement des citoyens', 'socialistes', 'socialiste'],

        'rpr' : ['rassemblement pour la république', 'rassemblement pour la republique', 'rpr'],

        'udf' : ['union pour la démocratie française', 'union pour la democratie francaise', 'udf'],
        
        'centre': ['mouvement démocrate', 'modem', 'centriste', 'centre des démocrates sociaux', 
                   'nouveau centre', 'les centristes', 'agir', 'alliance centriste', 
                   'centre', 'mouvement des réformateurs'],
        
        'ecolo': ['vert', 'verts', 'écologie', 'ecolo', 'génération écologie', 'nature et animaux', 
                  'région verte', 'écologiste', 'biosphère', 'generation ecologie', 'écologie les verts',
                  'ecologie les verts', 'ecologie'],
        
        'extreme_droite': ['front national', 'fn', 'mouvement national', 'jeunesse nationaliste', 
                           'alsace d\'abord', 'identitaire', 'patriote', "trop d'immigrés la france aux français"],
        
        'extreme_gauche': ['lutte ouvrière', 'ligue communiste', 'npa', 'nouveau parti anticapitaliste',
                           'parti des travailleurs', 'trotskyste', 'alternative libertaire',
                           'parti communiste français', 'ligue communiste révolutionnaire', 'communistes',
                           'communiste'],
        
        'regionaliste': ['corsica', 'breton', 'occitan', 'alsacien', 'basque', 'euskal', 'unitat catalana',
                         'union démocratique bretonne', 'udb', 'autonomie', 'indépendantiste', 'abertzale',
                         'corse', 'corsa', 'eusko alkartasuna', 'esquerra republicana de catalunya'],
        
        'sans_etiquette': ['sans étiquette', 'divers', 'indépendant', 'apolitique', 'aucun parti', 
                           'non mentionné', 'non inscrit', 'hors des partis', 'société civile', 'libre', 'nan',
                           'sans parti politique', 'aucune formation politique']
    }
    # Analyse de chaque sous-parti
    for parti in sous_partis:
        # Vérification Gauche
        if any(k in parti for k in keywords['ps']):
            is_ps = True
        # Vérification Droite
        if any(k in parti for k in keywords['rpr']) or any(k in parti for k in keywords['udf']):
            is_udf_rpr = True
        # Vérification Centre
        if any(k in parti for k in keywords['centre']):
            is_centre = True
        # Vérification Écologie
        if any(k in parti for k in keywords['ecolo']):
            is_eco = True
        # Vérification Extrême Droite
        if any(k in parti for k in keywords['extreme_droite']):
            is_extreme_droite = True
        # Vérification Extrême Gauche
        if any(k in parti for k in keywords['extreme_gauche']):
            is_extreme_gauche = True
        # Vérification Régionaliste
        if any(k in parti for k in keywords['regionaliste']):
            is_regionaliste = True
        # Vérification Sans étiquette
        if any(k in parti for k in keywords['sans_etiquette']):
            is_sans_etiquette = True

    if is_eco and is_extreme_gauche : 
        return 'Extrême-Gauche'  # 'Ecologiste-Extreme-Gauche

    if is_extreme_droite: return 'Extrême-Droite'
    if is_extreme_gauche: return 'Extrême-Gauche'
    if is_ps: return 'Parti socialiste - Gauche'
    if is_udf_rpr: return 'UDF/RPR'
    if is_centre: return 'Centre'
    if is_eco: return 'Ecologiste'
    if is_regionaliste: return 'Régionaliste'
    if is_sans_etiquette: return 'Sans étiquette / Divers'
    
    # Si rien n'a matché (cas rares ou nouveaux partis)
    return 'Autre / Non classé'


def nettoyage_profession_foi(texte):

    if not isinstance(texte, str):
        return ""
    
    texte = re.sub(r'[\u2600-\u26FF\u2700-\u27BF\u25A0-\u25FF]', ' ', texte)
    #texte = re.sub(r'[A-Z]{4,}\s[A-Z]{4,}', '', texte) 

    # Enlever les 'ELECTIONS LEGISLATIVES'
    texte = re.sub(r'ÉLECTIONS\s[A-Z\s]+-\s[A-Z]+\s\d{4}', '', texte, flags=re.IGNORECASE)

    # Suppression des noms de candidats/suppléants
    texte = re.sub(r'\n?[A-Z][a-z]+\s+[A-Z]{2,}\s+(?:candidat|suppléant)[^\n]*', '', texte) # Suppression des noms de candidats/suppléants
    
    # Suppression de la mention 'Sciences Po/ fonds CEVIPOF'
    texte = re.sub(r'.*?(?:Sciences Po|fonds CEVIPOF|Archives).*?$', '', texte, flags=re.MULTILINE | re.IGNORECASE)
    
    # Nettoyage des espaces et retours à la ligne
    texte = re.sub(r'\n{3,}', '\n\n', texte) 
    texte = re.sub(r'\s+', ' ', texte) 

    # Gestion des apostrophes
    texte = texte.replace('\u2019', "'").replace('\u2018', "'")
    texte = texte.replace('\u2032', "'")
    
    # Suppression des césures
    texte = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', texte)

    # Gestion des listes
    texte = re.sub(r'^\s*[-•·]\s*', '', texte, flags=re.MULTILINE)

    # Gestion des numéros de pages s'il y en a 
    texte = re.sub(r'^\s*\d+\s*$', '', texte, flags=re.MULTILINE)

    return texte.strip()


def detecter_textes_similaires(df, col_texte, col_parti, col_id, seuil=0.90):
    """
    Détecte les textes identiques et similaires, par parti politique.
    
    Args:
        df : DataFrame avec les textes
        col_texte : nom de la colonne contenant les textes
        col_parti : nom de la colonne du parti politique
        col_id : nom de la colonne d'identifiant de la profession de foi
        seuil : seuil de similarité (0.90 = 90%)
    
    Returns:
        dict avec les résultats par parti
    """
    resultats = {}
    
    for parti, groupe in df.groupby(col_parti):
        textes = groupe[col_texte].fillna("").tolist()
        ids = groupe[col_id].tolist()
        n = len(textes)
        
        if n < 2:
            continue
        
        # Doublons
        hashes = [hashlib.md5(t.encode()).hexdigest() for t in textes]
        hash_counts = pd.Series(hashes).value_counts()
        nb_doublons_exacts = int((hash_counts > 1).sum())
        
        # Similarité TF-IDF
        vectorizer = TfidfVectorizer(min_df=1)
        tfidf_matrix = vectorizer.fit_transform(textes)
        similarites = cosine_similarity(tfidf_matrix)
        
        # Récupère les paires similaires (triangle supérieur uniquement)
        paires_similaires = []
        for i in range(n):
            for j in range(i + 1, n):
                score = similarites[i, j]
                if score >= seuil:
                    paires_similaires.append({
                        "id_1": ids[i],
                        "id_2" : ids[j],
                        "similarite" : round(score, 3),
                        "texte_1" : textes[i][:80] + "...",  # aperçu
                        "texte_2" : textes[j][:80] + "...",
                    })
        
        resultats[parti] = {
            "nb_textes" : n,
            "nb_doublons_exacts" : nb_doublons_exacts,
            "nb_paires_similaires": len(paires_similaires)
        }
    
    return resultats


def resumer_resultats(resultats):
    """Produit un DataFrame de synthèse par parti."""
    rows = []
    for parti, res in resultats.items():
        rows.append({
            "parti" : parti,
            "nb_textes" : res["nb_textes"],
            "nb_doublons_exacts"  : res["nb_doublons_exacts"],
            "nb_paires_similaires": res["nb_paires_similaires"],
            "taux_similarite" : round(res["nb_paires_similaires"] / (res["nb_textes"]**2-res["nb_textes"]), 2)
        })
    return pd.DataFrame(rows).sort_values("nb_paires_similaires", ascending=False)

docs = []

with zipfile.ZipFile(ZIP_PATH) as z:
    for file in z.namelist():
        if file.endswith(".txt"):
            with z.open(file) as f:
                text = f.read().decode("utf-8", errors="ignore")
                docs.append({"file": file, "text": text})

transcriptions = pd.DataFrame(docs)
transcriptions = transcriptions.assign(
    id=transcriptions["file"].str.extract(r'([^/]+)\.txt'),
    annee=transcriptions["file"].str.extract(r'text_files/(\d{4})')
)
transcriptions = transcriptions.drop(columns = ["file"])
transcriptions = transcriptions.merge(data[['id', 'titulaire-soutien']], on='id', how='left')
transcriptions['parti_synthetique']=transcriptions['titulaire-soutien'].apply(classifier_parti)
transcriptions['texte_nettoye'] = transcriptions['text'].apply(nettoyage_profession_foi)

resultats_similarites = detecter_textes_similaires(transcriptions, "texte_nettoye", "parti_synthetique", "id", 0.90)
df_similarites = resumer_resultats(resultats_similarites)