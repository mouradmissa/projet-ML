"""
Pipeline complet KNN pour la recommandation de priorité (BASELINE FAIBLE)
Objectif: Prédire Priority (Low, Medium, High) - Comparaison avec Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Chemins portables (racine ML = parent de "Recommandation priorite")
MODEL_DIR = Path(__file__).resolve().parent
ML_ROOT = MODEL_DIR.parent.parent.parent
DATA_CANDIDATES = [
    ML_ROOT / "Project-Management-2-enriched.csv",
    ML_ROOT / "Project-Management-2.csv",
]

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
K_RANGE = range(1, 21)  # Tester k de 1 à 20

print("=" * 80)
print("KNN - RECOMMANDATION DE PRIORITÉ (BASELINE)")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT ET INSPECTION
# ============================================================================
print("\n" + "=" * 80)
print("1. CHARGEMENT ET INSPECTION DES DONNÉES")
print("=" * 80)

data_path = next((p for p in DATA_CANDIDATES if p.is_file()), None)
if data_path is None:
    raise FileNotFoundError(
        "CSV introuvable. Placez Project-Management-2-enriched.csv à la racine du dossier ML."
    )
df = pd.read_csv(data_path)

print(f"\n✓ Dataset chargé avec succès: {len(df)} lignes, {len(df.columns)} colonnes")

# Afficher les informations de base
print("\n--- Informations du DataFrame ---")
print(df.info())

print("\n--- Aperçu des premières lignes ---")
print(df.head())

print("\n--- Statistiques descriptives ---")
print(df.describe())

print("\n--- Valeurs manquantes ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante")

print("\n--- Doublons ---")
duplicates = df.duplicated().sum()
print(f"Nombre de doublons: {duplicates}")

print("\n--- Distribution de la cible 'Priority' ---")
print(df['Priority'].value_counts())
print("\nProportions:")
print(df['Priority'].value_counts(normalize=True))

# ============================================================================
# 2. NETTOYAGE DES DONNÉES (IDENTIQUE)
# ============================================================================
print("\n" + "=" * 80)
print("2. NETTOYAGE DES DONNÉES")
print("=" * 80)

# Copie pour le nettoyage
df_clean = df.copy()

# 2.1 Supprimer les doublons
before_dup = len(df_clean)
df_clean = df_clean.drop_duplicates()
after_dup = len(df_clean)
print(f"\n✓ Doublons supprimés: {before_dup - after_dup} lignes")

# 2.2 Imputation des valeurs manquantes
numeric_cols = ['Progress', 'Budget', 'Planned_Duration_Days']
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"✓ {col}: valeurs manquantes imputées avec la médiane ({median_val})")

if df_clean['Priority'].isnull().sum() > 0:
    mode_val = df_clean['Priority'].mode()[0]
    df_clean['Priority'].fillna(mode_val, inplace=True)
    print(f"✓ Priority: valeurs manquantes imputées avec le mode ({mode_val})")

# 2.3 Supprimer les colonnes constantes
const_cols = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
if const_cols:
    df_clean = df_clean.drop(columns=const_cols)
    print(f"\n✓ Colonnes constantes supprimées: {const_cols}")
else:
    print("\n✓ Aucune colonne constante détectée")

# 2.4 Traitement des outliers (IQR)
def cap_outliers_iqr(df, column):
    """Plafonner les outliers avec la méthode IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return outliers_count, lower_bound, upper_bound

print("\n--- Traitement des outliers (IQR) ---")
for col in ['Budget', 'Planned_Duration_Days']:
    count, lower, upper = cap_outliers_iqr(df_clean, col)
    print(f"✓ {col}: {count} outliers plafonnés (bornes: [{lower:.2f}, {upper:.2f}])")

print(f"\n✓ Dataset après nettoyage: {len(df_clean)} lignes")

# ============================================================================
# 3. ENCODAGE DE LA CIBLE (IDENTIQUE)
# ============================================================================
print("\n" + "=" * 80)
print("3. ENCODAGE DE LA CIBLE")
print("=" * 80)

# Priority est ordinale: Low < Medium < High
priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df_clean['Priority_encoded'] = df_clean['Priority'].map(priority_mapping)

print("\nMapping de Priority:")
for key, value in priority_mapping.items():
    count = (df_clean['Priority_encoded'] == value).sum()
    print(f"  {key} → {value} ({count} instances)")

print(f"\n✓ Encodage réussi")

# ============================================================================
# 4. SÉPARATION X / y (IDENTIQUE)
# ============================================================================
print("\n" + "=" * 80)
print("4. SÉPARATION DES FEATURES ET DE LA CIBLE")
print("=" * 80)

feature_cols = ['Progress', 'Budget', 'Planned_Duration_Days']
X = df_clean[feature_cols].copy()
y = df_clean['Priority_encoded'].copy()

print(f"\n✓ Features (X): {X.shape}")
print(f"  Colonnes: {feature_cols}")
print(f"\n✓ Cible (y): {y.shape}")

# ============================================================================
# 5. DIVISION TRAIN/TEST (données brutes, puis scaling seulement sur le train)
# ============================================================================
print("\n" + "=" * 80)
print("5. DIVISION TRAIN/TEST")
print("=" * 80)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X.values, y.values,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nOK Division effectuee (test_size={TEST_SIZE}, stratifiee)")
print(f"\nTrain set: {X_train_raw.shape[0]} samples")
print("Distribution des classes dans train:")
print(pd.Series(y_train).value_counts().sort_index())
print(f"\nTest set: {X_test_raw.shape[0]} samples")
print("Distribution des classes dans test:")
print(pd.Series(y_test).value_counts().sort_index())

# ============================================================================
# 6. SCALING (fit sur train uniquement — obligatoire pour KNN)
# ============================================================================
print("\n" + "=" * 80)
print("6. SCALING DES FEATURES (StandardScaler sur train uniquement)")
print("=" * 80)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print("OK Scaler ajuste sur le jeu d'entrainement; train et test transformes.")

# ============================================================================
# 7. RECHERCHE DU MEILLEUR K
# ============================================================================
print("\n" + "=" * 80)
print("7. RECHERCHE DU MEILLEUR K")
print("=" * 80)

print(f"\nTester k de {K_RANGE.start} à {K_RANGE.stop - 1}")
print(f"Validation croisée: {CV_FOLDS}-fold")
print(f"Métrique: F1-score macro\n")

# Stocker les scores pour chaque k
k_scores = []
k_values = list(K_RANGE)

print("k   | F1-macro (CV) | Std")
print("-" * 35)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Validation croisée avec F1-score macro
    cv_scores = cross_val_score(
        knn, X_train, y_train, 
        cv=CV_FOLDS, 
        scoring='f1_macro'
    )
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    k_scores.append(mean_score)
    
    print(f"{k:2d}  | {mean_score:.4f}        | {std_score:.4f}")

# Trouver le meilleur k
best_k_idx = np.argmax(k_scores)
best_k = k_values[best_k_idx]
best_k_score = k_scores[best_k_idx]

print("\n" + "=" * 35)
print(f"✓ Meilleur k: {best_k}")
print(f"✓ F1-score macro (CV): {best_k_score:.4f}")
print("=" * 35)

# Visualisation des scores en fonction de k
plt.figure(figsize=(12, 6))
plt.plot(k_values, k_scores, marker='o', linewidth=2, markersize=8)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Meilleur k = {best_k}')
plt.xlabel('Nombre de voisins (k)', fontsize=12)
plt.ylabel('F1-score macro (validation croisée)', fontsize=12)
plt.title('Recherche du meilleur k pour KNN', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / 'knn_priority_k_search.png', dpi=150)
print("\n✓ Graphique de recherche de k sauvegardé: knn_priority_k_search.png")

# ============================================================================
# 8. ENTRAÎNEMENT DU MODÈLE FINAL
# ============================================================================
print("\n" + "=" * 80)
print("8. ENTRAÎNEMENT DU MODÈLE FINAL")
print("=" * 80)

# Entraîner avec le meilleur k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

print(f"\n✓ Modèle KNN entraîné avec k={best_k}")

# ============================================================================
# 9. ÉVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("9. ÉVALUATION DU MODÈLE")
print("=" * 80)

# Prédictions
y_pred = best_knn.predict(X_test)

# Métriques globales
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Accuracy: {accuracy:.4f}")

# Précision, rappel, F1-score par classe et macro
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, zero_division=0
)

print("\n--- Métriques par classe ---")
class_names = ['Low (0)', 'Medium (1)', 'High (2)']
for i, class_name in enumerate(class_names):
    print(f"\n{class_name}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall:    {recall[i]:.4f}")
    print(f"  F1-score:  {f1[i]:.4f}")
    print(f"  Support:   {support[i]}")

# Métriques macro
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro', zero_division=0
)

print("\n--- Métriques macro (moyenne non pondérée) ---")
print(f"Precision macro: {macro_precision:.4f}")
print(f"Recall macro:    {macro_recall:.4f}")
print(f"F1-score macro:  {macro_f1:.4f}")

# Rapport de classification complet
print("\n--- Rapport de classification complet ---")
print(classification_report(
    y_test, y_pred, 
    target_names=class_names,
    zero_division=0
))

# Matrice de confusion
print("\n--- Matrice de confusion ---")
cm = confusion_matrix(y_test, y_pred)
print("\n     Predicted")
print("         Low  Med  High")
print(f"Low     {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}")
print(f"Med     {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}")
print(f"High    {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}")

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Matrice de confusion - KNN Priority')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'knn_priority_confusion_matrix.png', dpi=150)
print("\n✓ Matrice de confusion sauvegardée: knn_priority_confusion_matrix.png")

# ============================================================================
# 10. SAUVEGARDE DU MODÈLE
# ============================================================================
print("\n" + "=" * 80)
print("10. SAUVEGARDE DU MODÈLE")
print("=" * 80)

# Sauvegarder le modèle
model_path = MODEL_DIR / 'knn_priority_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_knn, f)
print(f"\nOK Modele sauvegarde: knn_priority_model.pkl")

# Sauvegarder le scaler
scaler_path = MODEL_DIR / 'knn_priority_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"OK Scaler sauvegarde: knn_priority_scaler.pkl")

# Sauvegarder les features
features_path = MODEL_DIR / 'knn_priority_features.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"OK Features sauvegardees: knn_priority_features.pkl")

# Sauvegarder les métadonnées
meta_path = MODEL_DIR / 'knn_priority_meta.pkl'
metadata = {
    'priority_mapping': priority_mapping,
    'class_names': class_names,
    'best_k': best_k,
    'best_k_score': best_k_score,
    'f1_macro': macro_f1,
    'accuracy': accuracy
}
with open(meta_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Métadonnées sauvegardées: knn_priority_meta.pkl")

# ============================================================================
# 11. RÉSUMÉ ET COMPARAISON
# ============================================================================
print("\n" + "=" * 80)
print("11. RÉSUMÉ FINAL")
print("=" * 80)

print(f"\n✓ DATASET")
print(f"  Lignes après nettoyage: {len(df_clean)}")
print(f"  Features utilisées: {feature_cols}")
print(f"  Scaling: StandardScaler (OBLIGATOIRE pour KNN)")

print(f"\n✓ DISTRIBUTION DES CLASSES")
print(f"  Low (0):    {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"  Medium (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"  High (2):   {(y == 2).sum()} ({(y == 2).sum()/len(y)*100:.1f}%)")

print(f"\n✓ HYPERPARAMÈTRE RETENU")
print(f"  Meilleur k: {best_k}")
print(f"  F1-score (CV): {best_k_score:.4f}")

print(f"\n✓ PERFORMANCES SUR TEST SET")
print(f"  Accuracy:        {accuracy:.4f}")
print(f"  F1-score macro:  {macro_f1:.4f}")
print(f"  Precision macro: {macro_precision:.4f}")
print(f"  Recall macro:    {macro_recall:.4f}")

print("\n" + "=" * 80)
print("12. POURQUOI KNN EST UNE BASELINE FAIBLE")
print("=" * 80)

print("""
KNN présente plusieurs limitations par rapport à Random Forest:

1. SENSIBILITÉ À L'ÉCHELLE
   - KNN utilise la distance euclidienne entre points
   - Nécessite un scaling obligatoire des features
   - Random Forest n'est pas affecté par l'échelle

2. ABSENCE DE SEUILS NATURELS
   - KNN vote parmi les k voisins les plus proches
   - Ne capture pas les seuils de décision complexes
   - Random Forest crée des règles de décision basées sur des seuils

3. DIFFICULTÉ AVEC LES CLASSES DÉSÉQUILIBRÉES
   - Dans ce dataset: High=52%, Medium=30%, Low=18%
   - KNN a tendance à favoriser la classe majoritaire
   - Random Forest peut mieux équilibrer via le bootstrapping

4. PAS D'INTERPRÉTABILITÉ
   - KNN ne fournit pas d'importance des features
   - Impossible de comprendre quels facteurs influencent la priorité
   - Random Forest donne l'importance de chaque variable

5. SENSIBILITÉ AU BRUIT
   - KNN peut être perturbé par des points aberrants
   - Random Forest est robuste grâce à l'agrégation d'arbres

6. SCALABILITÉ
   - KNN doit calculer la distance à tous les points d'entraînement
   - Lent pour de grands datasets
   - Random Forest est plus efficace en prédiction

Ces limitations font de KNN une baseline faible, utile pour
quantifier l'amélioration apportée par Random Forest.
""")

print("\n" + "=" * 80)
print("✓ PIPELINE KNN TERMINÉ AVEC SUCCÈS!")
print("=" * 80)
