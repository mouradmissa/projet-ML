"""
Pipeline complet Random Forest pour la recommandation de priorité
Objectif: Prédire Priority (Low, Medium, High) à partir de Progress, Budget, Planned_Duration_Days
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

print("=" * 80)
print("RANDOM FOREST - RECOMMANDATION DE PRIORITÉ")
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
# 2. NETTOYAGE DES DONNÉES
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
# (Pas nécessaire ici car aucune valeur manquante, mais inclus pour la complétude)
numeric_cols = ['Progress', 'Budget', 'Planned_Duration_Days']
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"✓ {col}: valeurs manquantes imputées avec la médiane ({median_val})")

# Imputation de Priority (mode) si nécessaire
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

# 2.4 Traitement des outliers (IQR) sur Budget et Planned_Duration_Days
def cap_outliers_iqr(df, column):
    """Plafonner les outliers avec la méthode IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Compter les outliers avant traitement
    outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    
    # Plafonner
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return outliers_count, lower_bound, upper_bound

print("\n--- Traitement des outliers (IQR) ---")
for col in ['Budget', 'Planned_Duration_Days']:
    count, lower, upper = cap_outliers_iqr(df_clean, col)
    print(f"✓ {col}: {count} outliers plafonnés (bornes: [{lower:.2f}, {upper:.2f}])")

print(f"\n✓ Dataset après nettoyage: {len(df_clean)} lignes")

# ============================================================================
# 3. ENCODAGE DE LA CIBLE
# ============================================================================
print("\n" + "=" * 80)
print("3. ENCODAGE DE LA CIBLE")
print("=" * 80)

# Priority ordinale: Low=0, Medium=1, High=2 (ordre explicite)
priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df_clean['Priority_encoded'] = df_clean['Priority'].map(priority_mapping)

print("\nMapping de Priority:")
for key, value in priority_mapping.items():
    count = (df_clean['Priority_encoded'] == value).sum()
    print(f"  {key} → {value} ({count} instances)")

# Vérification
print(f"\n✓ Encodage réussi")
print(f"Distribution de Priority_encoded:")
print(df_clean['Priority_encoded'].value_counts().sort_index())

# ============================================================================
# 4. SÉPARATION X / y
# ============================================================================
print("\n" + "=" * 80)
print("4. SÉPARATION DES FEATURES ET DE LA CIBLE")
print("=" * 80)

# Sélectionner les features
feature_cols = ['Progress', 'Budget', 'Planned_Duration_Days']
X = df_clean[feature_cols].copy()
y = df_clean['Priority_encoded'].copy()

print(f"\n✓ Features (X): {X.shape}")
print(f"  Colonnes: {feature_cols}")
print(f"\n✓ Cible (y): {y.shape}")
print(f"  Distribution:")
print(y.value_counts().sort_index())

# ============================================================================
# 5. DIVISION TRAIN/TEST
# ============================================================================
print("\n" + "=" * 80)
print("5. DIVISION TRAIN/TEST")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y  # Stratification pour garder les proportions des classes
)

print(f"\n✓ Division effectuée (test_size={TEST_SIZE}, stratifiée)")
print(f"\nTrain set: {X_train.shape[0]} samples")
print("Distribution des classes dans train:")
print(y_train.value_counts().sort_index())
print(f"\nTest set: {X_test.shape[0]} samples")
print("Distribution des classes dans test:")
print(y_test.value_counts().sort_index())

# ============================================================================
# 6. SCALING (OPTIONNEL POUR RANDOM FOREST)
# ============================================================================
print("\n" + "=" * 80)
print("6. SCALING (non appliqué pour Random Forest)")
print("=" * 80)
print("\n✓ Random Forest n'est pas sensible à l'échelle des features")
print("  Scaling non nécessaire (contrairement à KNN)")

# ============================================================================
# 7. ENTRAÎNEMENT ET OPTIMISATION
# ============================================================================
print("\n" + "=" * 80)
print("7. ENTRAÎNEMENT ET OPTIMISATION (GRIDSEARCHCV)")
print("=" * 80)

# Modèle de base
print("\n--- Modèle de base ---")
rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
f1_base = f1_score(y_test, y_pred_base, average='macro')
print(f"✓ F1-score macro (modèle de base): {f1_base:.4f}")

# Optimisation avec GridSearchCV
print("\n--- Optimisation des hyperparamètres ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

print(f"Grille de recherche:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

# GridSearchCV avec F1-score macro et 3-fold CV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=param_grid,
    cv=CV_FOLDS,
    scoring='f1_macro',  # Optimiser le F1-score macro
    n_jobs=-1,
    verbose=1
)

print(f"\nEntraînement en cours (validation croisée {CV_FOLDS}-fold)...")
grid_search.fit(X_train, y_train)

print(f"\n✓ Optimisation terminée")
print(f"\nMeilleurs hyperparamètres:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nMeilleur F1-score macro (CV): {grid_search.best_score_:.4f}")

# Meilleur modèle
best_rf = grid_search.best_estimator_

# ============================================================================
# 8. ÉVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("8. ÉVALUATION DU MODÈLE OPTIMISÉ")
print("=" * 80)

# Prédictions
y_pred = best_rf.predict(X_test)

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Matrice de confusion - Random Forest Priority')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'rf_priority_confusion_matrix.png', dpi=150)
print("\n✓ Matrice de confusion sauvegardée: rf_priority_confusion_matrix.png")

# Importance des features
print("\n--- Importance des features ---")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Visualisation de l'importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Importance des features - Random Forest Priority')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'rf_priority_feature_importance.png', dpi=150)
print("\n✓ Importance des features sauvegardée: rf_priority_feature_importance.png")

# ============================================================================
# 9. SAUVEGARDE DU MODÈLE
# ============================================================================
print("\n" + "=" * 80)
print("9. SAUVEGARDE DU MODÈLE")
print("=" * 80)

# Sauvegarder le modèle
model_path = MODEL_DIR / 'rf_priority_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_rf, f)
print(f"\nOK Modele sauvegarde: rf_priority_model.pkl")

# Sauvegarder les features
features_path = MODEL_DIR / 'rf_priority_features.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"OK Features sauvegardees: rf_priority_features.pkl")

# Sauvegarder les métadonnées
meta_path = MODEL_DIR / 'rf_priority_meta.pkl'
metadata = {
    'priority_mapping': priority_mapping,
    'class_names': class_names,
    'best_params': grid_search.best_params_,
    'f1_macro': macro_f1,
    'accuracy': accuracy,
    'feature_importance': feature_importance.to_dict('records')
}
with open(meta_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Métadonnées sauvegardées: rf_priority_meta.pkl")

# ============================================================================
# 10. RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 80)
print("10. RÉSUMÉ FINAL")
print("=" * 80)

print(f"\n✓ DATASET")
print(f"  Lignes après nettoyage: {len(df_clean)}")
print(f"  Features utilisées: {feature_cols}")

print(f"\n✓ DISTRIBUTION DES CLASSES")
print(f"  Low (0):    {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"  Medium (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"  High (2):   {(y == 2).sum()} ({(y == 2).sum()/len(y)*100:.1f}%)")

print(f"\n✓ HYPERPARAMÈTRES RETENUS")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n✓ PERFORMANCES")
print(f"  Accuracy:       {accuracy:.4f}")
print(f"  F1-score macro: {macro_f1:.4f}")
print(f"  Precision macro: {macro_precision:.4f}")
print(f"  Recall macro:    {macro_recall:.4f}")

print(f"\n✓ IMPORTANCE DES VARIABLES")
for _, row in feature_importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

print("\n" + "=" * 80)
print("✓ PIPELINE TERMINÉ AVEC SUCCÈS!")
print("=" * 80)
