"""
COMPARAISON DÉTAILLÉE: Random Forest vs KNN
Pour la recommandation de priorité (Priority: Low, Medium, High)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

_BASE = Path(__file__).resolve().parent

print("=" * 80)
print("COMPARAISON RANDOM FOREST vs KNN - RECOMMANDATION DE PRIORITÉ")
print("=" * 80)

# ============================================================================
# 1. RÉSUMÉ DES PERFORMANCES
# ============================================================================
print("\n" + "=" * 80)
print("1. RÉSUMÉ DES PERFORMANCES")
print("=" * 80)

results = {
    'Modèle': ['Random Forest', 'KNN'],
    'Accuracy': [0.5333, 0.4167],
    'F1-score macro': [0.4040, 0.4131],
    'Precision macro': [0.4404, 0.4321],
    'Recall macro': [0.4143, 0.4482]
}

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# Différences
print("\n--- Différences (Random Forest - KNN) ---")
print(f"Accuracy:        {0.5333 - 0.4167:+.4f} (+11.66 points de pourcentage)")
print(f"F1-score macro:  {0.4040 - 0.4131:-.4f} (-0.91 points de pourcentage)")
print(f"Precision macro: {0.4404 - 0.4321:+.4f} (+0.83 points de pourcentage)")
print(f"Recall macro:    {0.4143 - 0.4482:-.4f} (-3.39 points de pourcentage)")

# ============================================================================
# 2. ANALYSE PAR CLASSE
# ============================================================================
print("\n" + "=" * 80)
print("2. PERFORMANCES PAR CLASSE")
print("=" * 80)

print("\n--- LOW (classe minoritaire) ---")
rf_low = {'Precision': 0.2857, 'Recall': 0.1818, 'F1': 0.2222}
knn_low = {'Precision': 0.3158, 'Recall': 0.5455, 'F1': 0.4000}
print(f"Random Forest: P={rf_low['Precision']:.4f}, R={rf_low['Recall']:.4f}, F1={rf_low['F1']:.4f}")
print(f"KNN:           P={knn_low['Precision']:.4f}, R={knn_low['Recall']:.4f}, F1={knn_low['F1']:.4f}")
print(f"Différence F1: {knn_low['F1'] - rf_low['F1']:+.4f} (KNN meilleur sur LOW)")

print("\n--- MEDIUM ---")
rf_med = {'Precision': 0.4444, 'Recall': 0.2222, 'F1': 0.2963}
knn_med = {'Precision': 0.3333, 'Recall': 0.4444, 'F1': 0.3810}
print(f"Random Forest: P={rf_med['Precision']:.4f}, R={rf_med['Recall']:.4f}, F1={rf_med['F1']:.4f}")
print(f"KNN:           P={knn_med['Precision']:.4f}, R={knn_med['Recall']:.4f}, F1={knn_med['F1']:.4f}")
print(f"Différence F1: {knn_med['F1'] - rf_med['F1']:+.4f} (KNN meilleur sur MEDIUM)")

print("\n--- HIGH (classe majoritaire) ---")
rf_high = {'Precision': 0.5909, 'Recall': 0.8387, 'F1': 0.6933}
knn_high = {'Precision': 0.6471, 'Recall': 0.3548, 'F1': 0.4583}
print(f"Random Forest: P={rf_high['Precision']:.4f}, R={rf_high['Recall']:.4f}, F1={rf_high['F1']:.4f}")
print(f"KNN:           P={knn_high['Precision']:.4f}, R={knn_high['Recall']:.4f}, F1={knn_high['F1']:.4f}")
print(f"Différence F1: {knn_high['F1'] - rf_high['F1']:-.4f} (RF meilleur sur HIGH)")

# ============================================================================
# 3. VISUALISATION DES COMPARAISONS
# ============================================================================
print("\n" + "=" * 80)
print("3. GÉNÉRATION DES GRAPHIQUES DE COMPARAISON")
print("=" * 80)

# Graphique 1: Comparaison des métriques globales
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'F1-macro', 'Precision\nmacro', 'Recall\nmacro']
rf_scores = [0.5333, 0.4040, 0.4404, 0.4143]
knn_scores = [0.4167, 0.4131, 0.4321, 0.4482]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='#2E86AB')
bars2 = ax.bar(x + width/2, knn_scores, width, label='KNN', color='#F77F00')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparaison des performances globales', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.7)

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(_BASE / 'comparison_global_metrics.png', dpi=150)
print("✓ Graphique sauvegardé: comparison_global_metrics.png")
plt.close()

# Graphique 2: F1-score par classe
fig, ax = plt.subplots(figsize=(10, 6))
classes = ['Low', 'Medium', 'High']
rf_f1_per_class = [0.2222, 0.2963, 0.6933]
knn_f1_per_class = [0.4000, 0.3810, 0.4583]

x = np.arange(len(classes))
bars1 = ax.bar(x - width/2, rf_f1_per_class, width, label='Random Forest', color='#2E86AB')
bars2 = ax.bar(x + width/2, knn_f1_per_class, width, label='KNN', color='#F77F00')

ax.set_ylabel('F1-score', fontsize=12)
ax.set_title('F1-score par classe de priorité', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(_BASE / 'comparison_f1_per_class.png', dpi=150)
print("✓ Graphique sauvegardé: comparison_f1_per_class.png")
plt.close()

# Graphique 3: Precision vs Recall par classe
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest
classes_plot = ['Low', 'Medium', 'High']
rf_precision = [0.2857, 0.4444, 0.5909]
rf_recall = [0.1818, 0.2222, 0.8387]

axes[0].scatter(rf_recall, rf_precision, s=200, c=['#E63946', '#F77F00', '#06A77D'], alpha=0.7)
for i, txt in enumerate(classes_plot):
    axes[0].annotate(txt, (rf_recall[i], rf_precision[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=12)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('Recall', fontsize=12)
axes[0].set_ylabel('Precision', fontsize=12)
axes[0].set_title('Random Forest', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-0.05, 1.05)
axes[0].set_ylim(-0.05, 1.05)

# KNN
knn_precision = [0.3158, 0.3333, 0.6471]
knn_recall = [0.5455, 0.4444, 0.3548]

axes[1].scatter(knn_recall, knn_precision, s=200, c=['#E63946', '#F77F00', '#06A77D'], alpha=0.7)
for i, txt in enumerate(classes_plot):
    axes[1].annotate(txt, (knn_recall[i], knn_precision[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=12)
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('KNN', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-0.05, 1.05)
axes[1].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(_BASE / 'comparison_precision_recall.png', dpi=150)
print("✓ Graphique sauvegardé: comparison_precision_recall.png")
plt.close()

# ============================================================================
# 4. ANALYSE DES MATRICES DE CONFUSION
# ============================================================================
print("\n" + "=" * 80)
print("4. ANALYSE DES MATRICES DE CONFUSION")
print("=" * 80)

print("\n--- Random Forest ---")
print("         Predicted")
print("         Low  Med  High")
print("Low        2    2    7    (confond souvent Low avec High)")
print("Med        3    4   11    (confond Medium avec High)")
print("High       2    3   26    (bonne détection de High)")

print("\n--- KNN ---")
print("         Predicted")
print("         Low  Med  High")
print("Low        6    3    2    (meilleure détection de Low)")
print("Med        6    8    4    (répartition plus équilibrée)")
print("High       7   13   11    (confond High avec Medium)")

print("\n--- Observation clé ---")
print("• Random Forest: forte tendance à prédire HIGH (classe majoritaire)")
print("  → Recall élevé sur HIGH (0.84) mais faible sur LOW et MEDIUM")
print("  → Biais vers la classe majoritaire")
print("\n• KNN: prédictions plus équilibrées entre les classes")
print("  → Meilleur recall sur LOW (0.55) et MEDIUM (0.44)")
print("  → Mais perd en performance sur HIGH (recall 0.35)")

# ============================================================================
# 5. AVANTAGES ET INCONVÉNIENTS
# ============================================================================
print("\n" + "=" * 80)
print("5. AVANTAGES ET INCONVÉNIENTS")
print("=" * 80)

print("\n--- RANDOM FOREST ---")
print("✓ Avantages:")
print("  • Meilleure accuracy globale (53.3% vs 41.7%)")
print("  • Excellente performance sur la classe majoritaire (HIGH)")
print("  • Fournit l'importance des features (Budget = 39%, Duration = 34%, Progress = 26%)")
print("  • Pas de scaling nécessaire")
print("  • Robuste aux outliers")
print("\n✗ Inconvénients:")
print("  • Faible performance sur les classes minoritaires (LOW, MEDIUM)")
print("  • Biais fort vers la classe majoritaire")
print("  • F1-macro légèrement inférieur à KNN")

print("\n--- KNN ---")
print("✓ Avantages:")
print("  • Meilleur F1-macro (0.413 vs 0.404)")
print("  • Meilleures performances sur LOW et MEDIUM")
print("  • Prédictions plus équilibrées entre classes")
print("  • Meilleur recall macro (0.448 vs 0.414)")
print("\n✗ Inconvénients:")
print("  • Accuracy plus faible (41.7% vs 53.3%)")
print("  • Nécessite un scaling obligatoire")
print("  • Pas d'interprétabilité (pas d'importance des features)")
print("  • Mauvaise performance sur HIGH")
print("  • Sensible au choix de k")

# ============================================================================
# 6. RECOMMANDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("6. RECOMMANDATIONS")
print("=" * 80)

print("""
Choix du modèle selon le contexte:

1. Si ACCURACY GLOBALE est prioritaire:
   → Utiliser RANDOM FOREST
   • Meilleure précision générale (53%)
   • Bon pour classifier la majorité des tâches
   
2. Si ÉQUITÉ ENTRE CLASSES est importante:
   → Utiliser KNN
   • Meilleur F1-macro (traite toutes les classes de manière plus équitable)
   • Crucial si les classes minoritaires sont importantes (ex: détecter les LOW)
   
3. Si INTERPRÉTABILITÉ est nécessaire:
   → Utiliser RANDOM FOREST
   • Fournit l'importance des features
   • Permet de comprendre quels facteurs influencent la priorité
   • Budget et durée planifiée sont les facteurs les plus importants

4. Pour AMÉLIORER les performances:
   • Collecter plus de données (300 lignes est limité)
   • Ajouter des features pertinentes (Risk_Level, Task_Status, etc.)
   • Tester des techniques de rééquilibrage (SMOTE, class_weight)
   • Essayer des modèles plus avancés (XGBoost, LightGBM)
   • Optimiser les seuils de décision pour chaque classe

CONCLUSION:
Les deux modèles ont des performances modérées mais complémentaires.
Random Forest est plus robuste globalement, mais KNN est meilleur pour
détecter les classes minoritaires. Le choix dépend des priorités métier.
""")

print("\n" + "=" * 80)
print("✓ ANALYSE DE COMPARAISON TERMINÉE")
print("=" * 80)
