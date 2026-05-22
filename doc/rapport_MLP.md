# 🧠 Rapport Modélisation Deep Learning : Multi-Layer Perceptron (MLP)

## Système Intelligent Multi-Modèles pour la Rétention Client

> **Projet :** M1 Data Science - Certification RNCP40875 (Bloc 2)  
> **Date :** Mai 2026  
> **Framework :** PyTorch

---

## 1. Architecture et Choix Techniques

### 1.1 Architecture du réseau

Le modèle MLP implémenté est un réseau de neurones dense à **3 couches cachées** :

```
Input (43 features) → 128 → 64 → 32 → 1 (output)
```

**Justification :** 
- Architecture progressive décroissante permettant une compression hiérarchique de l'information
- Profondeur suffisante pour capturer des **interactions non-linéaires** entre variables comportementales, financières et contractuelles
- Pas de sur-complexification : compromis entre capacité d'apprentissage et risque d'overfitting

**Paramètres entraînables :** ~13 000 paramètres

### 1.2 Fonction d'activation et sortie

- **Couches cachées :** ReLU (Rectified Linear Unit)
  - Standard pour réseaux denses
  - Évite le problème de gradient qui s'annule (vanishing gradient)
  
- **Couche de sortie :** Logit (sortie linéaire sans activation)
  - La sortie brute du réseau est un **logit** (valeur réelle non bornée)
  - Interprétation : `sigmoid(logit) = probabilité de churn`
  - Sigmoid appliqué uniquement lors de l'inférence pour obtenir `P(churn) ∈ [0, 1]`

### 1.3 Fonction de perte : BCEWithLogitsLoss

**Choix :** `BCEWithLogitsLoss` avec `pos_weight`

**Pourquoi cette loss ?**

1. **Combinaison efficace :** sigmoid + Binary Cross-Entropy en une seule opération
   - Plus stable numériquement que sigmoid séparé
   - Évite les problèmes d'overflow/underflow

2. **Gestion du déséquilibre via `pos_weight`** ⭐ (CRITIQUE)
   - Dataset : 90% fidèles / 10% churners → ratio 9:1
   - `pos_weight = n_neg / n_pos ≈ 9.0`
   - **Effet :** La loss pénalise **9x plus fort** les erreurs sur les churners
   - **Équivalent :** `class_weight='balanced'` dans sklearn
   - **Objectif :** Forcer le modèle à ne pas ignorer la classe minoritaire

Sans `pos_weight`, le modèle apprendrait simplement à prédire "fidèle" pour tous les clients (accuracy 90% sans valeur prédictive).

---

## 2. Stratégies de Régularisation et Optimisation

### 2.1 Techniques de régularisation

#### BatchNorm (Batch Normalization)
- Appliqué **après chaque couche dense**
- Normalise les activations à chaque mini-batch
- **Bénéfices :** Stabilise et accélère la convergence, réduit la sensibilité à l'initialisation

#### Dropout (0.3)
- Désactive aléatoirement **30% des neurones** à chaque passe d'entraînement
- **Effet :** Réduit l'overfitting en empêchant les co-adaptations entre neurones
- Appliqué après chaque activation ReLU

#### Weight Decay (1e-4)
- Régularisation L2 intégrée dans l'optimiseur Adam
- Pénalise les poids trop grands

### 2.2 Optimiseur et Learning Rate Scheduling

**Optimiseur :** Adam (lr=1e-3)
- Convergence rapide et adaptative
- Bien adapté aux données tabulaires

**ReduceLROnPlateau**
- Surveillance de la `val_loss`
- Si pas d'amélioration pendant **5 époques** : `lr × 0.5`
- **Effet :** Affine la convergence en fin d'entraînement

### 2.3 Hyperparamètres d'entraînement

- **Nombre d'époques :** 60
- **Batch size :** 256
- **Seuil de décision :** 0.5
- **Device :** CPU (ou GPU si disponible)

---

## 3. Exploration : Under-Sampling

### 3.1 Motivation

Tester une approche alternative à `pos_weight` pour gérer le déséquilibre :
- **Stratégie :** Réduire artificiellement le nombre de clients fidèles dans le train set
- **Implémentation :** Conservation de 50% des non-churners (échantillonnage aléatoire)
- **Nouveau ratio :** ~82% fidèles / ~18% churners (vs 90/10 initial)

### 3.2 Résultats

**Observation :** Performances **similaires** au modèle entraîné sur le dataset complet.

**Conclusion :**
- L'**under-sampling** ne dégrade pas les performances, mais n'apporte pas de gain significatif
- La stratégie **`pos_weight`** est **plus efficace** dans ce contexte :
  - Conserve la totalité des exemples d'entraînement (plus de données disponibles)
  - Évite la perte d'information due à la suppression de données
  - Plus simple à implémenter et à ajuster

**Recommandation :** Privilégier `pos_weight` pour ce projet. L'under-sampling pourrait être utile uniquement si le dataset initial était beaucoup plus volumineux (> 100k observations).

---

## 4. Évaluation du Modèle

### 4.1 Métriques de performance (Test Set)

```
==================================================
RAPPORT DE CLASSIFICATION — MLP PyTorch
==================================================
              precision    recall  f1-score   support

      Fidèle       0.95      0.69      0.80      1796
     Churner       0.21      0.70      0.32       204

    accuracy                           0.69      2000
   macro avg       0.58      0.70      0.56      2000
weighted avg       0.88      0.69      0.75      2000

ROC-AUC : 0.7457
```

### 4.2 Analyse des performances

#### Métriques clés sur la classe Churner (prioritaire)

- **Recall : 0.70 (70%)** ⭐ **OBJECTIF ATTEINT**
  - Le modèle détecte **70% des churners**
  - 30% de faux négatifs (churners non détectés)
  - **Impact métier :** Permet d'identifier la majorité des clients à risque

- **F1-Score : 0.32**
  - Compromis entre Precision et Recall
  - Relativement faible en raison de la Precision basse

- **Precision : 0.21 (21%)**
  - Sur 100 prédictions "churner", seulement 21 sont correctes
  - **Impact métier :** Beaucoup de faux positifs (clients fidèles classés à tort comme churners)
  - **Conséquence :** Surcoût opérationnel (actions de rétention sur clients stables)

- **ROC-AUC : 0.7457**
  - Performance globale satisfaisante (> 0.70)
  - Le modèle **discrimine correctement** les deux classes
  - Supérieur à un classifieur aléatoire (AUC = 0.50)

#### Matrice de Confusion (Test Set)

**[Insérer graphique : Matrice de confusion MLP]**

|  | Prédit Fidèle | Prédit Churner |
|---|---|---|
| **Réel Fidèle** | 1244 (VN) | 552 (FP) |
| **Réel Churner** | 61 (FN) | 143 (VP) |

**Lecture :**
- **143 Vrais Positifs** : Churners correctement détectés
- **61 Faux Négatifs** : Churners manqués ⚠️ (coût métier élevé)
- **552 Faux Positifs** : Fidèles classés à tort comme churners (surcoût opérationnel)
- **1244 Vrais Négatifs** : Fidèles correctement classés

### 4.3 Interprétation métier

#### Points forts ✅

1. **Recall élevé (70%)** : Le modèle remplit son objectif principal
   - Permet de **cibler la majorité des clients à risque** avec des actions de rétention
   - Minimise la perte de revenus due aux churners non détectés

2. **ROC-AUC > 0.74** : Capacité de discrimination satisfaisante
   - Le modèle peut **ordonner les clients par risque de churn**
   - Utile pour prioriser les actions (cibler d'abord les scores les plus élevés)

3. **Gestion efficace du déséquilibre**
   - Sans `pos_weight`, le modèle classerait tout comme "fidèle" (Recall = 0%)
   - Ici, le modèle détecte effectivement 70% des churners

#### Points d'amélioration ⚠️

1. **Precision faible (21%)**
   - **79% de faux positifs** parmi les prédictions "churner"
   - **Compromis assumé** : privilégier le Recall (ne pas manquer de churners) au détriment de la Precision
   - **Solution possible :** Ajuster le seuil de décision (threshold tuning)
     - Augmenter le seuil (ex. 0.6 au lieu de 0.5) → ↑ Precision, ↓ Recall
     - Abaisser le seuil (ex. 0.4) → ↓ Precision, ↑ Recall

2. **F1-Score modéré (0.32)**
   - Reflète le déséquilibre entre Precision et Recall
   - Acceptable dans un contexte où le **coût des faux négatifs** (churners manqués) est **supérieur** au coût des faux positifs (actions inutiles)

### 4.4 Courbe ROC

**[Insérer graphique : Courbe ROC MLP]**

La courbe ROC confirme la capacité du modèle à distinguer churners et non-churners indépendamment du seuil de décision. L'AUC de 0.7457 indique une **performance supérieure à un classifieur aléatoire** (AUC = 0.50) et **acceptable** pour une première itération deep learning.

**Optimisation possible :** Analyse de la courbe Precision-Recall (PR curve) pour choisir un seuil optimal adapté aux contraintes métier.

---

## 5. Comparaison avec les Modèles de Machine Learning Classique

### 5.1 Positionnement du MLP

| Critère | MLP (Deep Learning) | Random Forest / XGBoost |
|---------|---------------------|-------------------------|
| **Recall** | 0.70 | À comparer |
| **F1-Score** | 0.32 | À comparer |
| **ROC-AUC** | 0.7457 | À comparer |
| **Interprétabilité** | Faible (boîte noire) | Moyenne (feature importance) |
| **Temps d'entraînement** | Modéré (GPU recommandé) | Rapide |
| **Hyperparamètres** | Nombreux (architecture, dropout, lr, etc.) | Moins nombreux |

**Intérêt du MLP :**
- Capture des **interactions non-linéaires complexes** entre variables
- Potentiel d'amélioration via **architecture plus profonde** ou **feature engineering**
- Approche complémentaire aux modèles d'arbres (ensemble learning possible)

**Limites du MLP :**
- **Boîte noire** : difficile d'expliquer pourquoi un client est classé comme churner
- Nécessite un **tuning d'hyperparamètres** plus complexe
- Sensible à l'initialisation des poids (reproductibilité avec `torch.manual_seed`)

---

## 6. Améliorations Futures

### 6.1 Optimisation du seuil de décision (Threshold Tuning)

- **Problème actuel :** Seuil fixé à 0.5 (valeur par défaut)
- **Solution :** Analyser la courbe Precision-Recall pour choisir un seuil optimal
  - Exemple : Seuil à 0.35 → ↑ Recall (ex. 80%), ↓ Precision
  - Adaptation selon la **tolérance métier aux faux positifs**

### 6.2 Architecture et hyperparamètres

- Tester des architectures plus profondes (4-5 couches)
- Grid Search ou Random Search sur :
  - Taux de dropout (0.2, 0.3, 0.4, 0.5)
  - Learning rate (1e-4, 5e-4, 1e-3)
  - Nombre de neurones par couche

### 6.3 Techniques avancées de gestion du déséquilibre

- **Focal Loss** : Pénalise davantage les exemples mal classés
- **SMOTE** (Synthetic Minority Over-sampling) : Génération synthétique de churners
- **Ensemble avec sous-échantillonnages multiples** (Balanced Bagging)

### 6.4 Interprétabilité

- **SHAP pour MLP** : Explicabilité locale (pourquoi CE client est prédit churner ?)
- **Attention mechanisms** : Visualiser quelles features sont les plus importantes pour chaque prédiction

---

## 7. Conclusion

### 7.1 Synthèse

Le modèle MLP développé atteint un **Recall de 70%** sur la classe churner, répondant à l'objectif principal de **détection de la majorité des clients à risque**. L'architecture à 3 couches cachées, combinée aux techniques de régularisation (Dropout, BatchNorm) et à la stratégie `pos_weight` pour gérer le déséquilibre, permet d'obtenir des performances **supérieures à un modèle aléatoire** (ROC-AUC = 0.7457).

**Points clés :**
✅ **Recall prioritaire** : 70% des churners détectés  
✅ **Gestion efficace du déséquilibre** : `pos_weight` > under-sampling  
✅ **Architecture modulaire** : 3 couches cachées + régularisation  
✅ **Performance globale acceptable** : ROC-AUC = 0.7457  
⚠️ **Precision faible** : 79% de faux positifs (compromis assumé)  

### 7.2 Validation de l'exigence Deep Learning

**Exigence RNCP40875 (Bloc 2) :** Développer un modèle Deep Learning pour une tâche de classification.

✅ **VALIDÉ** : MLP (Multi-Layer Perceptron) implémenté en PyTorch avec :
- Architecture dense à 3 couches cachées
- Gestion du déséquilibre via `pos_weight`
- Techniques de régularisation avancées
- Évaluation rigoureuse avec métriques adaptées

Le modèle MLP constitue une **approche complémentaire** aux modèles de Machine Learning classique (Random Forest, XGBoost) et devra être comparé de manière rigoureuse pour sélectionner le **modèle candidat final** du système intelligent de rétention client.

---

**Annexes :**
- Code source : [notebooks/04_MLP.ipynb](../notebooks/04_MLP.ipynb)
- Modèle sauvegardé : [models/mlp_model_state_dict.pt](../models/mlp_model_state_dict.pt)
- Graphiques : [img/](../img/)
