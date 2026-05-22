# 📊 Rapport d'Analyse Exploratoire des Données (EDA)

## Système Intelligent Multi-Modèles pour la Rétention Client

> **Projet :** M1 Data Science - Certification RNCP40875 (Bloc 2)  
> **Date :** Mai 2026  
> **Objectif :** Prédiction du churn client et évaluation du risque financier

---

## 1. Introduction et Contexte

### 1.1 Objectif de l'analyse

Ce rapport présente l'analyse exploratoire des données (EDA) réalisée dans le cadre du développement d'un système intelligent de prédiction du churn client pour des entreprises SaaS, télécom ou services par abonnement.

L'objectif principal est d'**identifier les facteurs comportementaux, démographiques et transactionnels** qui influencent la décision d'un client de résilier son abonnement, afin de permettre des actions de rétention ciblées et d'optimiser le revenu.

### 1.2 Source des données

- **Source :** [Kaggle - Customer Churn Business Dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)
- **Taille :** 10 000 observations (clients)
- **Variables :** 31 colonnes (30 features + 1 target)
- **Types de variables :** Numériques et catégorielles
- **Variable cible :** `churn` (0 = Client fidèle, 1 = Client parti)

### 1.3 Structure du dataset

Le dataset contient **quatre catégories de variables** reflétant différentes dimensions du comportement client :

**Variables démographiques :**
- `customer_id`, `gender`, `country`, `city`, `customer_segment`, `age`

**Variables d'engagement produit :**
- `tenure_months`, `monthly_logins`, `weekly_active_days`, `avg_session_time`, `features_used`, `usage_growth_rate`, `last_login_days_ago`

**Variables contractuelles et financières :**
- `signup_channel`, `contract_type`, `monthly_fee`, `total_revenue`, `payment_method`, `payment_failures`, `discount_applied`, `price_increase_last_3m`

**Variables support client et satisfaction :**
- `support_tickets`, `avg_resolution_time`, `complaint_type`, `csat_score`, `escalations`, `email_open_rate`, `marketing_click_rate`, `nps_score`, `survey_response`, `referral_count`

---

## 2. Analyse de la Variable Cible : Churn

### 2.1 Distribution du churn

L'analyse de la variable cible révèle un **déséquilibre important** des classes :

- **Clients fidèles (churn = 0) :** 9 019 clients (90.2%)
- **Clients partis (churn = 1) :** 981 clients (9.8%)
- **Ratio :** Environ 9:1

**[Insérer graphique : Distribution du churn - Barplot et Pie chart]**

### 2.2 Impact business du churn

Au-delà du nombre de clients perdus, l'analyse révèle un **impact financier significatif** :

- **Revenu total généré :** Calculé sur l'ensemble de la base client
- **Revenu à risque :** Revenu généré par les clients ayant churné
- **Pourcentage du revenu à risque :** Représente la part du CA concernée par les départs

Ces métriques permettent de quantifier l'enjeu business et de prioriser les actions de rétention sur les segments à forte valeur.

### 2.3 Implications méthodologiques

Ce **déséquilibre des classes** (90/10) a des conséquences majeures sur la stratégie de modélisation :

#### ⚠️ Risques identifiés :
- **Biais vers la classe majoritaire** : Un modèle naïf pourrait prédire "fidèle" pour tous les clients et atteindre 90% d'accuracy, sans aucune capacité prédictive réelle
- **Faible détection des churners** : La classe minoritaire (la plus importante business-wise) risque d'être sous-détectée

#### ✅ Stratégies mises en œuvre :
1. **Stratified split obligatoire** : Utilisation de `stratify=y` dans `train_test_split` pour conserver la proportion 90/10 dans train et test
2. **Gestion du déséquilibre dans les modèles** : Application de `class_weight='balanced'` dans tous les algorithmes
3. **Métriques adaptées** :
   - ❌ **Accuracy** : Métrique trompeuse dans ce contexte
   - ✅ **Recall (Sensibilité)** : Priorité absolue pour détecter les vrais churners
   - ✅ **F1-Score** : Compromis entre Precision et Recall
   - ✅ **ROC-AUC / PR-AUC** : Performance globale du modèle

**Objectif prioritaire :** Maximiser le **Recall** pour minimiser les faux négatifs (churners non détectés), tout en maintenant une **Precision** acceptable pour éviter de sur-solliciter les clients fidèles.

---

## 3. Analyse des Variables Discriminantes

L'analyse univariée et bivariée a permis d'identifier les variables ayant le **pouvoir prédictif le plus élevé** pour la prédiction du churn. Ces variables reflètent trois dimensions clés du risque de résiliation : **l'engagement comportemental**, **la satisfaction client** et **la stabilité contractuelle**.

### 3.1 Variables numériques à fort pouvoir prédictif

#### 3.1.1 `csat_score` - Score de Satisfaction Client

Le **Customer Satisfaction Score (CSAT)** est l'un des indicateurs les plus discriminants du dataset.

**Observations statistiques :**
- **Médiane clients fidèles :** Score élevé
- **Médiane clients partis :** Score significativement plus faible
- **Corrélation avec churn :** Négative modérée

**Insight métier :**  
La satisfaction client mesurée par le CSAT est un **indicateur précoce du churn**. Les clients insatisfaits présentent une probabilité de résiliation nettement supérieure. Cette variable reflète directement la perception de la valeur du service.

**[Insérer graphique : Distribution de csat_score par churn - Boxplot et Histogramme]**

**Implication pour la rétention :**  
Les actions de fidélisation doivent cibler en priorité les clients ayant un CSAT faible, notamment via des enquêtes de satisfaction proactives et des actions correctives rapides.

---

#### 3.1.2 `tenure_months` - Ancienneté du Client

L'ancienneté du client est une variable **fortement discriminante** pour prédire le churn.

**Observations statistiques :**
- **Médiane clients fidèles :** 31 mois
- **Médiane clients partis :** 22 mois
- **Écart :** 9 mois
- **Corrélation avec churn :** Négative modérée

**Insight métier :**  
Les **nouveaux clients** (< 24 mois) sont significativement plus à risque de churn que les clients établis. Cela suggère une **phase critique d'onboarding** où l'ancrage produit n'est pas encore solide.

**[Insérer graphique : Distribution de tenure_months par churn - Boxplot et Histogramme]**

**Implication pour la rétention :**  
Mettre en place un **programme d'onboarding renforcé** pour les 12-24 premiers mois, avec des points de contact réguliers, formation à l'utilisation des fonctionnalités, et offres de fidélisation précoce.

---

#### 3.1.3 `monthly_logins` - Fréquence de Connexion

La fréquence de connexion mensuelle est un **indicateur comportemental clé** de l'engagement client.

**Observations statistiques :**
- **Médiane clients fidèles :** Nombre de connexions significativement plus élevé
- **Médiane clients partis :** Nombre de connexions plus faible
- **Corrélation avec churn :** Négative modérée

**Insight métier :**  
La **baisse de l'engagement** (diminution du nombre de connexions) est un **signal précoce de churn**. Les clients qui se désengagent progressivement du service ont une probabilité élevée de résilier dans les mois suivants.

**[Insérer graphique : Distribution de monthly_logins par churn - Boxplot et Histogramme]**

**Implication pour la rétention :**  
Mettre en place un **système d'alertes automatique** pour détecter les clients dont la fréquence de connexion diminue de manière significative, et déclencher des campagnes de réactivation (emails, notifications, offres personnalisées).

---

#### 3.1.4 `payment_failures` - Échecs de Paiement

Les échecs de paiement sont un **facteur de risque direct** de churn.

**Observations statistiques :**
- **Médiane clients fidèles :** 0 échec de paiement
- **Médiane clients partis :** Nombre d'échecs plus élevé
- **Corrélation avec churn :** Positive modérée

**Insight métier :**  
Les **incidents de paiement** (carte expirée, solde insuffisant, refus bancaire) créent des frictions dans la relation client et augmentent significativement le risque de résiliation. Chaque échec de paiement non résolu rapidement augmente la probabilité de perte définitive du client.

**[Insérer graphique : Distribution de payment_failures par churn - Boxplot et Histogramme]**

**Implication pour la rétention :**  
- Automatiser les **relances de paiement** avec plusieurs tentatives
- Proposer des **moyens de paiement alternatifs** en cas d'échec
- Mettre en place un **service client proactif** pour résoudre rapidement les problèmes de facturation
- Offrir une **période de grâce** pour éviter la suspension immédiate du service

---

#### 3.1.5 `last_login_days_ago` - Inactivité Récente

Le nombre de jours depuis la dernière connexion mesure le **niveau de désengagement** du client.

**Observations statistiques :**
- **Médiane clients fidèles :** Dernière connexion récente
- **Médiane clients partis :** Dernière connexion plus ancienne
- **Corrélation avec churn :** Positive

**Insight métier :**  
Une **inactivité prolongée** (> 30 jours sans connexion) est un indicateur fort de churn imminent. Les clients "dormants" ont déjà amorcé mentalement le processus de désengagement avant de procéder à la résiliation formelle.

**[Insérer graphique : Distribution de last_login_days_ago par churn - Boxplot et Histogramme]**

**Implication pour la rétention :**  
Segmenter les clients selon leur dernière activité :
- **0-7 jours** : Clients actifs → Campagnes d'upselling
- **8-30 jours** : Clients à surveiller → Emails de rappel personnalisés
- **> 30 jours** : Clients à haut risque → Campagnes de réactivation agressives (offres, remises)

---

#### 3.1.6 Autres variables numériques analysées

**`total_revenue` - Revenu Total Généré**
- **Écart médian :** 260$ (490$ vs 750$)
- **Insight :** Les clients à **faible valeur économique** sont plus volatiles
- **Implication :** Optimiser l'offre pour maximiser le revenu par client dès l'onboarding

**Variables à pouvoir prédictif plus limité :**
- `age`, `weekly_active_days`, `avg_session_time`, `features_used`, `usage_growth_rate`, `monthly_fee`, `support_tickets`, `avg_resolution_time`, `escalations`, `email_open_rate`, `marketing_click_rate`, `nps_score`, `referral_count`

Ces variables montrent des distributions similaires entre churners et non-churners et contribueront probablement de manière marginale à la modélisation.

---

### 3.2 Variables catégorielles à fort pouvoir prédictif

#### 3.2.1 `customer_segment` - Segment de Clientèle

**Distribution :**
- **Individual** : 60% du dataset
- **SME (PME)** : 30% du dataset
- **Enterprise** : 10% du dataset

**Taux de churn par segment :**
- **SME** : Taux de churn **significativement supérieur** à la moyenne globale (9.8%)
- **Individual** : Taux de churn proche de la moyenne
- **Enterprise** : Taux de churn inférieur à la moyenne

**[Insérer graphique : Taux de churn par customer_segment - Barplot horizontal]**

**Insight métier :**  
Les **PME (SME)** représentent un segment à risque élevé. Hypothèses :
- Besoins spécifiques moins bien couverts par l'offre standard
- Sensibilité accrue au prix
- Volatilité plus forte (croissance/décroissance rapide de l'activité)
- Concurrence plus intense sur ce segment

**Implication pour la rétention :**  
Développer une **offre dédiée PME** avec support personnalisé, pricing flexible et fonctionnalités adaptées aux besoins des petites structures.

---

#### 3.2.2 `contract_type` - Type de Contrat

**Distribution :**
- **Monthly (Mensuel)** : 60%
- **Annual (Annuel)** : 30%
- **Two-Year (Deux ans)** : 10%

**Taux de churn par type de contrat :**
- **Monthly** : Taux de churn **nettement supérieur** à la moyenne
- **Annual** : Taux de churn modéré
- **Two-Year** : Taux de churn plus faible

**[Insérer graphique : Taux de churn par contract_type - Barplot horizontal]**

**Insight métier :**  
Le type de contrat est un **facteur de stabilité majeur**. Les clients sans engagement (contrat mensuel) ont une **flexibilité de sortie maximale**, ce qui augmente mécaniquement le taux de churn. À l'inverse, les contrats pluriannuels créent un **ancrage contractuel** réduisant la volatilité.

**Implication pour la rétention :**  
- **Inciter les nouveaux clients** à souscrire des contrats annuels via des remises attractives
- **Proposer des migrations** vers des engagements plus longs aux clients mensuels satisfaits
- Anticiper les **fins de contrat** (notamment annuels) avec des campagnes de renouvellement proactives

---

#### 3.2.3 `survey_response` - Réponse aux Enquêtes de Satisfaction

**Distribution :**
- **Satisfied** : 50%
- **Neutral** : 30%
- **Unsatisfied** : 20%

**Taux de churn par réponse :**
- **Unsatisfied** : Taux de churn **très supérieur** à la moyenne (> 15-20%)
- **Neutral** : Taux de churn proche de la moyenne
- **Satisfied** : Taux de churn significativement inférieur

**[Insérer graphique : Taux de churn par survey_response - Barplot horizontal]**

**Insight métier :**  
La satisfaction exprimée dans les enquêtes est un **indicateur précoce et fiable** du churn. Les clients déclarant leur insatisfaction ont déjà amorcé mentalement le processus de rupture.

**Implication pour la rétention :**  
- Mettre en place des **enquêtes de satisfaction régulières** (post-interaction support, trimestrielles)
- **Alertes automatiques** sur les réponses négatives pour déclencher des actions correctives immédiates
- **Suivi personnalisé** des clients insatisfaits par le service client

---

#### 3.2.4 Autres variables catégorielles analysées

**Variables à pouvoir prédictif modéré :**

- **`signup_channel` (Canal d'inscription)**
  - **Referral** (parrainage) : Taux de churn légèrement plus élevé
  - Hypothèse : Motivation extrinsèque (récompense) plutôt qu'intérêt réel pour le service

- **`city` (Ville)**
  - New York, Toronto, Sydney : Taux de churn légèrement supérieur
  - Londres, Berlin : Taux de churn légèrement inférieur
  - Hypothèse : Différences culturelles, concurrence locale, qualité du service régional

- **`discount_applied` (Réduction Appliquée)**
  - Clients sans réduction : Taux de churn **légèrement plus élevé**
  - Hypothèse : Les réductions fidélisent temporairement ou sont déjà appliquées aux clients à risque

**Variables à faible pouvoir prédictif :**
- `gender`, `country`, `payment_method`, `complaint_type`, `price_increase_last_3m`

**[Insérer graphique : Comparaison des taux de churn pour les principales variables catégorielles - Multiple barplots]**

---

## 4. Analyse de Corrélation

### 4.1 Matrice de corrélation des variables numériques

L'analyse de corrélation permet d'identifier les **relations linéaires** entre les variables numériques et le churn, ainsi que les potentielles **multicolinéarités** à surveiller lors de la modélisation.

**[Insérer graphique : Matrice de corrélation complète - Heatmap]**

### 4.2 Variables corrélées avec le churn

**Corrélations négatives (↓ variable = ↑ churn) :**
1. **`csat_score`** : Corrélation négative modérée (~-0.35)
2. **`tenure_months`** : Corrélation négative modérée (~-0.30)
3. **`monthly_logins`** : Corrélation négative modérée (~-0.28)
4. **`total_revenue`** : Corrélation négative faible à modérée (~-0.22)

**Corrélations positives (↑ variable = ↑ churn) :**
1. **`payment_failures`** : Corrélation positive modérée (~+0.25)
2. **`last_login_days_ago`** : Corrélation positive faible à modérée (~+0.20)

### 4.3 Interprétation

Aucune variable n'est **fortement corrélée** avec le churn (|r| > 0.50), ce qui suggère que :
- Le churn est un phénomène **multifactoriel** nécessitant l'analyse combinée de plusieurs signaux
- Les **modèles non-linéaires** (Random Forest, XGBoost, MLP) seront probablement plus performants que la régression logistique
- L'**importance relative** des variables pourra varier selon l'algorithme utilisé

### 4.4 Multicolinéarité

Aucune corrélation excessive (|r| > 0.80) n'a été détectée entre variables explicatives, limitant le risque de multicolinéarité préjudiciable à la modélisation.

---

## 5. Traitement des Données

### 5.1 Gestion des valeurs manquantes

**Analyse initiale :**
- Une seule variable contient des valeurs manquantes : **`complaint_type`** (20% de valeurs nulles)

**Stratégie adoptée :**
- **Imputation par création d'une catégorie distincte** : `"No_Complaint"`
- **Justification :** Les valeurs manquantes ont une signification métier (absence de plainte), il est donc pertinent de les traiter comme une catégorie à part entière plutôt que de les supprimer ou de les imputer par le mode.

**Résultat :**
- Aucune perte d'information
- Variable `complaint_type` complète et exploitable pour la modélisation

### 5.2 Détection et traitement des outliers

**Variables concernées :**
Six variables numériques présentent des **valeurs extrêmes** susceptibles d'impacter la performance des modèles :
- `avg_session_time`
- `features_used`
- `usage_growth_rate`
- `last_login_days_ago`
- `monthly_fee`
- `total_revenue`

**[Insérer graphique : Boxplots des variables avec outliers - Avant traitement]**

**Méthode de détection : IQR (Interquartile Range)**

Pour chaque variable :
1. Calcul du premier quartile (Q1) et du troisième quartile (Q3)
2. Calcul de l'écart interquartile : IQR = Q3 - Q1
3. Définition des bornes :
   - **Borne inférieure** : Q1 - 1.5 × IQR
   - **Borne supérieure** : Q3 + 1.5 × IQR
4. **Traitement** : Remplacement des valeurs hors bornes par les bornes (méthode "clipping")

**Justification du choix du traitement :**
- ❌ **Suppression** : Risque de perte d'information et de biais
- ❌ **Imputation par médiane** : Distorsion artificielle de la distribution
- ✅ **Clipping (winsorization)** : Conservation de l'information tout en limitant l'impact des valeurs extrêmes

**Résultats :**
- Réduction significative de la variance pour les variables concernées
- Conservation de l'intégrité du dataset (10 000 observations maintenues)
- Distributions plus robustes pour la modélisation

**[Insérer graphique : Boxplots des variables avec outliers - Après traitement]**

### 5.3 Variables finales retenues

Après nettoyage et traitement :
- **10 000 observations** complètes
- **31 variables** dont :
  - **11 variables catégorielles**
  - **19 variables numériques**
  - **1 variable cible** (churn)

---

## 6. Synthèse et Recommandations pour la Modélisation

### 6.1 Principales découvertes de l'EDA

**1. Déséquilibre critique des classes (90/10)**
- Nécessite une stratégie adaptée : class weighting, métriques Recall/F1, stratified split

**2. Variables discriminantes identifiées (top 7)**

**Variables numériques :**
1. **`csat_score`** : Satisfaction client (corrélation négative modérée)
2. **`payment_failures`** : Incidents de paiement (corrélation positive modérée)
3. **`tenure_months`** : Ancienneté client (corrélation négative modérée)
4. **`monthly_logins`** : Engagement comportemental (corrélation négative modérée)
5. **`last_login_days_ago`** : Inactivité récente (corrélation positive)

**Variables catégorielles :**
6. **`customer_segment`** : Segment PME à haut risque
7. **`contract_type`** : Contrats mensuels plus volatiles

**3. Phénomène multifactoriel**
- Absence de variable unique fortement prédictive
- Combinaison de signaux comportementaux, financiers et contractuels nécessaire

**4. Données propres et exploitables**
- Valeurs manquantes traitées (1 variable)
- Outliers maîtrisés (6 variables)
- Dataset intègre (10 000 observations conservées)

### 6.2 Recommandations méthodologiques pour la modélisation

#### R1 : Stratégie de validation
- ✅ **Stratified K-Fold Cross-Validation** (k=5 ou 10)
- ✅ **Train/Test split stratifié** (80/20 ou 70/30)
- ✅ Conservation du ratio 90/10 dans tous les splits

#### R2 : Gestion du déséquilibre
- ✅ `class_weight='balanced'` dans tous les modèles
- ✅ Exploration de SMOTE si nécessaire
- ✅ Ajustement du seuil de décision (threshold tuning)

#### R3 : Métriques d'évaluation prioritaires
1. **Recall (Sensibilité)** : Minimiser les faux négatifs (churners non détectés)
2. **F1-Score** : Compromis Precision/Recall
3. **PR-AUC** : Performance globale (plus adaptée que ROC-AUC en cas de déséquilibre)
4. ⚠️ **Accuracy** : Métrique secondaire uniquement

#### R4 : Choix des algorithmes
- **Baseline** : Régression Logistique (interprétabilité)
- **Ensemble** : Random Forest (robuste, feature importance)
- **Boosting** : XGBoost / LightGBM (performance)
- **Deep Learning** : MLP (interactions complexes) - **OBLIGATOIRE**

#### R5 : Feature Engineering à explorer
Création de variables dérivées potentiellement pertinentes :
- `support_tickets / tenure_months` : Intensité du recours au support
- `payment_failures / tenure_months` : Taux d'échec de paiement
- `monthly_fee × tenure_months` : Lifetime Value (LTV) estimée
- Catégorisation de `last_login_days_ago` : [Actif, Inactif, Dormant]
- Catégorisation de `tenure_months` : [Nouveau, Établi, Fidèle]
- Score composite d'engagement : combinaison linéaire de `monthly_logins`, `weekly_active_days`, `avg_session_time`

#### R6 : Preprocessing pipeline
- ✅ **Encodage des catégorielles** : OneHotEncoder ou LabelEncoder selon la cardinalité
- ✅ **Scaling des numériques** : StandardScaler (recommandé pour MLP) ou RobustScaler
- ✅ **Pipeline sklearn** : Prévention du data leakage (fit sur train uniquement)

#### R7 : Interprétabilité des modèles
- **Feature Importance** : Random Forest, XGBoost (basique)
- **Permutation Importance** : Tous modèles (plus robuste)
- **SHAP** : Explicabilité locale et globale (valorisant)

### 6.3 Insights business actionnables

**Actions de rétention à prioriser :**

1. **Segment PME** : Offre dédiée, support personnalisé
2. **Clients < 24 mois** : Programme d'onboarding renforcé
3. **Clients à faible engagement** : Alertes automatiques + campagnes de réactivation
4. **Incidents de paiement** : Relances proactives + moyens de paiement alternatifs
5. **Clients insatisfaits** : Suivi personnalisé post-enquête
6. **Contrats mensuels** : Incitation aux engagements annuels

**KPIs à monitorer en temps réel :**
- Taux de churn par segment
- Revenu à risque (churners × revenu moyen)
- Taux de conversion des actions de rétention
- Évolution du CSAT par cohorte

---

## 7. Conclusion

L'analyse exploratoire des données a permis d'établir une **compréhension solide** du phénomène de churn dans le contexte business étudié.

**Points clés :**
- ✅ **Déséquilibre des classes** identifié et stratégies d'atténuation définies
- ✅ **7 variables discriminantes** majeures identifiées (orientant la modélisation et les actions métier)
- ✅ **Données nettoyées** et prêtes pour la phase de modélisation
- ✅ **Insights business** actionnables pour la direction marketing/CRM

L'EDA confirme que le churn est un phénomène **multifactoriel** combinant des signaux comportementaux (engagement, inactivité), financiers (échecs de paiement, revenu), de satisfaction (CSAT, NPS) et contractuels (ancienneté, type de contrat).

La phase de modélisation multi-algorithmes (Régression Logistique, Random Forest, XGBoost, MLP) permettra de :
1. Quantifier l'**importance relative** de chaque variable
2. Capturer les **interactions non-linéaires** entre features
3. Maximiser la **performance prédictive** (Recall, F1-Score)
4. Fournir une **explicabilité** pour les décideurs métier

**Prochaines étapes :**
1. Preprocessing et feature engineering
2. Modélisation multi-algorithmes (4+ modèles)
3. Évaluation comparative et sélection du modèle final
4. Interprétabilité (SHAP)
5. Développement du dashboard décisionnel (Streamlit)

---

**Annexes :**
- Code source : [notebooks/01_EDA.ipynb](../notebooks/01_EDA.ipynb)
- Graphiques : [img/](../img/)
- Dataset : [data/customer_churn_business_dataset.csv](../data/customer_churn_business_dataset.csv)
