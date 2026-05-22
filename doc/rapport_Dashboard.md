# 📊 Rapport Dashboard Interactif : Churn Intelligence

## Système Intelligent Multi-Modèles pour la Rétention Client

> **Projet :** M1 Data Science - Certification RNCP40875 (Bloc 2)  
> **Date :** Mai 2026  
> **Framework :** Streamlit

---

## 1. Introduction et Objectifs

### 1.1 Contexte

Le dashboard **Churn Intelligence** constitue l'interface décisionnelle du système de prédiction du churn. Il transforme les prédictions des modèles de machine learning en **insights actionnables** pour les équipes marketing, CRM et direction commerciale.

### 1.2 Objectifs du dashboard

✅ **Visualiser les KPIs clés** : Clients à risque, revenu menacé, taux de churn estimé  
✅ **Identifier les clients prioritaires** : Tableau des clients à forte valeur et fort risque  
✅ **Comprendre les facteurs de risque** : Visualisation des variables discriminantes  
✅ **Comparer les modèles** : Performances des algorithmes ML et Deep Learning  

### 1.3 Technologies utilisées

- **Framework :** Streamlit (Python)
- **Visualisations :** Plotly Express (graphiques interactifs)
- **Modèle déployé :** XGBoost (sélectionné comme modèle final)
- **Seuil de décision :** Probabilité ≥ 32% → Client classé "à risque"

---

## 2. Page d'Accueil : Vue d'Ensemble et KPIs

### 2.1 Objectif de la page

La page d'accueil fournit une **vue synthétique** de l'état de la base client et des risques de résiliation. Elle s'adresse principalement aux décideurs (direction marketing, CRM) nécessitant une lecture rapide de la situation.

### 2.2 KPIs affichés

Le dashboard présente **4 indicateurs clés** mis à jour en temps réel :

#### 1. 👥 **Clients total**
- Nombre total de clients dans la base
- Permet de contextualiser les métriques suivantes

#### 2. ⚠️ **Clients à risque**
- Nombre de clients dont la probabilité de churn ≥ 32%
- **Interprétation métier :** Clients nécessitant une action de rétention immédiate
- Seuil justifié par l'optimisation Recall/Precision du modèle XGBoost

#### 3. 📉 **Taux de désabonnement estimé**
- Pourcentage de la base client classée "à risque"
- **Benchmark :** À comparer aux taux historiques de churn réel
- Permet de suivre l'évolution du risque dans le temps

#### 4. 💰 **Revenu mensuel à risque**
- Somme des frais mensuels (`monthly_fee`) des clients à risque
- **Impact business direct :** Estimation du MRR (Monthly Recurring Revenue) menacé
- Indicateur prioritaire pour la direction financière

**[Insérer screenshot : Page d'accueil avec les 4 KPIs]**

### 2.3 Facteurs de risque de résiliation

La page d'accueil affiche un **graphique de feature importance** généré par le modèle XGBoost, montrant l'importance relative de chaque variable dans la prédiction du churn.

**Variables les plus influentes (top 5)** :
1. **CSAT Score** : Score de satisfaction client
2. **Payment Failures** : Nombre d'échecs de paiement
3. **Tenure Months** : Ancienneté du client
4. **Monthly Logins** : Fréquence de connexion
5. **Customer Segment** : Segment de clientèle (SME, Individual, Enterprise)

**Interprétation métier :** Ces variables représentent les **leviers d'action prioritaires** pour les campagnes de rétention.

**[Insérer graphique : Feature Importance XGBoost]**

---

## 3. Page "Clients à Risque" : Priorisation des Actions

### 3.1 Objectif de la page

Identifier les **clients à fidéliser en priorité** en croisant le risque de churn avec la valeur économique du client. Cette page répond à la question : **"Sur quels clients concentrer nos ressources de rétention ?"**

### 3.2 Méthodologie de classement

**Métrique de priorisation :** **Revenu à risque** = `monthly_fee × probabilité de churn`

**Justification :**
- Un client avec un faible `monthly_fee` mais une probabilité de churn élevée représente un risque financier limité
- Un client avec un `monthly_fee` élevé mais une faible probabilité de churn ne nécessite pas d'action immédiate
- Le **produit des deux** identifie les clients à **fort impact financier ET forte probabilité de départ**

### 3.3 Tableau des Top 10 clients à fidéliser

Le dashboard affiche un tableau interactif des **10 clients prioritaires**, classés par revenu à risque décroissant.

**Colonnes affichées :**
- **ID Client** : Identifiant unique
- **Score de satisfaction** : CSAT (indicateur de satisfaction)
- **Échecs paiement** : Nombre d'incidents de paiement
- **Ancienneté (mois)** : Durée de relation client
- **Connexions mensuelles** : Fréquence d'utilisation du service
- **Segment** : SME, Individual ou Enterprise
- **Frais mensuels ($)** : Revenu récurrent mensuel
- **Probabilité de churn** : Score du modèle (%)
- **Revenu à risque ($)** : Métrique de priorisation (en surbrillance)

**Mise en forme :**
- Dégradé de couleurs sur **"Revenu à risque"** (rouge foncé = priorité maximale)
- Dégradé de couleurs sur **"Probabilité de churn"** (orange = risque élevé)
- Format professionnel (valeurs monétaires formatées, pourcentages)

**[Insérer screenshot : Tableau Top 10 clients à risque]**

### 3.4 Valeur métier

Ce tableau permet aux équipes CRM de :
✅ **Prioriser les appels de rétention** sur les clients à plus fort enjeu financier  
✅ **Personnaliser les offres** en fonction du profil client (segment, ancienneté, satisfaction)  
✅ **Anticiper les départs** avant la résiliation effective  
✅ **Mesurer l'efficacité** des actions de rétention (suivi du taux de succès)  

---

## 4. Page "Données" : Analyse des Facteurs Critiques

### 4.1 Objectif de la page

Visualiser les **relations entre les variables discriminantes et le risque de churn** à travers des graphiques interactifs. Cette page répond à la question : **"Quels sont les profils clients les plus à risque ?"**

### 4.2 Visualisations implémentées

#### 4.2.1 Taux de résiliation par segment client

**Type :** Bar chart (Plotly)  
**Insight :** Le segment **SME (Petites et Moyennes Entreprises)** présente un taux de churn significativement **supérieur** aux segments Individual et Enterprise.

**Implication métier :** Développer une **offre dédiée PME** avec support personnalisé et pricing adapté.

**[Insérer graphique : Taux de résiliation par segment]**

---

#### 4.2.2 Taux de résiliation par type de contrat

**Type :** Bar chart (Plotly)  
**Insight :** Les contrats **mensuels** (sans engagement) présentent un taux de churn **nettement supérieur** aux contrats annuels et bi-annuels.

**Implication métier :**
- Inciter les nouveaux clients à souscrire des contrats annuels (remises attractives)
- Proposer des migrations contractuelles aux clients mensuels satisfaits
- Anticiper les fins de contrat annuel avec des campagnes de renouvellement proactives

**[Insérer graphique : Taux de résiliation par contrat]**

---

#### 4.2.3 Taux de résiliation selon les échecs de paiement

**Type :** Bar chart (Plotly)  
**Insight :** Chaque échec de paiement **augmente significativement** le risque de churn. Les clients avec 3+ échecs ont un taux de résiliation critique.

**Implication métier :**
- Automatiser les **relances de paiement** (plusieurs tentatives, moyens alternatifs)
- Mettre en place un **service client proactif** en cas d'échec
- Proposer une **période de grâce** pour éviter la suspension immédiate du service

**[Insérer graphique : Taux de résiliation selon échecs paiement]**

---

#### 4.2.4 Distribution du score de satisfaction (CSAT)

**Type :** Histogramme superposé (clients fidèles vs clients à risque)  
**Insight :** Les clients à risque présentent une distribution de CSAT **décalée vers les valeurs basses**, confirmant le lien fort entre insatisfaction et churn.

**Implication métier :**
- Mettre en place des **enquêtes de satisfaction régulières**
- Déclencher des **alertes automatiques** sur les réponses négatives
- Assurer un **suivi personnalisé** des clients insatisfaits

**[Insérer graphique : Distribution CSAT par statut]**

---

#### 4.2.5 Distribution de l'ancienneté client

**Type :** Histogramme superposé (clients fidèles vs clients à risque)  
**Insight :** Les clients à risque ont une **ancienneté médiane plus faible** (nouveaux clients moins ancrés dans le service).

**Implication métier :**
- Renforcer le **programme d'onboarding** (12-24 premiers mois)
- Points de contact réguliers et formation à l'utilisation des fonctionnalités
- Offres de fidélisation précoce pour ancrer la relation

**[Insérer graphique : Distribution ancienneté par statut]**

---

#### 4.2.6 Distribution des connexions mensuelles

**Type :** Histogramme superposé (clients fidèles vs clients à risque)  
**Insight :** Les clients à risque se connectent **moins fréquemment** (désengagement progressif avant résiliation).

**Implication métier :**
- Système d'**alertes automatiques** sur baisse de la fréquence de connexion
- Campagnes de **réactivation** (emails, notifications, offres personnalisées)
- Segmentation : 0-7 jours (actifs), 8-30 jours (à surveiller), > 30 jours (haut risque)

**[Insérer graphique : Distribution connexions mensuelles par statut]**

---

### 4.3 Interactivité des graphiques

Tous les graphiques sont générés avec **Plotly Express**, offrant :
✅ **Zoom** et **pan** pour explorer les données  
✅ **Tooltip** avec valeurs détaillées au survol  
✅ **Légendes cliquables** pour isoler des catégories  
✅ **Export image** pour intégration dans des rapports  

---

## 5. Page "Comparaison des Modèles" : Validation des Choix Techniques

### 5.1 Objectif de la page

Présenter une **comparaison visuelle** des performances des différents algorithmes testés (Machine Learning classique et Deep Learning), justifiant le choix du modèle final déployé dans la plateforme.

### 5.2 Modèles Machine Learning comparés

Le dashboard affiche côte à côte les performances de **trois modèles de ML classique** :
1. **Régression Logistique** (baseline)
2. **Random Forest** (ensemble method)
3. **XGBoost** (gradient boosting)

**Visualisations :**
- **Matrices de confusion** : Comparaison des taux de vrais/faux positifs/négatifs
- **Courbes ROC** : Comparaison des AUC (Area Under Curve)

**[Insérer graphique : Matrices de confusion ML classiques]**  
**[Insérer graphique : Courbes ROC ML classiques]**

### 5.3 Modèle Deep Learning (MLP)

Une section dédiée présente les performances du **Multi-Layer Perceptron (PyTorch)** :
- **Matrice de confusion** spécifique au MLP
- **Courbe ROC** du réseau de neurones

**[Insérer graphique : MLP - Matrice de confusion + Courbe ROC]**

### 5.4 Valeur de cette page

Cette page démontre la **rigueur méthodologique** du projet :
✅ Comparaison objective de plusieurs approches  
✅ Transparence sur les performances de chaque algorithme  
✅ Justification du choix du modèle final (XGBoost)  
✅ Validation de l'exigence Deep Learning (MLP développé et évalué)  

**Note :** Les détails techniques (métriques précises, hyperparamètres) sont documentés dans les rapports de modélisation dédiés.

---

## 6. Architecture Technique du Dashboard

### 6.1 Structure de l'application

**Architecture multi-pages Streamlit :**
```
src/dashboard/
│
├── Accueil.py              # Page principale (KPIs + feature importance)
└── pages/
    ├── 1_Clients_a_risque.py   # Top 10 clients à fidéliser
    ├── 2_Donnees.py            # Visualisations des facteurs critiques
    ├── 3_Prediction.py         # [Non décrite dans ce rapport]
    └── 4_Comparaison_modeles.py # Performances des algorithmes
```

### 6.2 Pipeline de prédiction en temps réel

**Chargement des données** :
1. Import du dataset via `charger_donnees()`
2. Prétraitement via `traiter_donnees()` (valeurs manquantes, outliers)

**Génération des prédictions** :
3. Chargement du modèle XGBoost sérialisé (`models/xgboost_model.pkl`)
4. Chargement du preprocessor sklearn (`models/preprocessor.pkl`)
5. Transformation des features
6. Calcul des probabilités de churn pour tous les clients

**Mise en cache** :
- Décorateur `@st.cache_data` pour éviter les recalculs à chaque interaction
- Rechargement uniquement si données modifiées

### 6.3 Paramétrage du seuil de décision

**Seuil global :** `SEUIL_CHURN = 0.32` (32%)

**Justification :**
- Optimisé pour maximiser le **Recall** (détection des churners)
- Compromis entre faux positifs (coût opérationnel) et faux négatifs (perte de revenus)
- Ajustable selon les contraintes métier de l'entreprise

---

## 7. Valeur Business et Impact Métier

### 7.1 Pour les équipes CRM

✅ **Identification proactive** des clients à risque  
✅ **Priorisation** des actions de rétention par impact financier  
✅ **Personnalisation** des offres selon le profil client  
✅ **Mesure de l'efficacité** des campagnes de fidélisation  

### 7.2 Pour la direction marketing

✅ **Compréhension des facteurs de churn** (segments, contrats, satisfaction)  
✅ **Optimisation des stratégies** de pricing et d'engagement contractuel  
✅ **Suivi en temps réel** des KPIs de rétention  
✅ **Aide à la décision** basée sur des données objectives  

### 7.3 Pour la direction financière

✅ **Quantification du revenu à risque** (MRR menacé)  
✅ **Projection de l'impact** des campagnes de rétention  
✅ **Optimisation du ROI** des investissements CRM  
✅ **Anticipation des pertes** de chiffre d'affaires  

---

## 8. Avantages de l'Approche Streamlit

### 8.1 Rapidité de développement

- **Framework léger** : Pas de HTML/CSS/JavaScript à gérer
- **Syntaxe Python pure** : Accessibilité pour les data scientists
- **Déploiement simplifié** : `streamlit run Accueil.py`

### 8.2 Fonctionnalités natives

✅ **Layouts responsive** : Colonnes, tabs, expanders  
✅ **Widgets interactifs** : Sliders, selectbox, dataframes  
✅ **Mise en cache automatique** : Optimisation des performances  
✅ **Rechargement à chaud** : Développement itératif rapide  

### 8.3 Déploiement

**Options de déploiement :**
- **Streamlit Cloud** : Hébergement gratuit avec intégration GitHub
- **Docker** : Conteneurisation pour environnements on-premise
- **Azure/AWS/GCP** : Déploiement cloud entreprise

---

## 9. Améliorations Futures

### 9.1 Fonctionnalités additionnelles

- **Filtres dynamiques** : Sélection par segment, contrat, ancienneté
- **Export de rapports** : Génération PDF/Excel des clients à risque
- **Historisation** : Suivi de l'évolution des KPIs dans le temps
- **Alertes automatiques** : Notifications email pour clients critiques

### 9.2 Intégration CRM

- **API REST** : Connecteur vers Salesforce, HubSpot, etc.
- **Synchronisation bidirectionnelle** : Mise à jour des statuts clients
- **Webhook** : Déclenchement d'actions automatiques dans le CRM

### 9.3 Advanced Analytics

- **Simulations** : Impact de scénarios sur le taux de churn
- **Cohorte analysis** : Évolution du churn par cohorte d'inscription
- **Survival analysis** : Courbes de rétention par profil client

---

## 10. Conclusion

### 10.1 Synthèse

Le dashboard **Churn Intelligence** transforme les prédictions du modèle XGBoost en **outil décisionnel opérationnel** pour les équipes métier. Il répond à trois besoins critiques :

1. **Visibilité** : KPIs synthétiques pour décideurs
2. **Priorisation** : Identification des clients à forte valeur et fort risque
3. **Compréhension** : Visualisation des facteurs de churn actionnables

### 10.2 Validation de l'exigence RNCP40875 (Bloc 2)

**Compétence C3.2 : Élaborer une communication infographique visuelle**

✅ **Tableaux de bord interactifs** : 4 pages Streamlit avec KPIs et visualisations  
✅ **Collaboration équipes métiers** : Design orienté utilisateur (CRM, Marketing, Direction)  
✅ **Extraction de connaissances en temps réel** : Prédictions et insights actualisés  
✅ **Aide à la décision éclairée** : Priorisation des actions de rétention  
✅ **Inclusivité** : Interface accessible, graphiques clairs, tooltips explicatifs  

### 10.3 Impact métier attendu

En déployant ce dashboard, l'entreprise dispose d'une plateforme pour :
- **Réduire le taux de churn** de X% (à mesurer post-déploiement)
- **Optimiser le ROI** des campagnes de rétention (ciblage précis)
- **Préserver le revenu mensuel récurrent** (actions proactives)
- **Améliorer la satisfaction client** (traitement des signaux faibles)

Le dashboard constitue le **point d'entrée unique** pour toutes les parties prenantes impliquées dans la stratégie de rétention client, transformant des modèles statistiques complexes en insights métier actionnables.

---

**Annexes :**
- Code source : [src/dashboard/](../src/dashboard/)
- Commande de lancement : `streamlit run src/dashboard/Accueil.py`
- Framework : [Streamlit Documentation](https://docs.streamlit.io/)
