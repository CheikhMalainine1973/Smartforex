Smart Forex : le bit de ce projet et de predire la tendence ou la besse d'une marcher finencier comme forex a l'aide des models entrenais a analyser(technique ou sentiment ) et prend la desicion final ou la prediction global 
Architecture : l'architecture proposer c’est une architecture en plusieurs modèles spécialisés, qu’on appelle souvent un système en pipeline ou en ensemble modulaire. 

Systeme en pipline :

Module 1 : Analyse Technique

Données : OHLCV (Open, High, Low, Close, Volume)
Modèle : LSTM ou Temporal Transformer
Sortie : prédiction brute (tendance, probabilité de hausse/baisse)

Module 2 : Analyse de Sentiment / News

Données : titres d’articles, tweets, annonces économiques
Modèle : FinBERT / FinGPT / autre NLP model
Sortie : score de sentiment ou de confiance du marché

Module 3 : Fusion & Prédiction Finale

Données : sortie des deux modules précédents
Modèle : MLP (Multi-Layer Perceptron), XGBoost, ou simple réseau dense
Objectif : prendre une décision plus robuste et contextuelle




Étapes principales du projet :
Étape	    Tâche	                     Description
1	Collecte de données		Forex (OHLCV) via API (ex: Yahoo Finance, AlphaVantage) + News
2	Prétraitement	        	Normalisation, calcul d’indicateurs techniques, nettoyage de texte
3	Modélisation Module 1		LSTM entraîné sur les données techniques
4	Modélisation Module 2		Sentiment analysis via FinBERT
5	Fusion des données		Combine les sorties des deux modules
6	Modélisation finale	        Entraîner un classifieur final
7	Évaluation & tests	        Metrics : Accuracy, F1-score, MAE, etc.
8	Visualisation & interprétation	Graphiques, tableaux de décisions
9	Automatisation			Script ou interface pour lancer l’analyse complète
