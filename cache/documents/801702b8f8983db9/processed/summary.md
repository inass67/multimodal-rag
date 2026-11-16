**Sujet** : Ce guide présente la configuration et le fonctionnement d'un environnement Spark 4.0 et JupyterLab conteneurisé avec Docker.

**Points clés** :
* L'environnement exécute Spark 4.0 et JupyterLab sans installation locale et nécessite 8 Go de RAM au minimum, dont au moins 4 Go alloués à Docker Desktop (page 1).
* Le démarrage s'effectue via `docker compose up -d` et l'accès à JupyterLab se fait sur `http://localhost:8888` avec le jeton de connexion « bigdata » (page 2).
* La session offre Spark en version 4.0.x, une interface Spark UI sur `http://localhost:4040` et la possibilité de lancer `spark-shell` ou `pyspark` depuis le terminal (page 3).
* La sauvegarde des fichiers et notebooks dans le dossier `work` est obligatoire sous peine de perte définitive lors de la suppression du conteneur (page 4).
* Le système fonctionne en mode local dans un conteneur unique et utilise le réseau `bigdata` pour permettre l'ajout futur de Kafka et Airflow (page 5).

**À retenir** : Cet environnement Docker offre une solution prête à l'emploi pour utiliser Spark 4.0 et JupyterLab en mode local tout en étant préparé pour l'intégration future de services comme Kafka et Airflow. Il convient de toujours enregistrer ses travaux dans le dossier `work` pour éviter la perte de données.