.PHONY: help up down restart logs test lint mlflow airflow grafana prometheus api-health

help:
	@echo "Commandes disponibles :"
	@echo "  make up          - Démarrer tous les services"
	@echo "  make down        - Arrêter tous les services"
	@echo "  make restart     - Redémarrer tous les services"
	@echo "  make logs        - Voir les logs en temps réel"
	@echo "  make test        - Lancer les tests pytest"
	@echo "  make lint        - Vérifier le code avec ruff"
	@echo "  make mlflow      - Ouvrir MLflow (localhost:5000)"
	@echo "  make airflow     - Ouvrir Airflow (localhost:8080)"
	@echo "  make grafana     - Ouvrir Grafana (localhost:3000)"
	@echo "  make prometheus  - Ouvrir Prometheus (localhost:9090)"
	@echo "  make api-health  - Vérifier le statut de l'API"
	@echo "  make reload      - Recharger le modèle sans redémarrer"

up:
	docker-compose up -d --build
	@echo "✅ Services démarrés. MLflow: http://localhost:5000 | API: http://localhost/health"

down:
	docker-compose down
	@echo "🛑 Services arrêtés."

restart:
	docker-compose down && docker-compose up -d --build

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-mlflow:
	docker-compose logs -f mlflow

logs-airflow:
	docker-compose logs -f airflow-webserver airflow-scheduler

test:
	pytest tests/ -v

lint:
	ruff check mlops tests --ignore E501,E402,F401,F541

mlflow:
	@echo "MLflow disponible sur http://localhost:5000"
	start http://localhost:5000

airflow:
	@echo "Airflow disponible sur http://localhost:8080 (admin/admin)"
	start http://localhost:8080

grafana:
	@echo "Grafana disponible sur http://localhost:3000 (admin/admin)"
	start http://localhost:3000

prometheus:
	@echo "Prometheus disponible sur http://localhost:9090"
	start http://localhost:9090

api-health:
	curl -s http://localhost/health | python -m json.tool

reload:
	curl -s -X POST http://localhost/reload_model | python -m json.tool
	@echo "✅ Modèle rechargé depuis MLflow registry."

dvc-pull:
	dvc pull
	@echo "✅ Modèle téléchargé depuis DagsHub."

dvc-push:
	dvc push
	@echo "✅ Modèle pushé sur DagsHub."
