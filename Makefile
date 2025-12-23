# Makefile for Backgammon project

.PHONY: help install dev build test lint clean docker-up docker-down migrate shell

# Default target
help:
	@echo "Backgammon Project Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install      Install all dependencies"
	@echo "  make dev          Start development servers"
	@echo "  make test         Run all tests"
	@echo "  make lint         Run linters"
	@echo "  make shell        Open Django shell"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up    Start Docker containers"
	@echo "  make docker-down  Stop Docker containers"
	@echo "  make docker-logs  View container logs"
	@echo ""
	@echo "Database:"
	@echo "  make migrate      Run migrations"
	@echo "  make makemigrations  Create new migrations"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Clean build artifacts"

# Install dependencies
install:
	cd backend && pip install -r requirements/development.txt
	cd frontend && npm install

# Development
dev:
	@echo "Starting development servers..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:5173"
	docker compose up

# Backend only
dev-backend:
	cd backend && python manage.py runserver

# Frontend only
dev-frontend:
	cd frontend && npm run dev

# Build
build:
	docker compose build

# Testing
test:
	cd backend && pytest
	cd frontend && npm test

test-backend:
	cd backend && pytest -v

test-frontend:
	cd frontend && npm test

test-cov:
	cd backend && pytest --cov=apps --cov-report=html

# Linting
lint:
	cd backend && ruff check .
	cd frontend && npm run lint

lint-fix:
	cd backend && ruff check --fix .
	cd frontend && npm run lint:fix

# Type checking
typecheck:
	cd backend && mypy apps
	cd frontend && npm run typecheck

# Database
migrate:
	cd backend && python manage.py migrate

makemigrations:
	cd backend && python manage.py makemigrations

shell:
	cd backend && python manage.py shell

# Docker
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-build:
	docker compose build --no-cache

docker-clean:
	docker compose down -v --rmi local

# Production
prod-up:
	docker compose -f docker-compose.prod.yml up -d

prod-down:
	docker compose -f docker-compose.prod.yml down

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	cd frontend && rm -rf node_modules dist

# Create superuser
createsuperuser:
	cd backend && python manage.py createsuperuser

# Collect static files
collectstatic:
	cd backend && python manage.py collectstatic --noinput
