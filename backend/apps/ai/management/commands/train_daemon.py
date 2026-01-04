"""
Management command to train a player daemon.

Usage:
    python manage.py train_daemon <username> [--epochs 100] [--min-games 10]

This trains a behavioral cloning model that learns to play like
the specified user based on their historical game data.
"""
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = 'Train a player daemon (behavioral cloning model) for a user'

    def add_arguments(self, parser):
        parser.add_argument(
            'username',
            type=str,
            help='Username of the player to create a daemon for',
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='Number of training epochs (default: 100)',
        )
        parser.add_argument(
            '--min-games',
            type=int,
            default=10,
            help='Minimum completed games required for training (default: 10)',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Training batch size (default: 32)',
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=0.001,
            help='Learning rate (default: 0.001)',
        )
        parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            choices=['cpu', 'cuda', 'mps'],
            help='Device to train on (default: cpu)',
        )
        parser.add_argument(
            '--evaluate',
            action='store_true',
            help='Evaluate the model after training',
        )

    def handle(self, *args, **options):
        username = options['username']

        # Find the user
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise CommandError(f"User '{username}' does not exist")

        self.stdout.write(f"Training daemon for user: {username}")

        # Import training components
        from apps.ai.training.behavioral_cloning import (
            BehavioralCloningTrainer,
            TrainingConfig,
        )

        # Configure training
        config = TrainingConfig(
            epochs=options['epochs'],
            min_games=options['min_games'],
            batch_size=options['batch_size'],
            learning_rate=options['learning_rate'],
            device=options['device'],
        )

        # Create and run trainer
        trainer = BehavioralCloningTrainer(user, config)

        try:
            self.stdout.write("Preparing dataset...")
            trainer.prepare()

            self.stdout.write(f"Starting training for {config.epochs} epochs...")
            result = trainer.train()

            self.stdout.write(self.style.SUCCESS(
                f"\nTraining completed!"
                f"\n  Epochs trained: {result.epochs_trained}"
                f"\n  Final accuracy: {result.final_accuracy:.2%}"
                f"\n  Best validation accuracy: {result.best_accuracy:.2%}"
                f"\n  Training samples: {result.training_samples}"
                f"\n  Validation samples: {result.validation_samples}"
            ))

            # Save the model
            self.stdout.write("Saving model...")
            ai_model = trainer.save_model()
            self.stdout.write(self.style.SUCCESS(
                f"Model saved as: {ai_model.name}"
            ))

            # Optional evaluation
            if options['evaluate']:
                self.stdout.write("\nEvaluating model accuracy...")
                metrics = trainer.evaluate_accuracy()
                self.stdout.write(
                    f"  Action type accuracy: {metrics.get('action_type_accuracy', 0):.2%}"
                    f"\n  From point accuracy: {metrics.get('from_point_accuracy', 0):.2%}"
                    f"\n  To point accuracy: {metrics.get('to_point_accuracy', 0):.2%}"
                )

        except ValueError as e:
            raise CommandError(str(e))
        except Exception as e:
            raise CommandError(f"Training failed: {e}")
