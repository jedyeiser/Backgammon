"""
Factory Boy factories for the AI app.

These factories create test instances of AI models with
sensible defaults for testing.
"""
import factory
import uuid

from apps.ai.models import AIModel, TrainingSession, EvolutionSession


class AIModelFactory(factory.django.DjangoModelFactory):
    """Factory for creating AIModel instances."""

    class Meta:
        model = AIModel

    id = factory.LazyFunction(uuid.uuid4)
    name = factory.Sequence(lambda n: f'test_model_{n}')
    model_type = AIModel.ModelType.NEURAL
    description = factory.Faker('sentence')

    network_architecture = factory.LazyFunction(
        lambda: {
            'input_size': 198,
            'output_size': 1,
            'layers': [
                {'id': 'fc1', 'type': 'linear', 'in': 198, 'out': 40},
                {'id': 'act1', 'type': 'activation', 'fn': 'sigmoid'},
                {'id': 'fc2', 'type': 'linear', 'in': 40, 'out': 1},
                {'id': 'output', 'type': 'activation', 'fn': 'sigmoid'},
            ]
        }
    )

    generation = 0
    training_games = 0
    training_epochs = 0
    is_active = False


class RandomAIModelFactory(AIModelFactory):
    """Factory for random player models."""

    name = factory.Sequence(lambda n: f'random_player_{n}')
    model_type = AIModel.ModelType.RANDOM
    network_architecture = {}


class HeuristicAIModelFactory(AIModelFactory):
    """Factory for heuristic player models."""

    name = factory.Sequence(lambda n: f'heuristic_player_{n}')
    model_type = AIModel.ModelType.HEURISTIC
    network_architecture = {}


class TrainedAIModelFactory(AIModelFactory):
    """Factory for trained models with metrics."""

    training_games = 1000
    training_epochs = 100
    win_rate_vs_random = 0.75
    is_active = True


class EvolvedAIModelFactory(AIModelFactory):
    """Factory for evolved models with lineage."""

    model_type = AIModel.ModelType.EVOLVED
    generation = factory.Sequence(lambda n: n)
    mutation_history = factory.LazyFunction(
        lambda: [{'type': 'weight', 'sigma': 0.1}]
    )


class TrainingSessionFactory(factory.django.DjangoModelFactory):
    """Factory for training sessions."""

    class Meta:
        model = TrainingSession

    id = factory.LazyFunction(uuid.uuid4)
    model = factory.SubFactory(AIModelFactory)
    status = TrainingSession.Status.PENDING
    num_games = 1000
    learning_rate = 0.1
    lambda_value = 0.7
    games_completed = 0
