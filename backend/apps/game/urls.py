"""
URL configuration for the game app using DRF routers.

This module uses DefaultRouter to automatically generate URLs
for ViewSet actions.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .viewsets import GameViewSet, GameInviteViewSet

app_name = 'game'

router = DefaultRouter()
router.register(r'games', GameViewSet, basename='game')
router.register(r'invites', GameInviteViewSet, basename='invite')

urlpatterns = [
    path('', include(router.urls)),
]
