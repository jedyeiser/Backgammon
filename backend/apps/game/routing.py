"""WebSocket URL routing for the game app."""
from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/game/(?P<game_id>[0-9a-f-]+)/$', consumers.GameConsumer.as_asgi()),
    re_path(r'ws/lobby/$', consumers.LobbyConsumer.as_asgi()),
]
