"""URL configuration for the game app."""
from django.urls import path

from . import views

app_name = 'game'

urlpatterns = [
    # Game management
    path('games/', views.GameListView.as_view(), name='game_list'),
    path('games/active/', views.ActiveGamesView.as_view(), name='active_games'),
    path('games/open/', views.OpenGamesView.as_view(), name='open_games'),
    path('games/create/', views.GameCreateView.as_view(), name='game_create'),
    path('games/<uuid:id>/', views.GameDetailView.as_view(), name='game_detail'),
    path('games/<uuid:game_id>/join/', views.GameJoinView.as_view(), name='game_join'),
    path('games/<uuid:game_id>/move/', views.MakeMoveView.as_view(), name='game_move'),
    path('games/<uuid:game_id>/moves/', views.GameMoveHistoryView.as_view(), name='game_moves'),
    path('games/<uuid:game_id>/resign/', views.ResignGameView.as_view(), name='game_resign'),

    # Invites
    path('invites/', views.InviteListView.as_view(), name='invite_list'),
    path('invites/sent/', views.SentInvitesView.as_view(), name='sent_invites'),
    path('invites/create/', views.CreateInviteView.as_view(), name='create_invite'),
    path('invites/<uuid:invite_id>/respond/', views.RespondInviteView.as_view(), name='respond_invite'),
]
