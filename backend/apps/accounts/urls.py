"""URL configuration for the accounts app."""
from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from . import views

app_name = 'accounts'

urlpatterns = [
    # Authentication
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Profile management
    path('profile/', views.ProfileView.as_view(), name='profile'),
    path('password/change/', views.ChangePasswordView.as_view(), name='change_password'),

    # Public profiles
    path('users/<str:username>/', views.UserDetailView.as_view(), name='user_detail'),
    path('leaderboard/', views.LeaderboardView.as_view(), name='leaderboard'),
]
