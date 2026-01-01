"""
URL configuration for the accounts app using DRF routers.

This module combines router-based ViewSet URLs with standalone
JWT authentication endpoints.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView

from .viewsets import UserViewSet
from .views import LoginView, LogoutView

app_name = 'accounts'

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')

urlpatterns = [
    # Router-based endpoints
    path('', include(router.urls)),

    # JWT authentication endpoints (standalone)
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
