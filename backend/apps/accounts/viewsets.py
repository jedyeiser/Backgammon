"""
ViewSets for the accounts app.

This module provides ViewSet-based API endpoints for user management,
using DRF routers for automatic URL generation.
"""
from django.contrib.auth import get_user_model
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from drf_spectacular.utils import extend_schema, extend_schema_view

from .serializers import (
    UserSerializer,
    UserRegistrationSerializer,
    UserUpdateSerializer,
    ChangePasswordSerializer,
)

User = get_user_model()


@extend_schema_view(
    list=extend_schema(
        summary="List users",
        description="Returns a paginated list of all users.",
        tags=['auth'],
    ),
    retrieve=extend_schema(
        summary="Get user profile",
        description="Returns a user's public profile by username.",
        tags=['auth'],
    ),
    create=extend_schema(
        summary="Register a new user",
        description="Creates a new user account and returns authentication tokens.",
        tags=['auth'],
    ),
)
class UserViewSet(viewsets.ModelViewSet):
    """
    ViewSet for User model.

    Provides user management endpoints:
    - list: Get all users (paginated)
    - create: Register a new user
    - retrieve: Get a user's public profile
    - me: Get/update current user's profile
    - change_password: Change current user's password
    - leaderboard: Get top players by ELO rating
    """

    queryset = User.objects.all()
    lookup_field = 'username'

    def get_permissions(self):
        """Set permissions based on action."""
        if self.action in ['create', 'leaderboard']:
            return [AllowAny()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        """Return appropriate serializer based on action."""
        if self.action == 'create':
            return UserRegistrationSerializer
        if self.action in ['update', 'partial_update', 'me'] and self.request.method != 'GET':
            return UserUpdateSerializer
        if self.action == 'change_password':
            return ChangePasswordSerializer
        return UserSerializer

    def create(self, request, *args, **kwargs):
        """Register a new user and return tokens."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        refresh = RefreshToken.for_user(user)
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }, status=status.HTTP_201_CREATED)

    @extend_schema(
        summary="Get or update current user",
        description="GET returns the current user's profile. PUT/PATCH updates it.",
        tags=['auth'],
        responses={200: UserSerializer},
    )
    @action(detail=False, methods=['get', 'put', 'patch'])
    def me(self, request):
        """Get or update current user's profile."""
        if request.method == 'GET':
            return Response(UserSerializer(request.user).data)

        serializer = UserUpdateSerializer(
            request.user,
            data=request.data,
            partial=request.method == 'PATCH'
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(UserSerializer(request.user).data)

    @extend_schema(
        summary="Change password",
        description="Change the current user's password.",
        tags=['auth'],
        request=ChangePasswordSerializer,
    )
    @action(detail=False, methods=['post'])
    def change_password(self, request):
        """Change the current user's password."""
        serializer = ChangePasswordSerializer(
            data=request.data,
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)

        request.user.set_password(serializer.validated_data['new_password'])
        request.user.save()

        return Response({'message': 'Password changed successfully.'})

    @extend_schema(
        summary="Get leaderboard",
        description="Returns top 100 players sorted by ELO rating. Requires minimum 10 games played.",
        tags=['auth'],
        responses={200: UserSerializer(many=True)},
    )
    @action(detail=False, methods=['get'], permission_classes=[AllowAny])
    def leaderboard(self, request):
        """Get top 100 players by ELO rating."""
        users = User.objects.filter(
            games_played__gte=10
        ).order_by('-elo_rating')[:100]
        return Response(UserSerializer(users, many=True).data)
