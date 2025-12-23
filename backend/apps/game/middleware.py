"""
WebSocket middleware for JWT authentication.
"""
from urllib.parse import parse_qs
from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework_simplejwt.exceptions import TokenError
from django.contrib.auth import get_user_model

User = get_user_model()


@database_sync_to_async
def get_user_from_token(token_str: str):
    """Validate JWT token and return the user."""
    try:
        token = AccessToken(token_str)
        user_id = token.payload.get('user_id')
        return User.objects.get(id=user_id)
    except (TokenError, User.DoesNotExist):
        return AnonymousUser()


class JWTAuthMiddleware(BaseMiddleware):
    """
    Custom middleware that authenticates WebSocket connections using JWT.

    Token can be passed as:
    1. Query parameter: ws://host/ws/game/?token=xxx
    2. Subprotocol: Sec-WebSocket-Protocol: access_token, xxx
    """

    async def __call__(self, scope, receive, send):
        """Authenticate the connection and add user to scope."""
        # Try to get token from query string
        query_string = scope.get('query_string', b'').decode()
        query_params = parse_qs(query_string)
        token = query_params.get('token', [None])[0]

        # If no token in query, try subprotocols
        if not token:
            subprotocols = scope.get('subprotocols', [])
            if len(subprotocols) >= 2 and subprotocols[0] == 'access_token':
                token = subprotocols[1]

        # Authenticate
        if token:
            scope['user'] = await get_user_from_token(token)
        else:
            scope['user'] = AnonymousUser()

        return await super().__call__(scope, receive, send)
