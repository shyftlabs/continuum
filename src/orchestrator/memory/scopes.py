"""
Memory Scopes - Extensible scope management for memory operations.

Provides the MemoryScope class and a registry pattern for adding new scope types
without modifying existing code.

Adding a New Scope:
    1. Register it with register_scope()
    2. That's it - no code changes needed!

Example:
    ```python
    from orchestrator.memory.scopes import register_scope, MemoryScope

    # Register a new "team" scope
    register_scope(
        name="team",
        required_field="team_id",
        description="Memories isolated per team"
    )

    # Now you can use it
    scope = MemoryScope.from_isolation_mode("team", team_id="team-123")
    ```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# Scope Registry - Extensible scope definitions
# =============================================================================


@dataclass
class ScopeDefinition:
    """Definition of a memory scope type."""

    name: str
    required_field: str | None  # Field required for this scope (None for shared)
    description: str
    default_identifier: dict[str, str] | None = None  # For shared-like scopes


# Built-in scope definitions
_SCOPE_REGISTRY: dict[str, ScopeDefinition] = {}


def register_scope(
    name: str,
    required_field: str | None = None,
    description: str = "",
    default_identifier: dict[str, str] | None = None,
) -> None:
    """
    Register a new scope type.

    Args:
        name: Scope name (e.g., "user", "team", "org")
        required_field: Field required for this scope (e.g., "user_id", "team_id")
        description: Human-readable description
        default_identifier: Default identifier dict for scopes like "shared"

    Example:
        ```python
        # Add a team scope
        register_scope(
            name="team",
            required_field="team_id",
            description="Memories isolated per team"
        )

        # Add a global/shared scope
        register_scope(
            name="global",
            required_field=None,
            description="Global memories",
            default_identifier={"agent_id": "global"}
        )
        ```
    """
    _SCOPE_REGISTRY[name] = ScopeDefinition(
        name=name,
        required_field=required_field,
        description=description,
        default_identifier=default_identifier,
    )


def get_scope_definition(name: str) -> ScopeDefinition:
    """Get scope definition by name."""
    if name not in _SCOPE_REGISTRY:
        available = ", ".join(_SCOPE_REGISTRY.keys()) or "none"
        raise ValueError(
            f"Unknown scope type: {name}. "
            f"Available scopes: {available}. "
            f"Use register_scope() to add new scope types."
        )
    return _SCOPE_REGISTRY[name]


def list_scopes() -> list[str]:
    """List all registered scope names."""
    return list(_SCOPE_REGISTRY.keys())


def is_scope_registered(name: str) -> bool:
    """Check if a scope is registered."""
    return name in _SCOPE_REGISTRY


# =============================================================================
# Register Built-in Scopes
# =============================================================================

# Shared - accessible to everyone
register_scope(
    name="shared",
    required_field=None,
    description="Shared across all users/agents",
    default_identifier={"agent_id": "shared"},
)

# User - isolated per user
register_scope(
    name="user",
    required_field="user_id",
    description="Memories isolated per user",
)

# Agent - isolated per agent
register_scope(
    name="agent",
    required_field="agent_id",
    description="Memories isolated per agent",
)

# Conversation - isolated per conversation
register_scope(
    name="conversation",
    required_field="conversation_id",
    description="Memories isolated per conversation",
)


# =============================================================================
# MemoryIsolationLevel Enum (for backward compatibility)
# =============================================================================


class MemoryIsolationLevel(str, Enum):
    """
    Memory isolation levels.

    Note: New scopes added via register_scope() don't need enum entries.
    This enum exists for backward compatibility and IDE autocomplete.
    """

    SHARED = "shared"
    USER = "user"
    AGENT = "agent"
    CONVERSATION = "conversation"


# =============================================================================
# MemoryScope - Main scope management class
# =============================================================================


@dataclass
class MemoryScope:
    """
    Memory scope configuration.

    Encapsulates the identifiers used for memory scoping. Uses the scope
    registry to support extensible scope types.

    Built-in identifiers:
        - user_id: User identifier
        - agent_id: Agent identifier
        - conversation_id: Conversation identifier

    Custom identifiers can be stored in `custom_identifiers` for new scope types.

    Example:
        ```python
        # Create a user scope
        scope = MemoryScope.user("user-123")

        # Create from isolation mode
        scope = MemoryScope.from_isolation_mode(
            mode="user",
            user_id="user-123",
            agent_id="my-agent",
            conversation_id="conv-456",
        )

        # Get identifiers for provider
        identifiers = scope.to_identifiers()
        # Returns: {"user_id": "user-123"}
        ```
    """

    # Built-in identifiers
    user_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None

    # Custom identifiers for extensible scopes (e.g., team_id, org_id)
    custom_identifiers: dict[str, str] = field(default_factory=dict)

    # =========================================================================
    # Factory Methods - Built-in scopes
    # =========================================================================

    @classmethod
    def shared(cls) -> "MemoryScope":
        """
        Create a shared scope.

        Returns:
            MemoryScope with shared configuration.
        """
        return cls(agent_id="shared")

    @classmethod
    def user(cls, user_id: str) -> "MemoryScope":
        """
        Create a user scope.

        Args:
            user_id: User identifier

        Returns:
            MemoryScope with user isolation.
        """
        if not user_id:
            raise ValueError("user_id is required for user scope")
        return cls(user_id=user_id)

    @classmethod
    def agent(cls, agent_id: str) -> "MemoryScope":
        """
        Create an agent scope.

        Args:
            agent_id: Agent identifier

        Returns:
            MemoryScope with agent isolation.
        """
        if not agent_id:
            raise ValueError("agent_id is required for agent scope")
        return cls(agent_id=agent_id)

    @classmethod
    def conversation(cls, conversation_id: str) -> "MemoryScope":
        """
        Create a conversation scope.

        Args:
            conversation_id: Conversation identifier

        Returns:
            MemoryScope with conversation isolation.
        """
        if not conversation_id:
            raise ValueError("conversation_id is required for conversation scope")
        return cls(conversation_id=conversation_id)

    # =========================================================================
    # Generic Factory - Uses scope registry
    # =========================================================================

    @classmethod
    def from_isolation_mode(
        cls,
        mode: str | MemoryIsolationLevel,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        **custom_ids: str,
    ) -> "MemoryScope":
        """
        Create a scope from an isolation mode using the scope registry.

        This is the recommended way to create scopes as it supports
        both built-in and custom-registered scopes.

        Args:
            mode: Isolation level name (e.g., "user", "agent", "team")
            user_id: User identifier (for "user" mode)
            agent_id: Agent identifier (for "agent" mode)
            conversation_id: Conversation identifier (for "conversation" mode)
            **custom_ids: Custom identifiers (e.g., team_id="team-123")

        Returns:
            MemoryScope configured for the specified mode.

        Raises:
            ValueError: If required identifier is missing or mode is unknown.

        Example:
            ```python
            # Built-in scope
            scope = MemoryScope.from_isolation_mode("user", user_id="u123")

            # Custom scope (after registering "team")
            scope = MemoryScope.from_isolation_mode("team", team_id="t456")
            ```
        """
        if isinstance(mode, MemoryIsolationLevel):
            mode = mode.value

        # Get scope definition from registry
        scope_def = get_scope_definition(mode)

        # Build identifier mapping
        all_identifiers = {
            "user_id": user_id,
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            **custom_ids,
        }

        # Handle default identifier (for shared-like scopes)
        if scope_def.default_identifier:
            return cls(
                user_id=scope_def.default_identifier.get("user_id"),
                agent_id=scope_def.default_identifier.get("agent_id"),
                conversation_id=scope_def.default_identifier.get("conversation_id"),
                custom_identifiers={
                    k: v
                    for k, v in scope_def.default_identifier.items()
                    if k not in ("user_id", "agent_id", "conversation_id")
                },
            )

        # Validate required field
        if scope_def.required_field:
            value = all_identifiers.get(scope_def.required_field)
            if not value:
                raise ValueError(
                    f"'{scope_def.required_field}' is required for '{mode}' isolation mode. "
                    f"Provide {scope_def.required_field} parameter."
                )

        # Build scope based on mode
        if mode == "shared":
            return cls.shared()
        elif mode == "user" and user_id:
            return cls.user(user_id)
        elif mode == "agent" and agent_id:
            return cls.agent(agent_id)
        elif mode == "conversation" and conversation_id:
            return cls.conversation(conversation_id)
        else:
            # Custom scope - use the required field
            if scope_def.required_field:
                value = all_identifiers.get(scope_def.required_field)
                if scope_def.required_field in ("user_id", "agent_id", "conversation_id"):
                    # Built-in identifier
                    return cls(**{scope_def.required_field: value})
                else:
                    # Custom identifier
                    return cls(custom_identifiers={scope_def.required_field: value})
            else:
                return cls()

    @classmethod
    def from_identifiers(
        cls,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        **custom_ids: str,
    ) -> "MemoryScope":
        """
        Create a scope from explicit identifiers.

        Args:
            user_id: User identifier
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            **custom_ids: Custom identifiers

        Returns:
            MemoryScope with all provided identifiers.
        """
        return cls(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            custom_identifiers=custom_ids,
        )

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_identifiers(self) -> dict[str, str]:
        """
        Convert to identifier dictionary for providers.

        Returns only non-None identifiers.

        Returns:
            Dictionary with identifier key-value pairs.
        """
        identifiers: dict[str, str] = {}

        if self.user_id:
            identifiers["user_id"] = self.user_id
        if self.agent_id:
            identifiers["agent_id"] = self.agent_id
        if self.conversation_id:
            identifiers["conversation_id"] = self.conversation_id

        # Add custom identifiers
        identifiers.update(self.custom_identifiers)

        return identifiers

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with all scope fields.
        """
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "conversation_id": self.conversation_id,
            "custom_identifiers": self.custom_identifiers,
        }

    def get_primary_identifier(self, mode: str | MemoryIsolationLevel) -> dict[str, str]:
        """
        Get the primary identifier for a specific mode.

        Args:
            mode: Isolation level

        Returns:
            Dictionary with single identifier for the mode.

        Raises:
            ValueError: If required identifier is missing.
        """
        if isinstance(mode, MemoryIsolationLevel):
            mode = mode.value

        scope_def = get_scope_definition(mode)

        if scope_def.default_identifier:
            return scope_def.default_identifier.copy()

        if scope_def.required_field:
            # Check built-in fields first
            value = getattr(self, scope_def.required_field, None)
            if value is None:
                # Check custom identifiers
                value = self.custom_identifiers.get(scope_def.required_field)

            if not value:
                raise ValueError(f"'{scope_def.required_field}' is required for '{mode}' mode")
            return {scope_def.required_field: value}

        return {}

    # =========================================================================
    # Metadata Methods
    # =========================================================================

    def to_metadata(self) -> dict[str, Any]:
        """
        Get scope information as metadata.

        Returns:
            Dictionary with scope metadata (prefixed with _).
        """
        metadata: dict[str, Any] = {}

        if self.user_id:
            metadata["_user_id"] = self.user_id
        if self.agent_id:
            metadata["_agent_id"] = self.agent_id
        if self.conversation_id:
            metadata["_conversation_id"] = self.conversation_id

        # Add custom identifiers with prefix
        for key, value in self.custom_identifiers.items():
            metadata[f"_{key}"] = value

        return metadata

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_for_mode(self, mode: str | MemoryIsolationLevel) -> tuple[bool, str | None]:
        """
        Validate that required identifiers are present for a mode.

        Args:
            mode: Isolation level to validate against

        Returns:
            Tuple of (is_valid, error_message).
        """
        if isinstance(mode, MemoryIsolationLevel):
            mode = mode.value

        try:
            scope_def = get_scope_definition(mode)
        except ValueError as e:
            return False, str(e)

        if scope_def.default_identifier or not scope_def.required_field:
            return True, None

        # Check built-in fields
        value = getattr(self, scope_def.required_field, None)
        if value is None:
            # Check custom identifiers
            value = self.custom_identifiers.get(scope_def.required_field)

        if not value:
            return False, f"'{scope_def.required_field}' is required for '{mode}' isolation mode"

        return True, None

    def is_empty(self) -> bool:
        """Check if all identifiers are None/empty."""
        return (
            not self.user_id
            and not self.agent_id
            and not self.conversation_id
            and not self.custom_identifiers
        )

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        parts = []
        if self.user_id:
            parts.append(f"user_id={self.user_id!r}")
        if self.agent_id:
            parts.append(f"agent_id={self.agent_id!r}")
        if self.conversation_id:
            parts.append(f"conversation_id={self.conversation_id!r}")
        for key, value in self.custom_identifiers.items():
            parts.append(f"{key}={value!r}")

        return f"MemoryScope({', '.join(parts) if parts else 'empty'})"

    def __str__(self) -> str:
        """Human-readable string."""
        identifiers = self.to_identifiers()
        if not identifiers:
            return "MemoryScope(empty)"
        return f"MemoryScope({identifiers})"


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    "MemoryScope",
    # Enum (backward compat)
    "MemoryIsolationLevel",
    # Registry
    "ScopeDefinition",
    "register_scope",
    "get_scope_definition",
    "list_scopes",
    "is_scope_registered",
]
