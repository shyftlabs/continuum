"""
IntelligentMemoryClient — Memory with entity extraction, importance scoring,
time-weighted decay, and structured cross-session user knowledge.

Extends MemoryClient without modifying the existing provider pipeline.
All four features layer on top via pre/post-processing hooks on add() and search().

Features
--------
1. Importance scoring  — LLM assigns 0-1 score at store time; blended with
                         semantic similarity at retrieval time.
2. Time-weighted decay — Recent memories get a relevance boost; very old,
                         low-importance memories can be pruned automatically.
3. Entity memory       — Extracts named entities (people, orgs, products) from
                         conversation text and stores them as tagged memories so
                         vague references ("that vendor I mentioned") resolve correctly.
4. User knowledge      — Builds and updates a structured user profile (preferences,
                         employer, expertise, topics) across all sessions for a user.

Usage
-----
    from orchestrator.memory import IntelligentMemoryClient, IntelligenceConfig

    client = IntelligentMemoryClient(
        config=MemoryConfig(...),
        intelligence_config=IntelligenceConfig(
            enable_entity_memory=True,
            enable_user_profiles=True,
            enable_scoring=True,
            enable_decay=True,
        ),
    )

    # add() — scores importance, extracts entities, updates user profile
    await client.add("I met with John from Acme Corp", user_id="user-123")

    # search() — re-ranks results by blended score
    results = await client.search("that vendor I mentioned", user_id="user-123")

    # entity-specific search
    entities = await client.search_entities("vendor", user_id="user-123")

    # structured user profile
    profile = await client.get_user_profile("user-123")

    # prune old, unimportant memories
    pruned = await client.prune(user_id="user-123")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orchestrator.logging import get_logger
from orchestrator.memory.client import MemoryClient
from orchestrator.memory.config import MemoryConfig
from orchestrator.memory.types import MemoryAddResult, MemoryEntry, MemorySearchResult

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class IntelligenceConfig:
    """Configuration for IntelligentMemoryClient features."""

    # Feature toggles
    enable_entity_memory: bool = True    # Extract and store named entities
    enable_user_profiles: bool = True    # Build structured user knowledge per user
    enable_scoring: bool = True          # LLM importance score at store time
    enable_decay: bool = True            # Time-weighted relevance modifier at search time

    # Retrieval score weights (must sum to 1.0)
    semantic_weight: float = 0.6         # Weight for vector semantic similarity
    importance_weight: float = 0.3       # Weight for stored importance score
    decay_weight: float = 0.1            # Weight for recency boost/penalty

    # LLM model for scoring, extraction (defaults to container default)
    intelligence_model: str | None = None

    # Pruning: memories where importance + decay < threshold are deleted
    prune_threshold: float = 0.15


# =============================================================================
# IntelligentMemoryClient
# =============================================================================


class IntelligentMemoryClient(MemoryClient):
    """
    MemoryClient with importance scoring, entity memory, time decay, and user profiles.

    Drop-in replacement for MemoryClient. All existing methods work identically;
    the intelligence features are transparent additions.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        intelligence_config: IntelligenceConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._intel = intelligence_config or IntelligenceConfig()

    # -------------------------------------------------------------------------
    # add() — enriched with scoring, entity extraction, user profile update
    # -------------------------------------------------------------------------

    async def add(
        self,
        messages: str | list[dict[str, Any]] | list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        custom_prompt: str | None = None,
    ) -> MemoryAddResult:
        """
        Store memories with importance score, entity extraction, and profile update.

        Importance score is merged into metadata before the base add() call so
        it is stored in Qdrant alongside the memory text. Entity memories and
        user profile updates are stored as separate tagged entries.
        """
        llm = self._get_llm()
        text = self._to_text(messages)
        enriched_meta: dict[str, Any] = dict(metadata or {})
        enriched_meta["stored_at"] = datetime.now(timezone.utc).isoformat()

        # 1. Importance scoring
        if self._intel.enable_scoring and llm:
            importance = await self._score_importance(text, llm)
            enriched_meta["importance"] = importance

        # 2. Base store (fact extraction via mem0)
        result = await super().add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            metadata=enriched_meta,
            custom_prompt=custom_prompt,
        )

        # 3. Entity extraction (stored as tagged memories in same collection)
        if self._intel.enable_entity_memory and user_id and llm:
            await self._extract_and_store_entities(text, user_id, llm)

        # 4. User profile update
        if self._intel.enable_user_profiles and user_id and llm:
            await self._update_user_profile(user_id, text, llm)

        return result

    # -------------------------------------------------------------------------
    # search() — re-ranked by blended score
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> MemorySearchResult:
        """
        Search memories and re-rank by blending semantic similarity, importance,
        and time decay.

        final_score = (semantic × 0.6) + (importance × 0.3) + (decay × 0.1)
        """
        result = await super().search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            limit=limit,
            filters=filters,
        )

        if self._intel.enable_scoring or self._intel.enable_decay:
            result = self._rerank(result)

        return result

    # -------------------------------------------------------------------------
    # Entity memory
    # -------------------------------------------------------------------------

    async def search_entities(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search entity memories for a user.

        Returns entities (people, orgs, products) matching the query,
        as a list of dicts with name, type, attributes, and score.

        Example::

            entities = await client.search_entities("vendor", user_id="user-123")
            # → [{"name": "John", "type": "person", "org": "Acme Corp", "score": 0.91}]
        """
        # Fetch more than needed so we can filter to entity-tagged entries only
        results = await super().search(query, user_id=user_id, limit=limit * 4)
        entities = [
            e for e in results.results
            if e.metadata.get("memory_type") == "entity"
        ]
        out = []
        for e in entities[:limit]:
            rec: dict[str, Any] = {
                "name": e.metadata.get("entity_name", ""),
                "type": e.metadata.get("entity_type", ""),
                "score": e.score,
                "memory": e.memory,
            }
            # Merge any extra attributes stored in metadata
            for k, v in e.metadata.items():
                if k not in ("memory_type", "entity_name", "entity_type",
                             "importance", "stored_at"):
                    rec[k] = v
            out.append(rec)
        return out

    # -------------------------------------------------------------------------
    # User profile
    # -------------------------------------------------------------------------

    async def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """
        Retrieve the structured user knowledge profile.

        Returns a dict with keys like preferences, employer, expertise_level,
        communication_style, last_topics, or None if no profile exists yet.

        Example::

            profile = await client.get_user_profile("user-123")
            # → {
            #     "preferences": ["dark mode", "Python"],
            #     "employer": "Google",
            #     "expertise_level": "senior engineer",
            #     "communication_style": "concise",
            #     "last_topics": ["Temporal", "MCP"],
            # }
        """
        results = await super().search(
            "user profile preferences expertise employer",
            user_id=user_id,
            limit=10,
        )
        profile_entries = [
            e for e in results.results
            if e.metadata.get("memory_type") == "user_profile"
        ]
        if not profile_entries:
            return None

        # Use the most recently stored profile entry
        latest = max(
            profile_entries,
            key=lambda e: e.metadata.get("stored_at", ""),
        )
        # Profile JSON is stored in metadata["profile_json"] (mem0 rewrites
        # the memory text through LLM fact extraction, so we cannot rely on it).
        profile_json = latest.metadata.get("profile_json")
        if profile_json:
            try:
                return json.loads(profile_json)
            except Exception:
                pass
        return None

    # -------------------------------------------------------------------------
    # Pruning
    # -------------------------------------------------------------------------

    async def prune(
        self,
        *,
        user_id: str,
        threshold: float | None = None,
    ) -> int:
        """
        Delete memories whose relevance score (importance + decay) falls below threshold.

        Old, low-importance memories are pruned. Entity memories and user profile
        entries are never pruned automatically.

        Args:
            user_id: User whose memories to prune.
            threshold: Score below which a memory is pruned. Defaults to
                       IntelligenceConfig.prune_threshold (0.15).

        Returns:
            Number of memories deleted.
        """
        cutoff = threshold if threshold is not None else self._intel.prune_threshold
        entries = await self.get_all(user_id=user_id)
        now = datetime.now(timezone.utc)
        pruned = 0

        for entry in entries:
            # Never prune structured entries
            mtype = entry.metadata.get("memory_type")
            if mtype in ("entity", "user_profile"):
                continue

            importance = float(entry.metadata.get("importance", 0.5))
            decay = self._time_decay(entry, now)

            if importance + decay < cutoff:
                try:
                    await self.delete(entry.id)
                    pruned += 1
                    logger.debug(
                        f"IntelligentMemoryClient: pruned memory '{entry.id}' "
                        f"(importance={importance:.2f}, decay={decay:.2f})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to prune memory '{entry.id}': {e}")

        logger.info(
            f"IntelligentMemoryClient: pruned {pruned} memories for user '{user_id}'"
        )
        return pruned

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _rerank(self, result: MemorySearchResult) -> MemorySearchResult:
        """Re-rank search results by blending semantic similarity, importance, and decay."""
        now = datetime.now(timezone.utc)
        for entry in result.results:
            sem = entry.score if entry.score is not None else 0.5
            importance = float(entry.metadata.get("importance", 0.5))
            decay = self._time_decay(entry, now) if self._intel.enable_decay else 0.0

            entry.score = (
                sem * self._intel.semantic_weight
                + importance * self._intel.importance_weight
                + decay * self._intel.decay_weight
            )

        result.results.sort(key=lambda e: e.score or 0.0, reverse=True)
        return result

    def _time_decay(self, entry: MemoryEntry, now: datetime) -> float:
        """
        Return a recency modifier in range [-0.2, +0.3].

        < 1 day   → +0.3
        1-7 days  → +0.1
        7-90 days →  0.0
        > 90 days → -0.2
        """
        raw = entry.metadata.get("stored_at") or entry.created_at
        if not raw:
            return 0.0

        try:
            if isinstance(raw, str):
                stored = datetime.fromisoformat(raw)
            elif isinstance(raw, datetime):
                stored = raw
            else:
                return 0.0

            # Make both timezone-aware or both naive
            if stored.tzinfo is None:
                stored = stored.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)

            age_days = (now - stored).days
        except Exception:
            return 0.0

        if age_days < 1:
            return 0.3
        elif age_days < 7:
            return 0.1
        elif age_days < 90:
            return 0.0
        else:
            return -0.2

    async def _score_importance(self, text: str, llm: Any) -> float:
        """
        LLM call to assign importance score 0.0–1.0.

        Low (0.0–0.3):  trivial, casual, or transient facts
        Medium (0.4–0.6): useful context
        High (0.7–1.0): key facts, decisions, relationships, critical events
        """
        from orchestrator.llm.config import LLMConfig

        _label_map = {"trivial": 0.1, "low": 0.25, "medium": 0.5, "high": 0.8, "critical": 0.95}

        model = self._intel.intelligence_model or self._get_default_model()
        prompt = (
            "Classify the importance of this text for future reference.\n\n"
            f"Text: {text[:500]}\n\n"
            "Choose exactly one label:\n"
            "  trivial  — casual, temporary, or irrelevant ('User said okay')\n"
            "  low      — minor preference or passing remark\n"
            "  medium   — useful context (tools, work style, habits)\n"
            "  high     — significant fact (role, employer, key decision, relationship)\n"
            "  critical — major life/career event (promotion, acquisition, critical change)\n\n"
            "Reply with ONLY one word from the list above."
        )
        try:
            response = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(model=model, temperature=0.1, max_tokens=16),
                auto_session=False,
            )
            label = (response.content or "").strip().lower()
            # Match any of the known labels (handles extra punctuation/whitespace)
            for key, val in _label_map.items():
                if key in label:
                    return val
            return 0.5
        except Exception as e:
            logger.debug(f"Importance scoring failed: {e}")
            return 0.5

    async def _extract_and_store_entities(
        self,
        text: str,
        user_id: str,
        llm: Any,
    ) -> None:
        """
        Extract named entities from text and store each as a tagged memory.

        Entity memories are tagged with memory_type="entity" and include
        entity_name, entity_type, and any extracted attributes in metadata.
        They are stored with importance=0.8 (entities are always high-value).
        """
        from orchestrator.llm.config import LLMConfig

        model = self._intel.intelligence_model or self._get_default_model()
        prompt = (
            "Extract named entities from the text below.\n"
            "Entity types: person, organization, product, location\n\n"
            f"Text: {text[:800]}\n\n"
            "Reply ONLY with JSON in this exact format:\n"
            '{"entities": [{"name": "...", "type": "...", "attributes": {"key": "value"}}]}\n'
            "Use attributes for context (e.g. role, org, location). "
            'Return {"entities": []} if no entities found.'
        )
        try:
            response = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(model=model, temperature=0.1, max_tokens=1000),
                auto_session=False,
            )
            data = self._extract_json(response.content or '{"entities": []}')
            entities = data.get("entities", []) if isinstance(data, dict) else []
        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")
            return

        for entity in entities:
            name = entity.get("name", "").strip()
            etype = entity.get("type", "unknown")
            attrs = entity.get("attributes", {})
            if not name:
                continue

            # Build a readable memory string for vector search
            attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items()) if attrs else ""
            memory_text = f"Entity: {name} ({etype})"
            if attr_str:
                memory_text += f" — {attr_str}"

            entity_meta: dict[str, Any] = {
                "memory_type": "entity",
                "entity_name": name,
                "entity_type": etype,
                "importance": 0.8,
                "stored_at": datetime.now(timezone.utc).isoformat(),
            }
            entity_meta.update(attrs)

            try:
                await super().add(
                    memory_text,
                    user_id=user_id,
                    metadata=entity_meta,
                )
            except Exception as e:
                logger.debug(f"Failed to store entity '{name}': {e}")

    async def _update_user_profile(
        self,
        user_id: str,
        text: str,
        llm: Any,
    ) -> None:
        """
        Extract user facts from conversation text and merge into the user profile.

        The profile is a JSON document stored as a tagged memory entry.
        Each update fetches the latest profile, merges new facts, and stores
        the updated version.
        """
        from orchestrator.llm.config import LLMConfig

        existing = await self.get_user_profile(user_id)
        existing_str = json.dumps(existing) if existing else "{}"

        model = self._intel.intelligence_model or self._get_default_model()
        prompt = (
            "Extract NEW facts about the user from the conversation below. "
            "Return ONLY the keys that have new or changed information.\n\n"
            f"Conversation: {text[:500]}\n\n"
            f"Existing profile: {existing_str}\n\n"
            "Keys: preferences (list), employer (string), expertise_level (string), "
            "communication_style (string), last_topics (list)\n\n"
            "Reply with a single compact JSON object containing only updated keys. "
            "Example: {\"employer\":\"Stripe\",\"preferences\":[\"Python\",\"Go\"]}\n"
            "Return {} if nothing new."
        )
        try:
            import asyncio as _asyncio
            await _asyncio.sleep(1.5)  # pause to avoid rate-limiting after rapid scoring + entity calls
            response = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(model=model, temperature=0.1, max_tokens=300),
                auto_session=False,
            )
            raw = response.content or ""
            if not raw:
                logger.debug("User profile update: LLM returned empty response, skipping")
                return
            logger.debug(f"User profile raw response: {repr(raw[:200])}")
            parsed = self._extract_json(raw)
            # Only accept a dict — reject arrays or other types from partial parses
            delta = parsed if isinstance(parsed, dict) else {}
        except Exception as e:
            logger.debug(f"User profile update failed: {e}")
            return

        if not delta:
            logger.debug("User profile update: no new facts found, skipping store")
            return

        # Merge delta into existing profile (new values win; lists are unioned)
        updated = dict(existing or {})
        for k, v in delta.items():
            if isinstance(v, list) and isinstance(updated.get(k), list):
                seen = set()
                merged = []
                for item in updated[k] + v:
                    if item not in seen:
                        seen.add(item)
                        merged.append(item)
                updated[k] = merged
            else:
                updated[k] = v

        profile_meta: dict[str, Any] = {
            "memory_type": "user_profile",
            "importance": 1.0,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            # Store the full JSON in metadata so it survives mem0's fact extraction.
            # mem0 rewrites the memory text through LLM, but passes metadata through
            # unchanged — so we read back from metadata["profile_json"], not memory text.
            "profile_json": json.dumps(updated),
        }
        # Build a descriptive summary so mem0 extracts at least one fact
        # (mem0 only stores entries when it can extract facts from the text).
        summary_parts = []
        if updated.get("employer"):
            summary_parts.append(f"works at {updated['employer']}")
        if updated.get("expertise_level"):
            summary_parts.append(f"expertise: {updated['expertise_level']}")
        if updated.get("preferences"):
            summary_parts.append(f"preferences: {', '.join(str(p) for p in updated['preferences'])}")
        if updated.get("communication_style"):
            summary_parts.append(f"communication style: {updated['communication_style']}")
        if updated.get("last_topics"):
            summary_parts.append(f"recent topics: {', '.join(str(t) for t in updated['last_topics'])}")
        profile_summary = "User profile — " + ("; ".join(summary_parts) if summary_parts else "updated")

        try:
            # infer=False bypasses mem0's LLM fact extraction so the profile
            # summary is stored verbatim with our metadata (including profile_json).
            # Without this, mem0 deduplicates extracted facts against existing
            # memories and may store nothing if the facts are already known.
            await super().add(
                [{"role": "user", "content": profile_summary}],
                user_id=user_id,
                metadata=profile_meta,
                infer=False,
            )
            logger.debug(f"User profile updated for '{user_id}'")
        except Exception as e:
            logger.debug(f"Failed to store user profile for '{user_id}': {e}")

    def _get_llm(self) -> Any | None:
        """Lazy-load LLM client from the container."""
        try:
            from orchestrator.core.container import get_container
            return get_container().llm_client
        except Exception:
            return None

    def _get_default_model(self) -> str:
        from orchestrator.config import settings
        return settings.default_llm_model

    @staticmethod
    def _extract_json(content: str) -> Any:
        """
        Extract and parse a JSON object or array from LLM response content.

        Handles:
        - Markdown code fences (```json ... ```)
        - Bare JSON
        - Truncated JSON (cuts off at last complete key-value pair for dicts)
        Raises json.JSONDecodeError if no valid JSON can be recovered.
        """
        # Strip markdown fences
        stripped = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped.strip())

        # Try direct parse first
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Try to extract first complete JSON object via brace matching
        depth = 0
        start = -1
        for i, ch in enumerate(stripped):
            if ch == '{':
                if start == -1:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(stripped[start:i + 1])
                    except json.JSONDecodeError:
                        break

        # Recover truncated JSON object containing an array
        # e.g. {"entities": [{"name": "A"}, {"name": "B"}, {"name": "C  ← cut off
        # Collect all complete {...} items in the array before the truncation point.
        obj_start = stripped.find("{")
        if obj_start != -1:
            fragment = stripped[obj_start:]
            key_arr = re.search(r'"(\w+)"\s*:\s*\[', fragment)
            if key_arr:
                key = key_arr.group(1)
                arr_pos = key_arr.end() - 1  # position of '['
                items: list[Any] = []
                depth = 0
                item_start = -1
                for i, ch in enumerate(fragment):
                    if i < arr_pos:
                        continue
                    if ch == "{":
                        if depth == 0:
                            item_start = i
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and item_start != -1:
                            try:
                                items.append(json.loads(fragment[item_start : i + 1]))
                            except json.JSONDecodeError:
                                pass
                            item_start = -1
                if items:
                    return {key: items}

        # Recover truncated JSON — cut at last comma (or close open string if no
        # comma), then close all open brackets in reverse order. Handles:
        #   {"k": "v", "k2":       → cut at comma → {"k": "v"}
        #   {"prefs": ["a", "b     → cut at comma inside array → {"prefs": ["a"]}
        #   {"employer": "Stripe   → no comma, close open string → {"employer": "Stripe"}
        obj_start = stripped.find("{")
        if obj_start != -1:
            fragment = stripped[obj_start:]
            last_clean = fragment.rfind(",")

            if last_clean > 0:
                cut = fragment[:last_clean]
            else:
                # No comma — the fragment has at most one key-value pair, truncated
                # mid-value. Close any open string first (odd quote count → unclosed).
                cut = fragment
                if cut.count('"') % 2 == 1:
                    cut = cut + '"'

            # Close any unclosed brackets in reverse nesting order
            closers: list[str] = []
            in_str = False
            esc = False
            for ch in cut:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    closers.append("}")
                elif ch == "[":
                    closers.append("]")
                elif ch == "}" and closers and closers[-1] == "}":
                    closers.pop()
                elif ch == "]" and closers and closers[-1] == "]":
                    closers.pop()

            closing = "".join(reversed(closers))
            truncated = cut + closing
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

        # Fall back to first JSON array via brace matching
        depth = 0
        start = -1
        for i, ch in enumerate(stripped):
            if ch == '[':
                if start == -1:
                    start = i
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(stripped[start:i + 1])
                    except json.JSONDecodeError:
                        break

        raise json.JSONDecodeError("No valid JSON found", content, 0)

    @staticmethod
    def _to_text(messages: str | list[dict[str, Any]] | list[str]) -> str:
        """Convert messages to a flat text string for LLM processing."""
        if isinstance(messages, str):
            return messages
        parts = []
        for m in messages:
            if isinstance(m, str):
                parts.append(m)
            elif isinstance(m, dict):
                parts.append(m.get("content", ""))
        return "\n".join(parts)
