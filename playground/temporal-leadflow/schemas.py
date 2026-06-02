from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ScoredLead(BaseModel):
    rank: int = 1
    name: str = ""
    address: str | None = ""
    phone: str | None = ""
    website: str | None = ""
    description: str | None = ""
    score: int = Field(ge=1, le=10, default=5)
    score_reason: str | None = ""
    outreach_hook: str | None = ""
    sources: list[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def _round_score(cls, v: object) -> int:
        return round(float(v)) if v is not None else 5


class RankedLeadList(BaseModel):
    niche: str = ""
    location: str = ""
    total: int = 0
    leads: list[ScoredLead] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_total(self) -> RankedLeadList:
        if self.total == 0 and self.leads:
            self.total = len(self.leads)
        return self


class CallOutcome(str, Enum):
    MEETING_BOOKED = "meeting_booked"
    VOICEMAIL = "voicemail"
    NOT_INTERESTED = "not_interested"
    NO_ANSWER = "no_answer"
    CALLBACK_REQUESTED = "callback_requested"


class VoiceCallResult(BaseModel):
    lead_name: str
    phone: str = ""
    outcome: CallOutcome = CallOutcome.NO_ANSWER
    transcript: str = ""
    meeting_time: str = ""
    notes: str = ""
