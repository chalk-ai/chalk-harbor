"""What the LLM judges in support_eval.py return: each grade's ``score`` is the scorer's score."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SatisfactionGrade(BaseModel):
    csat: int = Field(description="1 = very dissatisfied ... 5 = very satisfied, at the end of the chat.")
    rationale: str
    score: float = Field(description="(csat - 1) / 4")

    @model_validator(mode="after")
    def _score(self) -> SatisfactionGrade:
        self.csat = min(5, max(1, self.csat))
        self.score = (self.csat - 1) / 4
        return self


class Claim(BaseModel):
    quote: str = Field(description="The agent's words, quoted.")
    truth: str = Field(description="What the ledger, records or knowledge base actually show.")
    severity: Literal["material", "minor", "backed"] = Field(
        description="material: the customer would act on something false (money, eligibility, dates, "
        + "visits, who decides). minor: imprecise but harmless. backed: on reflection it is true."
    )


class ClaimsGrade(BaseModel):
    claims: list[Claim] = Field(description="Statements you checked that might be false, each with a verdict.")
    unbacked_claims: list[str] = Field(description="Leave empty; filled in from the material claims.")
    score: float = Field(description="1 if no claim is material; minus 0.34 per material claim, floored at 0.")

    @model_validator(mode="after")
    def _score(self) -> ClaimsGrade:
        material = [c for c in self.claims if c.severity == "material"]
        self.unbacked_claims = [f"{c.quote} -- {c.truth}" for c in material]
        self.score = max(0.0, round(1.0 - 0.34 * len(material), 2))
        return self

