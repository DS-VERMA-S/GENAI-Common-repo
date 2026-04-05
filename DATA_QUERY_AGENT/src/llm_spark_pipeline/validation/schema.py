from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class FilterCondition(BaseModel):
    column: str
    operator: str = Field(
        ...,
        description="One of: =, !=, >, >=, <, <=, in, not_in, contains",
    )
    value: str


class Aggregation(BaseModel):
    column: str
    function: str = Field(..., description="One of: sum, avg, min, max, count")
    alias: Optional[str] = None


class OrderBy(BaseModel):
    column: str
    direction: str = Field("asc", description="asc or desc")


class ActionPlan(BaseModel):
    dataset: str
    select: Optional[List[str]] = None
    filters: Optional[List[FilterCondition]] = None
    aggregations: Optional[List[Aggregation]] = None
    group_by: Optional[List[str]] = None
    order_by: Optional[List[OrderBy]] = None
    limit: Optional[int] = None

    @field_validator("order_by")
    @classmethod
    def _validate_order(cls, v):
        if not v:
            return v
        for item in v:
            if item.direction not in ("asc", "desc"):
                raise ValueError("order_by.direction must be asc or desc")
        return v

