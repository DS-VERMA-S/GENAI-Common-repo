from __future__ import annotations

from typing import List

from llm_spark_pipeline.validation.schema import ActionPlan


def validate_plan(plan: ActionPlan, metadata: dict) -> List[str]:
    errors: List[str] = []
    columns = {col["name"] for col in metadata.get("columns", [])}
    allowed_actions = set(metadata.get("allowed_actions", []))

    if plan.select:
        if "select" not in allowed_actions:
            errors.append("select not allowed by metadata")
        invalid = [c for c in plan.select if c not in columns]
        if invalid:
            errors.append(f"invalid select columns: {invalid}")

    if plan.filters:
        if "filter" not in allowed_actions:
            errors.append("filter not allowed by metadata")
        invalid = [f.column for f in plan.filters if f.column not in columns]
        if invalid:
            errors.append(f"invalid filter columns: {invalid}")

    if plan.aggregations:
        if "aggregate" not in allowed_actions:
            errors.append("aggregate not allowed by metadata")
        invalid = [a.column for a in plan.aggregations if a.column not in columns]
        if invalid:
            errors.append(f"invalid aggregation columns: {invalid}")

    if plan.group_by:
        if "group_by" not in allowed_actions:
            errors.append("group_by not allowed by metadata")
        invalid = [c for c in plan.group_by if c not in columns]
        if invalid:
            errors.append(f"invalid group_by columns: {invalid}")

    if plan.order_by:
        if "order_by" not in allowed_actions:
            errors.append("order_by not allowed by metadata")
        invalid = [o.column for o in plan.order_by if o.column not in columns]
        if invalid:
            errors.append(f"invalid order_by columns: {invalid}")

    if plan.limit is not None:
        if "limit" not in allowed_actions:
            errors.append("limit not allowed by metadata")
        if plan.limit <= 0:
            errors.append("limit must be positive")

    if plan.dataset != metadata.get("dataset"):
        errors.append("dataset mismatch with metadata")

    return errors

