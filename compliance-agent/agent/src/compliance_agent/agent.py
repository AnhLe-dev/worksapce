"""Bounded orchestration over an injected review model and toolset."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from compliance_agent.models import ReviewRequest, ReviewResult
from compliance_agent.ports import ReviewModel
from compliance_agent.prompt import REVIEW_SYSTEM_PROMPT
from compliance_agent.tool_schemas import (
    ReviewToolExecution,
    ReviewToolName,
)
from compliance_agent.tools import ReviewToolset


class AgentExecutionError(RuntimeError):
    """Base class for technical bounded-agent failures."""


class AgentBudgetExceededError(AgentExecutionError):
    """Raised when a model or tool-call budget is exhausted."""


class AgentTimeoutError(AgentExecutionError):
    """Raised when the total agent execution deadline expires."""


class AgentResultBindingError(AgentExecutionError):
    """Raised when the model changes trusted request identities."""


class AgentWorkflowError(AgentExecutionError):
    """Raised when the model skips a required bounded tool step."""


@dataclass(frozen=True, slots=True)
class AgentLimits:
    max_model_calls: int = 8
    max_total_tool_calls: int = 40
    max_review_unit_calls: int = 1
    max_knowledge_search_calls: int = 20
    max_reference_calls: int = 20
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        values = (
            self.max_model_calls,
            self.max_total_tool_calls,
            self.max_review_unit_calls,
            self.max_knowledge_search_calls,
            self.max_reference_calls,
            self.timeout_seconds,
        )

        if any(value <= 0 for value in values):
            raise ValueError("Every agent limit must be positive")


@dataclass(frozen=True, slots=True)
class AgentExecution:
    result: ReviewResult
    tool_executions: tuple[ReviewToolExecution, ...]
    model_call_count: int


class BoundedReviewAgent:
    """Execute a model loop with deterministic identity and call limits."""

    def __init__(
        self,
        *,
        model: ReviewModel,
        limits: AgentLimits | None = None,
    ) -> None:
        self._model = model
        self._limits = limits or AgentLimits()

    async def run(
        self,
        *,
        request: ReviewRequest,
        tools: ReviewToolset,
    ) -> AgentExecution:
        try:
            async with asyncio.timeout(self._limits.timeout_seconds):
                return await self._run_loop(request=request, tools=tools)
        except TimeoutError as error:
            raise AgentTimeoutError(
                "Review agent exceeded its total execution timeout"
            ) from error

    async def _run_loop(
        self,
        *,
        request: ReviewRequest,
        tools: ReviewToolset,
    ) -> AgentExecution:
        executions: list[ReviewToolExecution] = []
        call_counts = {name: 0 for name in ReviewToolName}
        seen_call_ids: set[str] = set()

        for model_call_count in range(1, self._limits.max_model_calls + 1):
            action = await self._model.next_action(
                system_prompt=REVIEW_SYSTEM_PROMPT,
                request=request,
                observations=tuple(
                    execution.observation for execution in executions
                ),
            )

            if action.final_result is not None:
                self._validate_required_tool_coverage(
                    request=request,
                    executions=tuple(executions),
                )
                self._validate_result_binding(
                    request=request,
                    result=action.final_result,
                )
                return AgentExecution(
                    result=action.final_result,
                    tool_executions=tuple(executions),
                    model_call_count=model_call_count,
                )

            for call in action.tool_calls:
                if call.call_id in seen_call_ids:
                    raise AgentExecutionError(
                        "Model reused a tool call ID"
                    )

                seen_call_ids.add(call.call_id)
                call_counts[call.name] += 1
                self._enforce_tool_budgets(
                    call_counts=call_counts,
                    total_calls=len(executions) + 1,
                )
                executions.append(tools.execute(call))

        raise AgentBudgetExceededError("Model-call budget was exhausted")

    @staticmethod
    def _validate_required_tool_coverage(
        *,
        request: ReviewRequest,
        executions: tuple[ReviewToolExecution, ...],
    ) -> None:
        unit_calls = tuple(
            execution
            for execution in executions
            if execution.observation.name
            is ReviewToolName.GET_REVIEW_UNITS
        )

        if len(unit_calls) != 1:
            raise AgentWorkflowError(
                "Agent must load authorized review units exactly once"
            )

        searched_unit_ids = {
            execution.artifact.review_unit_id
            for execution in executions
            if execution.observation.name
            is ReviewToolName.SEARCH_SELECTED_KNOWLEDGE
        }

        if searched_unit_ids != set(request.review_unit_ids):
            raise AgentWorkflowError(
                "Agent must search knowledge for every review unit"
            )

    def _enforce_tool_budgets(
        self,
        *,
        call_counts: dict[ReviewToolName, int],
        total_calls: int,
    ) -> None:
        limits = self._limits

        if total_calls > limits.max_total_tool_calls:
            raise AgentBudgetExceededError(
                "Total tool-call budget was exhausted"
            )

        per_tool_limits = {
            ReviewToolName.GET_REVIEW_UNITS: limits.max_review_unit_calls,
            ReviewToolName.SEARCH_SELECTED_KNOWLEDGE: (
                limits.max_knowledge_search_calls
            ),
            ReviewToolName.GET_REFERENCE_UNIT: limits.max_reference_calls,
        }

        for name, count in call_counts.items():
            if count > per_tool_limits[name]:
                raise AgentBudgetExceededError(
                    f"Tool-call budget was exhausted for {name.value}"
                )

    @staticmethod
    def _validate_result_binding(
        *,
        request: ReviewRequest,
        result: ReviewResult,
    ) -> None:
        if result.contract_id != request.contract_id:
            raise AgentResultBindingError(
                "Model changed the trusted contract ID"
            )

        if result.knowledge_base_id != request.knowledge_base_id:
            raise AgentResultBindingError(
                "Model changed the selected knowledge-base ID"
            )

        if result.question != request.question:
            raise AgentResultBindingError(
                "Model changed the original review question"
            )

        if set(result.coverage.requested_review_unit_ids) != set(
            request.review_unit_ids
        ):
            raise AgentResultBindingError(
                "Model changed the requested review-unit scope"
            )
