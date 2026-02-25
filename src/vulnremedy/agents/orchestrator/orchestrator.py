"""
VulnRemedy LangGraph Orchestrator.

Replaces the manual sequential pipeline with a declarative StateGraph that
provides:
  - Conditional routing    — short-circuits cleanly when there is nothing to do
  - Checkpointed state     — every node's output is persisted in MemorySaver
  - Human-in-the-loop      — interrupt_before=["pr_creator"] pauses the graph
                             so plans can be reviewed before PRs are opened
  - Introspection          — graph.get_state() reveals exactly where a run stands

Graph topology:
                        ┌────────────────────────────────────────────────────┐
  START ──► scanner ──► │ route_after_scan                                    │
                        │   dependencies?  ──yes──► analyst                  │
                        │                  ──no───► END                      │
                        └────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                        ┌────────────────────────────────────────────────────┐
                        │ route_after_analyst                                 │
                        │   findings?      ──yes──► impact                   │
                        │                  ──no───► END                      │
                        └────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                        ┌────────────────────────────────────────────────────┐
                        │ route_after_impact                                  │
                        │   CRITICAL/HIGH? ──yes──► planner                  │
                        │                  ──no───► END                      │
                        └────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                        ┌────────────────────────────────────────────────────┐
                        │ route_after_planner                                 │
                        │   plans?         ──yes──► pr_creator  ◄─ INTERRUPT │
                        │                  ──no───► END                      │
                        └────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                                                       END

Human-approval workflow (auto_approve=False):
  1. orchestrator.run()  → graph pauses BEFORE pr_creator, returns interrupted state
  2. (human reviews plans via API)
  3. orchestrator.approve_and_resume(scan_id)  → graph resumes, PRs are created

Auto-approve mode (auto_approve=True):
  1. orchestrator.run()  → planner node auto-approves all DRAFT plans
  2. graph pauses at interrupt → .run() immediately calls approve_and_resume()
  3. Full pipeline runs without human intervention
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from vulnremedy.agents.cve_analyst.agent import CVEAnalystAgent
from vulnremedy.agents.impact_assessor.agent import ImpactAssessorAgent
from vulnremedy.agents.pr_creator.agent import PRCreatorAgent
from vulnremedy.agents.remediation.agent import RemediationPlannerAgent
from vulnremedy.agents.scanner.agent import ScannerAgent
from vulnremedy.models.impact import Priority
from vulnremedy.models.remediation import RemediationStatus
from vulnremedy.utils.logging import logger


# ─── State ────────────────────────────────────────────────────────────────────


class VulnRemediationState(TypedDict):
    """
    Shared state threaded through every node in the graph.

    Fields populated by each stage:
      scanner     → dependencies
      analyst     → findings
      impact      → impact_reports
      planner     → remediation_plans
      pr_creator  → prs_created

    errors accumulates across all nodes (using operator.add as reducer).
    short_circuit is set by routing functions when the pipeline stops early.
    """

    # Input (set once at graph start)
    repo: str
    branch: str
    dry_run: bool
    auto_approve: bool
    scan_id: str

    # Stage outputs
    dependencies: list[Any]
    findings: list[Any]
    impact_reports: list[Any]
    remediation_plans: list[Any]
    prs_created: list[Any]

    # Metadata
    short_circuit: Optional[str]
    errors: Annotated[list[str], operator.add]   # appends across nodes


# ─── Routing functions ────────────────────────────────────────────────────────
# Each returns the name of the next node to execute, or "end".
# These are passed to add_conditional_edges().


def _route_after_scan(state: VulnRemediationState) -> str:
    if not state.get("dependencies"):
        return "end"
    return "analyst"


def _route_after_analyst(state: VulnRemediationState) -> str:
    if not state.get("findings"):
        return "end"
    return "impact"


def _route_after_impact(state: VulnRemediationState) -> str:
    actionable = [
        r for r in state.get("impact_reports", [])
        if r.priority in (Priority.CRITICAL, Priority.HIGH)
    ]
    if not actionable:
        return "end"
    return "planner"


def _route_after_planner(state: VulnRemediationState) -> str:
    if not state.get("remediation_plans"):
        return "end"
    return "pr_creator"


# ─── Graph builder ────────────────────────────────────────────────────────────


def build_graph(
    scanner: Any = None,
    analyst: Any = None,
    assessor: Any = None,
    planner: Any = None,
    pr_creator: Any = None,
    checkpointer: Any = None,
) -> Any:
    """
    Build and compile the VulnRemedy StateGraph.

    Accepts optional agent instances for dependency injection (testing/mocking).
    When not provided, each agent is instantiated with default settings.

    The graph is compiled with:
      - interrupt_before=["pr_creator"]  → pauses for human approval
      - checkpointer (MemorySaver)       → persists state across invocations

    Args:
        scanner, analyst, assessor, planner, pr_creator:
            Optional agent instances. Pass mock objects for testing.
        checkpointer:
            LangGraph checkpointer. Required for interrupt/resume to work.
            Defaults to None (graph still works but cannot be resumed).

    Returns:
        Compiled LangGraph CompiledGraph.
    """
    _scanner = scanner or ScannerAgent()
    _analyst = analyst or CVEAnalystAgent()
    _assessor = assessor or ImpactAssessorAgent()
    _planner = planner or RemediationPlannerAgent()
    _pr_creator = pr_creator or PRCreatorAgent()

    # ── Node definitions ──────────────────────────────────────────────────────
    # Each node receives the full state and returns a partial update dict.
    # LangGraph merges the returned dict into the running state.

    def scanner_node(state: VulnRemediationState) -> dict[str, Any]:
        logger.info("LangGraph node: scanner", repo=state["repo"], branch=state["branch"])
        result = _scanner.scan(state["repo"], state["branch"])
        return {
            "dependencies": result.get("dependencies", []),
            "errors": result.get("errors", []),
        }

    def analyst_node(state: VulnRemediationState) -> dict[str, Any]:
        logger.info(
            "LangGraph node: cve_analyst",
            dependencies=len(state.get("dependencies", [])),
        )
        result = _analyst.analyze(
            dependencies=state["dependencies"],
            scan_id=state.get("scan_id"),
            repository=state["repo"],
            branch=state["branch"],
        )
        return {
            "findings": result.get("findings", []),
            "errors": result.get("errors", []),
        }

    def impact_node(state: VulnRemediationState) -> dict[str, Any]:
        logger.info(
            "LangGraph node: impact_assessor",
            findings=len(state.get("findings", [])),
        )
        result = _assessor.assess(
            findings=state["findings"],
            scan_id=state.get("scan_id"),
        )
        return {
            "impact_reports": result.get("impact_reports", []),
            "errors": result.get("errors", []),
        }

    def planner_node(state: VulnRemediationState) -> dict[str, Any]:
        logger.info(
            "LangGraph node: remediation_planner",
            impact_reports=len(state.get("impact_reports", [])),
        )
        result = _planner.plan(
            impact_reports=state["impact_reports"],
            scan_id=state.get("scan_id"),
        )
        plans = result.get("plans", [])

        # Auto-approve DRAFT plans if requested — before the interrupt gate
        if state.get("auto_approve"):
            for p in plans:
                if p.status == RemediationStatus.DRAFT:
                    p.approve(approver="pipeline-auto-approve")
            logger.info(
                "Auto-approved plans before pr_creator gate",
                total=len(plans),
            )

        return {
            "remediation_plans": plans,
            "errors": result.get("errors", []),
        }

    def pr_creator_node(state: VulnRemediationState) -> dict[str, Any]:
        logger.info(
            "LangGraph node: pr_creator",
            plans=len(state.get("remediation_plans", [])),
            dry_run=state.get("dry_run", True),
        )
        result = _pr_creator.create_prs(
            plans=state["remediation_plans"],
            dry_run=state.get("dry_run", True),
        )
        return {
            "prs_created": result.get("prs_created", []),
            "errors": result.get("errors", []),
        }

    # ── Graph assembly ────────────────────────────────────────────────────────

    workflow = StateGraph(VulnRemediationState)

    # Add nodes
    workflow.add_node("scanner", scanner_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("impact", impact_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("pr_creator", pr_creator_node)

    # Entry edge
    workflow.add_edge(START, "scanner")

    # Conditional edges with routing functions
    workflow.add_conditional_edges(
        "scanner",
        _route_after_scan,
        {"analyst": "analyst", "end": END},
    )
    workflow.add_conditional_edges(
        "analyst",
        _route_after_analyst,
        {"impact": "impact", "end": END},
    )
    workflow.add_conditional_edges(
        "impact",
        _route_after_impact,
        {"planner": "planner", "end": END},
    )
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"pr_creator": "pr_creator", "end": END},
    )
    workflow.add_edge("pr_creator", END)

    # Compile — interrupt BEFORE pr_creator so plans can be reviewed
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["pr_creator"],
    )


# ─── Orchestrator class ───────────────────────────────────────────────────────


class VulnRemedyOrchestrator:
    """
    High-level wrapper around the compiled LangGraph.

    Manages scan lifecycles including human-approval workflows.

    Usage:
        orchestrator = VulnRemedyOrchestrator()

        # Auto-approve + dry-run (good for testing)
        result = orchestrator.run("apache/log4j", dry_run=True, auto_approve=True)

        # Human-in-the-loop
        result = orchestrator.run("apache/log4j", dry_run=False, auto_approve=False)
        # → status="awaiting_approval" — human reviews result["remediation_plans"]
        result = orchestrator.approve_and_resume(result["scan_id"])
        # → status="complete"
    """

    def __init__(
        self,
        scanner: Any = None,
        analyst: Any = None,
        assessor: Any = None,
        planner: Any = None,
        pr_creator: Any = None,
    ) -> None:
        self._checkpointer = MemorySaver()
        self._graph = build_graph(
            scanner=scanner,
            analyst=analyst,
            assessor=assessor,
            planner=planner,
            pr_creator=pr_creator,
            checkpointer=self._checkpointer,
        )
        logger.info("VulnRemedyOrchestrator initialized")

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        repo: str,
        branch: str = "main",
        dry_run: bool = True,
        auto_approve: bool = False,
        scan_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Start a vulnerability scan and remediation pipeline.

        If auto_approve=False: graph pauses before pr_creator.
          Returned dict will have status="awaiting_approval".
          Call approve_and_resume(scan_id) to continue.

        If auto_approve=True: plans are approved in the planner node,
          and pr_creator runs immediately after.
          Returned dict will have status="complete".

        Args:
            repo:         Repository "owner/repo" or full GitHub URL.
            branch:       Branch to scan.
            dry_run:      If True, PR Creator does not call GitHub API.
            auto_approve: If True, all DRAFT plans are auto-approved.
            scan_id:      Optional UUID string. Auto-generated if not provided.

        Returns:
            Scan result dict with keys:
              scan_id, status, summary, remediation_plans, prs_created, errors.
        """
        scan_id = scan_id or str(uuid4())
        config = {"configurable": {"thread_id": scan_id}}

        initial_state: VulnRemediationState = {
            "repo": repo,
            "branch": branch,
            "dry_run": dry_run,
            "auto_approve": auto_approve,
            "scan_id": scan_id,
            "dependencies": [],
            "findings": [],
            "impact_reports": [],
            "remediation_plans": [],
            "prs_created": [],
            "short_circuit": None,
            "errors": [],
        }

        logger.info(
            "Orchestrator: starting scan",
            repo=repo,
            branch=branch,
            scan_id=scan_id,
            dry_run=dry_run,
            auto_approve=auto_approve,
        )

        # First graph invocation — may interrupt before pr_creator
        self._graph.invoke(initial_state, config=config)

        # Check if the graph paused at the interrupt point
        snapshot = self._graph.get_state(config)
        interrupted = bool(snapshot.next)

        if auto_approve and interrupted:
            # Auto-approve all plans then resume immediately
            return self.approve_and_resume(scan_id)

        return self._build_response(snapshot.values, scan_id, interrupted=interrupted)

    def approve_and_resume(self, scan_id: str) -> dict[str, Any]:
        """
        Approve all pending DRAFT plans and resume the graph.

        Call this after a run() that returned status="awaiting_approval".

        Args:
            scan_id: The scan ID returned by run().

        Returns:
            Updated scan result dict with status="complete".

        Raises:
            ValueError: If no scan with this scan_id exists.
            RuntimeError: If the scan is not in an interrupted state.
        """
        config = {"configurable": {"thread_id": scan_id}}
        snapshot = self._graph.get_state(config)

        if not snapshot.values:
            raise ValueError(f"No scan found with scan_id={scan_id!r}")

        if not snapshot.next:
            # Graph already finished — return current state
            logger.warning(
                "approve_and_resume called on a completed scan",
                scan_id=scan_id,
            )
            return self._build_response(snapshot.values, scan_id, interrupted=False)

        # Approve all DRAFT plans in the checkpointed state
        plans = snapshot.values.get("remediation_plans", [])
        approved_count = 0
        for plan in plans:
            if plan.status == RemediationStatus.DRAFT:
                plan.approve(approver="api-user")
                approved_count += 1

        # Flush the updated plans back into the checkpointer
        self._graph.update_state(config, {"remediation_plans": plans})

        logger.info(
            "Orchestrator: approved plans, resuming graph",
            scan_id=scan_id,
            approved=approved_count,
        )

        # Resume (None initial state = continue from checkpoint)
        self._graph.invoke(None, config=config)

        snapshot = self._graph.get_state(config)
        return self._build_response(snapshot.values, scan_id, interrupted=False)

    def get_status(self, scan_id: str) -> dict[str, Any]:
        """
        Return the current state of a scan (running, interrupted, or complete).

        Args:
            scan_id: The scan ID returned by run().

        Returns:
            Scan status dict. Returns {"error": "not found"} if scan_id is unknown.
        """
        config = {"configurable": {"thread_id": scan_id}}
        snapshot = self._graph.get_state(config)

        if not snapshot.values:
            return {"scan_id": scan_id, "status": "not_found", "error": f"No scan found: {scan_id}"}

        interrupted = bool(snapshot.next)
        return self._build_response(snapshot.values, scan_id, interrupted=interrupted)

    # ── Result assembly ───────────────────────────────────────────────────────

    @staticmethod
    def _build_response(
        state: dict[str, Any],
        scan_id: str,
        interrupted: bool,
    ) -> dict[str, Any]:
        """Convert raw LangGraph state into a clean API-friendly response dict."""
        plans = state.get("remediation_plans", [])
        prs = state.get("prs_created", [])
        impact_reports = state.get("impact_reports", [])
        actionable = sum(
            1 for r in impact_reports
            if r.priority in (Priority.CRITICAL, Priority.HIGH)
        )

        status = "awaiting_approval" if interrupted else "complete"

        return {
            "scan_id": scan_id,
            "status": status,
            "repository": state.get("repo", ""),
            "branch": state.get("branch", ""),
            "summary": {
                "dependencies_found": len(state.get("dependencies", [])),
                "vulnerabilities_found": len(state.get("findings", [])),
                "impact_reports": len(impact_reports),
                "actionable_reports": actionable,
                "plans_created": len(plans),
                "plans_approved": sum(
                    1 for p in plans if p.status in (
                        RemediationStatus.APPROVED, RemediationStatus.PR_CREATED
                    )
                ),
                "prs_created": len(prs),
                "errors_total": len(state.get("errors", [])),
            },
            "remediation_plans": plans,
            "prs_created": prs,
            "errors": state.get("errors", []),
        }
