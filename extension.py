"""Mirror's Linear project pulse: a compact strip and F8 fullscreen dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import json
import pathlib
import time
import unicodedata
import urllib.parse
import uuid
import webbrowser
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import cast, override

from mirror import config
from mirror.cli import (
    actions,
    commands,
    extension,
    extension_sources,
    host_tools,
    keymap,
    selection,
    state,
    workspace_application,
)
from prompt_toolkit import (
    application,
    data_structures,
    filters,
    formatted_text,
    layout,
    styles,
    utils,
    widgets,
)
from prompt_toolkit.key_binding import key_bindings, key_processor
from prompt_toolkit.layout import containers, controls

ACTIVE_TYPES = ("triage", "backlog", "unstarted", "started")
COLORS = ("#8496b2", "#83b8ff", "#ffd080", "#c7a6ff", "#69e1ca", "#f3a6c6")
GLYPHS = {"triage": "◇", "backlog": "◌", "unstarted": "○", "started": "◐"}
CONFIG_KEY = str(pathlib.Path(__file__).resolve())
FIELDS = [
    "id",
    "title",
    "status",
    "statusType",
    "assignee",
    "priority",
    "url",
    "labels",
    "updatedAt",
]
type Fragments = formatted_text.StyleAndTextTuples
type ToolCaller = Callable[
    [str, Mapping[str, object]], Awaitable[host_tools.ToolResult]
]


def clean(value: object) -> str:
    """Treat service text as plain text, never terminal escape sequences."""
    if not isinstance(value, str):
        return ""
    return " ".join(
        "".join(c for c in value if not unicodedata.category(c).startswith("C")).split()
    )


def clip(value: str, width: int) -> str:
    """Clip to terminal cells, including wide Unicode titles."""
    if width <= 0:
        return ""
    if utils.get_cwidth(value) <= width:
        return value
    result = ""
    for character in value:
        if utils.get_cwidth(result + character) > width - 1:
            break
        result += character
    return result + "…"


def object_fields(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return cast(dict[str, object], value)


def linear_url(value: object, resource: str = "issue") -> str:
    """Allow canonical HTTPS Linear links, never arbitrary external URLs."""
    if not isinstance(value, str) or clean(value) != value:
        return ""
    parsed = urllib.parse.urlsplit(value)
    parts = parsed.path.split("/")
    if (
        parsed.scheme == "https"
        and parsed.netloc == "linear.app"
        and len(parts) >= 4
        and parts[1] not in {"", ".", ".."}
        and parts[2] == resource
        and parts[3]
    ):
        return value
    return ""


@dataclasses.dataclass(frozen=True)
class Project:
    id: str
    name: str
    url: str

    @classmethod
    def parse(cls, value: object) -> Project:
        row = object_fields(value)
        project = cls(
            clean(row.get("uuid") or row.get("id")),
            clean(row.get("name")),
            linear_url(row.get("url"), "project"),
        )
        if not project.id or not project.name or not project.url:
            raise ValueError("Project needs an ID, name, and Linear project URL")
        return project

    def settings(self) -> config.JsonObject:
        return {"id": self.id, "name": self.name, "url": self.url}


def save_project(root: pathlib.Path, directory: pathlib.Path, project: Project) -> None:
    """Persist under the source URL so managed-checkout updates keep the selection."""
    settings = config.Config(root.parent / "config.json")
    repositories = extension_sources.repositories(
        settings.read().get("extensions"), root
    )
    key = next(
        (
            repository.key
            for repository in repositories.values()
            if directory.name == repository.name
            and directory.is_relative_to(repository.directory)
        ),
        str(directory / "extension.py"),
    )
    settings.set_extension_config(key, {"project": project.settings()})


@dataclasses.dataclass(frozen=True)
class Workflow:
    name: str
    kind: str

    @property
    def glyph(self) -> str:
        return GLYPHS[self.kind]


def workflow_order(workflow: Workflow) -> tuple[int, str]:
    return ACTIVE_TYPES.index(workflow.kind), workflow.name.casefold()


@dataclasses.dataclass(frozen=True)
class Issue:
    identifier: str
    title: str
    status: str
    assignee: str
    priority: int
    url: str
    labels: tuple[str, ...]
    updated: str
    kind: str

    @property
    def workflow(self) -> Workflow:
        return Workflow(self.status, self.kind)

    @classmethod
    def parse(cls, value: object) -> Issue:
        row = object_fields(value)
        identifier, title, status = (
            clean(row.get("identifier") or row.get("id")),
            clean(row.get("title")),
            clean(row.get("status")),
        )
        kind = clean(row.get("statusType"))
        if not identifier or not title or not status or not kind:
            raise ValueError("Missing issue identity, title, or workflow state")
        priority = row.get("priority", 0)
        if isinstance(priority, dict):
            priority = object_fields(priority).get("value", 0)
        labels = row.get("labels")
        return cls(
            identifier,
            title,
            status,
            clean(row.get("assignee")),
            priority if isinstance(priority, int) and 0 <= priority <= 4 else 0,
            linear_url(row.get("url")),
            tuple(
                clean(label)
                for label in cast(list[object], labels)
                if isinstance(label, str)
            )
            if isinstance(labels, list)
            else (),
            clean(row.get("updatedAt"))[:10],
            kind,
        )


@dataclasses.dataclass(frozen=True)
class Snapshot:
    issues: tuple[Issue, ...] = ()
    fetched_at: datetime.datetime | None = None
    busy: bool = False
    error: str = ""
    project: Project | None = None
    workflows: tuple[Workflow, ...] = ()

    def counts(self) -> tuple[int, ...]:
        return tuple(
            sum(issue.workflow == workflow for issue in self.issues)
            for workflow in self.workflows
        )

    def color(self, workflow: Workflow) -> str:
        return COLORS[self.workflows.index(workflow) % len(COLORS)]

    def problem(self) -> str:
        if self.error:
            return ("STALE · " if self.fetched_at is not None else "") + self.error
        return ""


class SyncError(Exception):
    """A safe, actionable failure message without raw service diagnostics."""


async def linear_read(
    call_tool: ToolCaller, name: str, arguments: Mapping[str, object]
) -> object:
    """Unwrap a complete MCP response without exposing service diagnostics."""
    result = await call_tool(
        "mcp_call",
        {"calls": [{"name": f"mcp__linear__{name}", "arguments": arguments}]},
    )
    if not result.success:
        raise SyncError("Linear unavailable · check /connections and retry")
    results = object_fields(json.loads(result.content)).get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise SyncError("Unexpected MCP response · cached data kept")
    item = object_fields(cast(list[object], results)[0])
    if item.get("success") is not True:
        raise SyncError("Linear unavailable or approval denied · check /connect linear")
    if item.get("truncated") is True:
        raise SyncError("Linear response truncated · cached data kept")
    content = item.get("content")
    if not isinstance(content, str):
        raise SyncError("Unexpected Linear response · cached data kept")
    return json.loads(content)


async def linear_pages(
    call_tool: ToolCaller, resource: str, arguments: Mapping[str, object]
) -> AsyncIterator[list[object]]:
    """Yield validated pages, rejecting truncated or looping pagination."""
    request = {
        "limit": 40,
        "includeArchived": False,
        "orderBy": "updatedAt",
        **arguments,
    }
    cursors: set[str] = set()
    for _ in range(25):
        page = object_fields(await linear_read(call_tool, f"list_{resource}", request))
        items = page.get(resource)
        if not isinstance(items, list) or not isinstance(page.get("hasNextPage"), bool):
            raise SyncError("Unexpected Linear page · cached data kept")
        yield cast(list[object], items)
        if page["hasNextPage"] is False:
            return
        cursor = page.get("cursor")
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise SyncError("Invalid Linear pagination · cached data kept")
        cursors.add(cursor)
        request = {**request, "cursor": cursor}
    raise SyncError(
        "More than 1,000 results; narrow projects with /linear project <search>"
    )


async def project_pages(
    call_tool: ToolCaller, query: str = ""
) -> AsyncIterator[tuple[Project, ...]]:
    arguments: dict[str, object] = {"fields": ["id", "uuid", "name", "url"]}
    if query:
        arguments["query"] = query
    async for rows in linear_pages(call_tool, "projects", arguments):
        yield tuple(Project.parse(row) for row in rows)


async def fetch_workflows(
    call_tool: ToolCaller, project: Project
) -> tuple[Workflow, ...]:
    """Discover active workflow states across every team attached to the project."""
    metadata = object_fields(
        await linear_read(call_tool, "get_project", {"query": project.id})
    )
    teams = metadata.get("teams")
    if not isinstance(teams, list) or not teams:
        raise SyncError("Project has no visible teams; cannot discover its workflow")
    workflows: set[Workflow] = set()
    team_ids = {
        clean(object_fields(team).get("id")) for team in cast(list[object], teams)
    }
    if "" in team_ids:
        raise ValueError("Missing project team identity")
    for team in sorted(team_ids):
        statuses = await linear_read(call_tool, "list_issue_statuses", {"team": team})
        if not isinstance(statuses, list):
            raise ValueError("Expected team workflow states")
        for value in cast(list[object], statuses):
            row = object_fields(value)
            workflow = Workflow(clean(row.get("name")), clean(row.get("type")))
            if workflow.kind in ACTIVE_TYPES and workflow.name:
                workflows.add(workflow)
    return tuple(sorted(workflows, key=workflow_order))


async def fetch_issues(call_tool: ToolCaller, project: Project) -> tuple[Issue, ...]:
    issues: dict[str, Issue] = {}
    async for rows in linear_pages(
        call_tool, "issues", {"project": project.id, "fields": FIELDS}
    ):
        issues.update(
            {
                issue.identifier: issue
                for issue in (Issue.parse(row) for row in rows)
                if issue.kind in ACTIVE_TYPES
            }
        )
    return tuple(
        sorted(
            issues.values(),
            key=lambda issue: (
                workflow_order(issue.workflow),
                issue.priority or 5,
                issue.identifier,
            ),
        )
    )


def compact_lines(snapshot: Snapshot, width: int) -> list[Fragments]:
    name = snapshot.project.name if snapshot.project else "Linear"
    title: Fragments = [("fg:#69e1ca bold", f" ◈ {clip(name.upper(), 24)} ")]
    if snapshot.error:
        title.append(("fg:#ffb795", "! "))
    if snapshot.project is None:
        return [[*title, ("class:footer", "· F8 choose project")]]
    title.append(("class:footer", "· F8 to open "))
    remaining = width - utils.get_cwidth(formatted_text.fragment_list_to_text(title))
    for index, (count, workflow) in enumerate(
        zip(snapshot.counts(), snapshot.workflows, strict=True)
    ):
        label = clip(workflow.name, 20 if width >= 100 else 9)
        text = f" {count} {label} "
        if utils.get_cwidth(text) > remaining - 9:
            title.append(("class:footer", f" +{len(snapshot.workflows) - index} more"))
            break
        title.append((f"fg:{snapshot.color(workflow)}", text))
        remaining -= utils.get_cwidth(text)
    return [title]


class Pulse(controls.UIControl):
    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot

    @override
    def create_content(self, width: int, height: int) -> controls.UIContent:
        del height
        lines = compact_lines(self.snapshot, width)
        return controls.UIContent(
            get_line=lambda index: lines[index], line_count=1, show_cursor=False
        )


@dataclasses.dataclass(frozen=True)
class ViewState:
    snapshot: Snapshot = dataclasses.field(default_factory=Snapshot)
    visible: bool = True


@dataclasses.dataclass(frozen=True)
class Command(actions.ExtensionAction):
    mode: str = "open"
    query: str = ""
    project: Project | None = None


@dataclasses.dataclass(frozen=True)
class PickerChanged:
    selector_id: str = ""


class ProjectPicker:
    """A cancellable, incremental producer for Mirror's native searchable selector."""

    def __init__(
        self, context: extension.Context, query: str, current: Project | None
    ) -> None:
        self.context = context
        self.query = query
        self.id = f"linear-project-{uuid.uuid4().hex}"
        self.projects = (
            {current.id: current} if current is not None and not query else {}
        )
        self.choices = tuple(
            selection.Choice(project.id, project.name, project.url)
            for project in self.projects.values()
        )
        self.loading = True
        self.current = current.id if current else None
        self.suspended: tuple[str, str | None] | None = None
        self.task: asyncio.Task[None] | None = None

    def open(self) -> None:
        self.context.dispatch(
            actions.OpenSelector(
                selection.Selector(
                    id=self.id,
                    title="Choose a Linear project",
                    extension_id=LinearDashboard.id(),
                    choices=self.choices,
                    current=self.current,
                    prefiltered=True,
                    loading=self.loading,
                    on_select=lambda value: actions.RoutedExtensionAction(
                        LinearDashboard.id(),
                        Command("select", project=self.projects[value]),
                    ),
                    empty="No projects found. Try /linear project <search> "
                    "or /connect linear.",
                )
            )
        )

    def yield_to_approval(self) -> None:
        current = self.context.get_state().selector
        selected = current.selected_choice()
        self.suspended = current.query, selected.value if selected else None
        self.context.dispatch(actions.ClearSelector())

    async def resume(self) -> None:
        if self.suspended is None:
            return
        query, selected = self.suspended
        self.suspended = None
        self.open()
        self.context.dispatch(actions.SelectorSearchChanged(query))
        await self.publish()
        current = self.context.get_state().selector
        if current.id == self.id and current.query == query:
            index = next(
                (
                    i
                    for i, choice in enumerate(current.filtered_choices())
                    if choice.value == selected
                ),
                0,
            )
            self.context.dispatch(
                actions.MoveSelectorSelection(index - current.menu.selected)
            )

    def active(self) -> bool:
        return self.context.get_state().selector.id == self.id

    async def publish(self) -> None:
        if not self.active():
            return
        query = self.context.get_state().selector.query
        choices = self.choices
        matches = await asyncio.to_thread(
            selection.fuzzy_matches,
            choices,
            query,
            lambda choice: (choice.label, choice.detail),
        )
        if self.active() and choices is self.choices:
            self.context.dispatch(
                actions.SelectorResultsLoaded(self.id, query, matches, self.loading)
            )

    async def load(self) -> None:
        try:
            async with asyncio.timeout(300):
                async for page in project_pages(self.context.call_tool, self.query):
                    if not self.active() and self.suspended is None:
                        return
                    for project in page:
                        self.projects.setdefault(project.id, project)
                    self.choices = tuple(
                        selection.Choice(project.id, project.name, project.url)
                        for project in sorted(
                            self.projects.values(),
                            key=lambda project: (project.name.casefold(), project.id),
                        )
                    )
                    await self.publish()
        except SyncError as error:
            if self.active() or self.suspended is not None:
                self.context.dispatch(
                    actions.ShowText(
                        f"Project list incomplete: {error}", style="yellow"
                    )
                )
        except (OSError, RuntimeError, ValueError, TimeoutError):
            if self.active() or self.suspended is not None:
                self.context.dispatch(
                    actions.ShowText(
                        "Project list incomplete. "
                        "Check /connections or retry /linear project.",
                        style="yellow",
                    )
                )
        finally:
            self.loading = False
            await self.publish()

    async def close(self) -> None:
        self.suspended = None
        if self.active():
            self.context.dispatch(actions.ClearSelector())
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


@dataclasses.dataclass(frozen=True)
class Updated(actions.ExtensionAction):
    snapshot: Snapshot


def slash_command(context: commands.CommandContext, arguments: str) -> actions.Action:
    del context
    mode, _, query = arguments.strip().partition(" ")
    mode = mode.lower() or "open"
    if mode not in {"open", "refresh", "hide", "show", "project"} or (
        query and mode != "project"
    ):
        return actions.ShowText("Usage: /linear [project [search]|refresh|hide|show]")
    return actions.RoutedExtensionAction(
        LinearDashboard.id(), Command(mode, query.strip())
    )


class LinearDashboard(extension.Extension[ViewState, Command | PickerChanged, None]):
    supports_local = True
    supports_remote = True
    reset_on_session_open = False

    def __init__(self) -> None:
        self.snapshot = Snapshot()
        self.lock = asyncio.Lock()
        self.dashboard: Dashboard | None = None
        self.project_picker: ProjectPicker | None = None
        self.opening = False
        self.foreground: extension.Context | None = None
        self.poller: asyncio.Task[None] | None = None
        self.wake = asyncio.Event()
        self.next_refresh = 0.0
        self.paused = False
        self.failures = 0

    @classmethod
    @override
    def id(cls) -> str:
        return "mirror-example-extension"

    @staticmethod
    @override
    def initial_state() -> ViewState:
        return ViewState()

    @override
    def render_region(self) -> extension.RenderRegion:
        return extension.RenderRegion.WIDGET_BELOW_PROMPT

    @override
    def slash_commands(self) -> Sequence[commands.CommandDef]:
        return (
            commands.CommandDef(
                "linear",
                "Linear dashboard (F8); project [search] / refresh / hide / show",
                slash_command,
                available_when_busy=True,
            ),
        )

    @override
    def keybindings(self) -> Sequence[keymap.Binding]:
        return (
            keymap.Binding(
                ("f8",), actions.RoutedExtensionAction(self.id(), Command())
            ),
        )

    @override
    async def on_config(self, context: extension.Context) -> None:
        configured = context.config.get("project")
        try:
            project = Project.parse(configured) if configured is not None else None
        except ValueError:
            self.paused = True
            self.snapshot = dataclasses.replace(
                self.snapshot, error="Invalid saved project · use /linear project"
            )
            self.dispatch(context, Updated(self.snapshot))
            return
        self.apply_project(context, project)

    def apply_project(
        self, context: extension.Context, project: Project | None
    ) -> None:
        if project == self.snapshot.project:
            return
        if self.dashboard is not None:
            self.dashboard.close()
        self.snapshot = Snapshot(project=project)
        self.paused = False
        self.failures = 0
        self.next_refresh = 0
        self.dispatch(self.foreground or context, Updated(self.snapshot))
        self.wake.set()

    async def choose_project(
        self, context: extension.Context, project: Project
    ) -> None:
        """Remember just the selected project, preserving consent and other settings."""
        root = (
            context.extensions.root
            if context.extensions is not None
            else config.default_path().parent / "extensions"
        )
        await asyncio.to_thread(
            save_project, root, pathlib.Path(CONFIG_KEY).parent, project
        )
        self.apply_project(context, project)

    async def stop_projects(self) -> None:
        picker, self.project_picker = self.project_picker, None
        if picker is not None:
            await picker.close()
            self.wake.set()

    async def show_projects(self, context: extension.Context, query: str = "") -> None:
        await self.stop_projects()
        current = self.snapshot.project
        picker = ProjectPicker(context, query, current)
        self.project_picker = picker
        if context.get_state().approval.request is not None:
            picker.suspended = "", picker.current
        else:
            picker.open()
        picker.task = asyncio.create_task(picker.load())

    @override
    async def on_load(self, context: None) -> None:
        del context
        self.poller = asyncio.create_task(self.watch())

    @override
    async def on_foreground(self, context: extension.Context) -> None:
        self.foreground = context
        self.dispatch(context, Updated(self.snapshot))
        self.wake.set()

    @override
    async def on_background(self, context: extension.Context) -> None:
        if (
            self.foreground is not None
            and self.foreground.workspace == context.workspace
        ):
            self.foreground = None
            self.wake.set()
        if (
            self.project_picker is not None
            and self.project_picker.context.workspace == context.workspace
        ):
            await self.stop_projects()

    async def watch(self) -> None:
        """Use one quiet timer, not one poller per workspace or render."""
        while True:
            await self.wake.wait()
            self.wake.clear()
            if (
                self.foreground is None
                or self.opening
                or self.paused
                or self.snapshot.project is None
                or self.project_picker is not None
            ):
                continue
            delay = self.next_refresh - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(self.wake.wait(), delay)
                except TimeoutError:
                    self.wake.set()
                continue
            await self.refresh(self.foreground)
            self.wake.set()

    @override
    def reduce(
        self, global_state: state.State, action: actions.Action
    ) -> tuple[ViewState, tuple[extension.PrivateEffect[Command | PickerChanged], ...]]:
        current = self.require_state(global_state)
        if isinstance(
            action,
            (
                actions.ApprovalRequested,
                actions.ClearApproval,
                actions.DecideApproval,
                actions.OpenSelector,
                actions.RequestSecret,
            ),
        ):
            return current, (self.effect(PickerChanged()),)
        request = global_state.selector.selector
        if (
            request is not None
            and request.extension_id == self.id()
            and isinstance(
                action,
                (
                    actions.SelectorSearchChanged,
                    actions.ClearSelector,
                    actions.CloseSelector,
                    actions.SelectorChoiceSelected,
                    actions.SessionOpened,
                ),
            )
        ):
            return current, (self.effect(PickerChanged(request.id)),)
        if isinstance(action, actions.RoutedExtensionAction):
            if isinstance(action.action, Updated):
                return dataclasses.replace(current, snapshot=action.action.snapshot), ()
            if (
                isinstance(action.action, Command)
                and not global_state.lifecycle.quitting
            ):
                if action.action.mode in {"hide", "show"}:
                    return dataclasses.replace(
                        current, visible=action.action.mode == "show"
                    ), ()
                return current, (self.effect(action.action),)
        return current, ()

    @override
    def render(
        self, global_state: state.State, extension_state: ViewState
    ) -> containers.AnyContainer | None:
        del global_state
        if not extension_state.visible:
            return None
        return containers.Window(Pulse(extension_state.snapshot), height=1)

    async def refresh(
        self, context: extension.Context, *, discover: bool = False
    ) -> None:
        """Discard stale-project responses after a configuration change."""
        project = self.snapshot.project
        if project is None or self.lock.locked():
            return
        async with self.lock:
            self.paused = False
            self.next_refresh = time.monotonic() + 120
            self.snapshot = dataclasses.replace(self.snapshot, busy=True)
            self.dispatch(context, Updated(self.snapshot))
            try:
                async with asyncio.timeout(120):
                    workflows = (
                        await fetch_workflows(context.call_tool, project)
                        if discover or self.snapshot.fetched_at is None
                        else self.snapshot.workflows
                    )
                    issues = await fetch_issues(context.call_tool, project)
                if self.snapshot.project == project:
                    workflows = tuple(
                        sorted(
                            set(workflows) | {issue.workflow for issue in issues},
                            key=workflow_order,
                        )
                    )
                    self.snapshot = Snapshot(
                        issues,
                        datetime.datetime.now(datetime.UTC),
                        project=project,
                        workflows=workflows,
                    )
            except SyncError as error:
                # Do not keep prompting after an MCP rejection or configuration failure.
                if self.snapshot.project == project:
                    self.paused = True
                    self.snapshot = dataclasses.replace(self.snapshot, error=str(error))
            except (OSError, RuntimeError, ValueError, TimeoutError):
                if self.snapshot.project == project:
                    self.snapshot = dataclasses.replace(
                        self.snapshot,
                        error="Sync failed · check /connections, then /linear refresh",
                    )
            finally:
                if self.snapshot.project == project:
                    self.snapshot = dataclasses.replace(self.snapshot, busy=False)
                    self.failures = (
                        min(self.failures + 1, 6) if self.snapshot.error else 0
                    )
                    retry_delay = min(120, 5 * 2 ** (self.failures - 1))
                    self.next_refresh = time.monotonic() + (
                        retry_delay if self.failures else 120
                    )
                self.dispatch(self.foreground or context, Updated(self.snapshot))

    @override
    async def handle_effect(
        self, context: extension.Context, effect: Command | PickerChanged
    ) -> None:
        if isinstance(effect, PickerChanged):
            picker = self.project_picker
            if (
                picker is None
                or picker.context.workspace != context.workspace
                or effect.selector_id not in {"", picker.id}
            ):
                return
            current = context.get_state()
            if current.secret.active or (
                current.selector.active
                and not picker.active()
                and not (current.selector.id or "").startswith("approval:")
            ):
                await self.stop_projects()
            elif current.approval.request is not None and picker.active():
                picker.yield_to_approval()
            elif picker.suspended is not None and current.approval.request is not None:
                return
            elif picker.suspended is not None and not current.selector.active:
                await picker.resume()
            elif picker.active():
                await picker.publish()
            else:
                await self.stop_projects()
            return
        if self.opening:
            return
        self.opening = True
        try:
            if effect.mode == "project" or (
                self.snapshot.project is None and effect.project is None
            ):
                await self.show_projects(context, effect.query)
                return
            await self.stop_projects()
            if self.lock.locked():
                async with self.lock:
                    pass  # Let pending MCP approvals finish in the parent UI.
            if effect.project is not None:
                await self.choose_project(context, effect.project)
            if (
                effect.mode == "refresh"
                or self.snapshot.fetched_at is None
                or self.snapshot.error
            ):
                await self.refresh(context, discover=effect.mode == "refresh")
            if effect.mode == "refresh":
                return
            self.dashboard = Dashboard(self.snapshot)
            while True:
                self.dashboard.update(self.snapshot)
                result = await self.dashboard.run()
                if result == "project":
                    await self.show_projects(context)
                    break
                if result != "refresh":
                    break
                # Keep the parent UI visible while the normal MCP approval flow runs.
                await self.refresh(context, discover=True)
        except SyncError as error:
            context.dispatch(actions.ShowText(str(error), style="yellow"))
        except (OSError, RuntimeError, ValueError, TimeoutError):
            context.dispatch(
                actions.ShowText(
                    "Could not load or save the project. "
                    "Check /connections, then retry /linear project.",
                    style="yellow",
                )
            )
        finally:
            self.dashboard = None
            self.opening = False
            self.wake.set()

    @override
    async def on_unload(self, context: None) -> None:
        del context
        self.foreground = None
        await self.stop_projects()
        if self.dashboard is not None:
            self.dashboard.close()
        if self.poller is not None:
            self.poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.poller
            self.poller = None
        self.foreground = None


_STYLE = styles.Style.from_dict(
    {
        "menu-border": "fg:#3f8f68 bold",
        "menu-item": "fg:#87a995",
        "menu-selected": "fg:#ffffff bg:#3f8f68 bold",
        "menu-detail": "fg:#74827a",
        "menu-detail-selected": "fg:#d3e3da nobold",
        "menu-footer": "fg:#74827a",
        "linear-error": "fg:ansiyellow",
    }
)


class IssueList(controls.UIControl):
    def __init__(self, dashboard: Dashboard) -> None:
        self.dashboard = dashboard

    @override
    def create_content(self, width: int, height: int) -> controls.UIContent:
        del height
        rows = self.dashboard.rows()
        selected = self.dashboard.selected_issue()
        lines: list[Fragments] = []
        cursor = 0
        previous: Workflow | None = None
        for issue in rows:
            if previous != issue.workflow:
                count = sum(row.workflow == issue.workflow for row in rows)
                lines.append(
                    [
                        (
                            "class:menu-border",
                            clip(
                                f"  {issue.workflow.glyph} "
                                f"{issue.status.upper()}  {count}",
                                width,
                            ),
                        )
                    ]
                )
                previous = issue.workflow
            chosen = issue == selected
            focused = chosen and not self.dashboard.searching
            style = "class:menu-selected" if focused else "class:menu-item"
            if chosen:
                cursor = len(lines)
            title = clip(f" {'▸' if chosen else ' '} {issue.title}", width)
            lines.append([(style, title + " " * (width - utils.get_cwidth(title)))])
            priority = ("", " ! Urgent", " ↑ High", " · Medium", " ↓ Low")[
                issue.priority
            ]
            metadata = clip(
                f"    {issue.identifier}  ·  "
                f"{issue.assignee or 'Unassigned'}{priority}",
                width,
            )
            detail_style = (
                "class:menu-selected class:menu-detail-selected"
                if focused
                else "class:menu-detail"
            )
            lines.append(
                [(detail_style, metadata + " " * (width - utils.get_cwidth(metadata)))]
            )
        if not lines:
            lines = [
                [
                    (
                        "class:menu-detail",
                        "  No matching issues. Clear search or change status.",
                    )
                ]
            ]
        return controls.UIContent(
            get_line=lambda index: lines[index],
            line_count=len(lines),
            cursor_position=data_structures.Point(0, cursor),
            show_cursor=False,
        )


class Dashboard:
    """A temporary alternate-screen browser; never changes issues or conversations."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.selected: str | None = None
        self.status_index = 0
        self.unassigned = False
        self.searching = False
        self.notice = ""
        self.app: application.Application[str] | None = None
        self.search = widgets.TextArea(
            height=1,
            multiline=False,
            prompt="  Search › ",
            read_only=filters.Condition(lambda: not self.searching),
        )
        self.search.window.always_hide_cursor = filters.Condition(
            lambda: not self.searching
        )
        self.search.window.style = lambda: (
            "class:menu-selected" if self.searching else "class:menu-detail"
        )
        self.table = containers.Window(
            IssueList(self),
            scroll_offsets=containers.ScrollOffsets(top=1, bottom=2),
            width=lambda: (
                application.get_app().output.get_size().columns * 3 // 5
                if self.split()
                else application.get_app().output.get_size().columns
            ),
        )

    def split(self) -> bool:
        size = application.get_app().output.get_size()
        return size.columns >= 105 and size.rows >= 24

    def rows(self) -> list[Issue]:
        query = self.search.text.casefold().split()
        return [
            issue
            for issue in self.snapshot.issues
            if (
                not self.status_index
                or issue.workflow == self.snapshot.workflows[self.status_index - 1]
            )
            and (not self.unassigned or not issue.assignee)
            and all(
                term
                in " ".join(
                    (issue.identifier, issue.title, issue.assignee, *issue.labels)
                ).casefold()
                for term in query
            )
        ]

    def selected_issue(self) -> Issue | None:
        rows = self.rows()
        issue = next(
            (issue for issue in rows if issue.identifier == self.selected),
            rows[0] if rows else None,
        )
        self.selected = issue.identifier if issue else None
        return issue

    def move(self, delta: int) -> None:
        rows = self.rows()
        selected = self.selected_issue()
        if selected is not None:
            self.selected = rows[
                max(0, min(len(rows) - 1, rows.index(selected) + delta))
            ].identifier

    def detail(self) -> Fragments:
        issue = self.selected_issue()
        if issue is None:
            return [("class:menu-detail", "\n  Select an issue to see its details.")]
        priority = ("No priority", "Urgent", "High", "Medium", "Low")[issue.priority]
        fields = (
            ("Status", issue.status),
            ("Assignee", issue.assignee or "Unassigned"),
            ("Priority", priority),
            ("Labels", " · ".join(issue.labels) or "—"),
            ("Updated", issue.updated or "Unknown"),
        )
        result: Fragments = [
            ("class:menu-border", f"\n  {issue.identifier}\n\n"),
            ("bold", f"  {issue.title}\n\n"),
        ]
        for label, value in fields:
            result.extend([("class:menu-detail", f"  {label:10}"), ("", value + "\n")])
        result.extend(
            [
                ("class:menu-border", "\n  Enter"),
                ("class:menu-detail", "  Open in Linear\n\n"),
                ("class:menu-detail", f"  {issue.url}"),
            ]
        )
        return result

    def filter_tabs(self) -> Fragments:
        width = application.get_app().output.get_size().columns
        groups = [
            ("All", len(self.snapshot.issues)),
            *[
                (workflow.name, count)
                for workflow, count in zip(
                    self.snapshot.workflows, self.snapshot.counts(), strict=True
                )
            ],
        ]
        tabs: list[str] = []
        for index, (label, count) in enumerate(groups):
            value = str(count) if self.snapshot.fetched_at else "—"
            label = clip(label, min(18, max(1, width - 24)))
            tabs.append(
                f"[{label} {value}] "
                if index == self.status_index
                else f" {label} {value}  "
            )
        start = 0
        while (
            start < self.status_index
            and sum(
                utils.get_cwidth(tab) for tab in tabs[start : self.status_index + 1]
            )
            > width - 16
        ):
            start += 1
        result: Fragments = [
            ("class:menu-detail", "  Status  " + ("‹ " if start else ""))
        ]
        remaining = (
            width - utils.get_cwidth(formatted_text.fragment_list_to_text(result)) - 4
        )
        for index in range(start, len(tabs)):
            text = tabs[index]
            if utils.get_cwidth(text) > remaining:
                result.append(("class:menu-detail", "›"))
                break
            result.append(
                (
                    "class:menu-border bold underline"
                    if index == self.status_index
                    else "class:menu-detail",
                    text,
                )
            )
            remaining -= utils.get_cwidth(text)
        return result

    def heading(self) -> Fragments:
        status = (
            self.snapshot.workflows[self.status_index - 1].name
            if self.status_index
            else "All active"
        )
        owner = " · unassigned" if self.unassigned else ""
        name = self.snapshot.project.name.upper() if self.snapshot.project else "LINEAR"
        rows, selected = self.rows(), self.selected_issue()
        position = (
            f" · {rows.index(selected) + 1}/{len(rows)} selected"
            if selected is not None
            else ""
        )
        return [
            ("class:menu-border", "  Linear  /  "),
            ("bold", clip(name, 28)),
            ("class:menu-detail", f"  · {status}{owner}{position}"),
        ]

    def update(self, snapshot: Snapshot) -> None:
        selected = (
            self.snapshot.workflows[self.status_index - 1]
            if self.status_index
            else None
        )
        self.snapshot = snapshot
        self.status_index = (
            snapshot.workflows.index(selected) + 1
            if selected in snapshot.workflows
            else 0
        )

    def body(self) -> containers.AnyContainer:
        detail = containers.Window(
            controls.FormattedTextControl(self.detail), wrap_lines=True
        )
        split = containers.VSplit(
            [
                self.table,
                containers.Window(width=1, char="│", style="class:menu-detail"),
                detail,
            ],
            padding=1,
        )
        return containers.HSplit(
            [
                containers.Window(height=1),
                containers.Window(
                    controls.FormattedTextControl(self.heading), height=1
                ),
                containers.Window(height=1),
                containers.Window(
                    controls.FormattedTextControl(self.filter_tabs), height=1
                ),
                self.search,
                containers.Window(height=1),
                containers.DynamicContainer(
                    lambda: split if self.split() else self.table
                ),
                containers.ConditionalContainer(
                    containers.Window(
                        controls.FormattedTextControl(
                            lambda: [
                                (
                                    "class:linear-error",
                                    "  " + (self.notice or self.snapshot.problem()),
                                )
                            ]
                        ),
                        height=1,
                    ),
                    filters.Condition(lambda: bool(self.notice or self.snapshot.error)),
                ),
                containers.Window(
                    controls.FormattedTextControl(self.footer),
                    height=lambda: (
                        2
                        if application.get_app().output.get_size().columns < 110
                        else 1
                    ),
                    style="class:menu-footer",
                ),
            ],
        )

    def footer(self) -> str:
        if self.searching:
            return "  Enter/Esc finish search  Ctrl-U clear  F8 close"
        navigation = "  ↑↓/jk move  / search  Tab status  Enter open"
        commands = "  F8/Esc close  p project  r refresh  u unassigned"
        separator = (
            "\n" if application.get_app().output.get_size().columns < 110 else "  "
        )
        return navigation + separator + commands

    def bindings(self) -> key_bindings.KeyBindingsBase:
        bindings = key_bindings.KeyBindings()
        browsing = filters.Condition(lambda: not self.searching)

        @bindings.add("escape", eager=True)
        def escape(event: key_processor.KeyPressEvent) -> None:
            if self.searching:
                self.searching = False
            else:
                event.app.exit(result="close")

        @bindings.add("f8")
        @bindings.add("c-c")
        @bindings.add("q", filter=browsing)
        def close(event: key_processor.KeyPressEvent) -> None:
            event.app.exit(result="close")

        @bindings.add("j", filter=browsing)
        @bindings.add("down")
        def down(event: key_processor.KeyPressEvent) -> None:
            del event
            self.move(1)

        @bindings.add("k", filter=browsing)
        @bindings.add("up")
        def up(event: key_processor.KeyPressEvent) -> None:
            del event
            self.move(-1)

        @bindings.add("pagedown")
        @bindings.add("pageup")
        def page(event: key_processor.KeyPressEvent) -> None:
            step = (
                max(1, (self.table.render_info.window_height - 2) // 2)
                if self.table.render_info
                else 5
            )
            self.move(-step if event.key_sequence[0].key == "pageup" else step)

        @bindings.add("home", filter=browsing)
        @bindings.add("end", filter=browsing)
        def endpoint(event: key_processor.KeyPressEvent) -> None:
            self.move(
                -len(self.rows())
                if event.key_sequence[0].key == "home"
                else len(self.rows())
            )

        @bindings.add("/", filter=browsing)
        def search(event: key_processor.KeyPressEvent) -> None:
            del event
            self.searching = True

        @bindings.add("tab", filter=browsing)
        @bindings.add("s-tab", filter=browsing)
        def cycle(event: key_processor.KeyPressEvent) -> None:
            delta = -1 if event.key_sequence[0].key == "s-tab" else 1
            self.status_index = (self.status_index + delta) % (
                len(self.snapshot.workflows) + 1
            )

        @bindings.add("u", filter=browsing)
        def unassigned(event: key_processor.KeyPressEvent) -> None:
            del event
            self.unassigned = not self.unassigned

        @bindings.add("p", filter=browsing)
        def project(event: key_processor.KeyPressEvent) -> None:
            event.app.exit(result="project")

        @bindings.add("r", filter=browsing)
        def refresh(event: key_processor.KeyPressEvent) -> None:
            event.app.exit(result="refresh")

        @bindings.add("enter")
        def enter(event: key_processor.KeyPressEvent) -> None:
            if self.searching:
                self.searching = False
            else:
                event.app.create_background_task(self.open_issue())

        @bindings.add("c-u")
        def clear(event: key_processor.KeyPressEvent) -> None:
            del event
            self.search.text = ""

        return key_bindings.merge_key_bindings(
            [workspace_application.terminal_bindings(), bindings]
        )

    async def open_issue(self) -> None:
        issue = self.selected_issue()
        if issue is not None:
            try:
                opened = bool(issue.url) and await asyncio.to_thread(
                    webbrowser.open, issue.url
                )
            except (OSError, webbrowser.Error):
                opened = False
            self.notice = (
                "Opened in your browser"
                if opened
                else "Could not open browser; use the issue URL in the detail pane"
            )
            if self.app is not None:
                self.app.invalidate()

    def create_app(
        self, parent: application.Application[object]
    ) -> application.Application[str]:
        """Reuse the host terminal without replacing its style or keybindings."""
        app = application.Application[str](
            layout=layout.Layout(self.body(), focused_element=self.search),
            key_bindings=self.bindings(),
            style=styles.merge_styles([_STYLE, parent.style])
            if parent.style
            else _STYLE,
            style_transformation=parent.style_transformation,
            color_depth=parent.color_depth,
            full_screen=True,
            input=parent.input,
            output=parent.output,
        )
        app.ttimeoutlen = parent.ttimeoutlen
        app.timeoutlen = parent.timeoutlen
        return app

    async def run(self) -> str:
        self.app = self.create_app(application.get_app())
        try:
            async with application.in_terminal():
                return await self.app.run_async(set_exception_handler=False)
        finally:
            self.app = None

    def close(self) -> None:
        if self.app is not None and self.app.is_running and not self.app.is_done:
            self.app.exit(result="close")
