"""INK & EVIDENCE — Multi-Agent Book Studio (Streamlit UI).

Run with:  streamlit run app.py

The workflow runs synchronously inside the script. Stage messages are written
by the workflow's progress callback at the moment each stage begins, so the UI
never shows progress that has not happened.
"""

from __future__ import annotations


import streamlit as st
from pydantic import ValidationError

from config import ConfigurationError, Settings, load_settings
from models.book_models import BookBrief, ChapterResult, ChapterStatus, ClaimVerdict, Severity
from models.research_models import SourceRegistry
from utils.logging_config import configure_logging
from workflow.book_workflow import BookRunResult, ProgressEvent, create_book_workflow
from workflow.run_artifacts import list_saved_runs

APP_TITLE = "INK & EVIDENCE — Multi-Agent Book Studio"
DEFAULT_TITLE = "Pay Me on UPI: How Digital Payments Changed Small Business in India"
DEFAULT_AUDIENCE = "First-time small-business owners in India"
DEFAULT_TONE = "friendly, clear, encouraging and practical"
DEFAULT_DOMAINS = "npci.org.in, rbi.org.in, pib.gov.in, gov.in"
LIVE_MODE = "Live research (Groq + web sources)"
DEMO_MODE = "Offline demo (mock provider + fictional fixtures)"

STATUS_LABELS = {
    ChapterStatus.PASSED: " Passed",
    ChapterStatus.PASSED_WITH_WARNINGS: " Passed with warnings",
    ChapterStatus.FAILED: "Failed",
    ChapterStatus.IN_PROGRESS: "… In progress",
    ChapterStatus.PENDING: "Pending",
}
VERDICT_LABELS = {
    ClaimVerdict.SUPPORTED: " Supported",
    ClaimVerdict.UNCERTAIN: " Uncertain",
    ClaimVerdict.UNSUPPORTED: "Unsupported",
    ClaimVerdict.CONTRADICTED: "Contradicted",
}

PAGE_CSS = """
<style>
.block-container { max-width: 1180px; padding-top: 2.2rem; }
h1 { font-size: 1.75rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }
[class*="st-key-chapter_preview"] {
    font-family: Georgia, "Iowan Old Style", "Source Serif Pro", serif;
    max-width: 70ch; line-height: 1.75; font-size: 1.05rem;
}
[class*="st-key-chapter_preview"] h2 { font-family: inherit; font-size: 1.45rem; }
[class*="st-key-chapter_preview"] h3 { font-family: inherit; font-size: 1.1rem; }
</style>
"""


def initialize_session_state() -> None:
    defaults = {"run_active": False, "pending_run": None, "last_result": None,
                "last_events": [], "last_error": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_app_settings() -> Settings | None:
    try:
        return load_settings()
    except ValidationError as error:
        st.error(f"Your .env or environment contains an invalid setting:\n\n{error}")
        return None


def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.markdown("### INK & EVIDENCE")
        st.caption(
            "Five agents plan, research, write, edit and fact-check a short book. "
            "Every citation points to a retrieved source."
        )
        st.divider()
        st.markdown("**Language model**")
        groq_ready = settings.groq_api_key is not None and settings.groq_model is not None
        if groq_ready:
            st.success(f"Groq ready · {settings.groq_model}", icon="🔑")
        else:
            st.warning("Groq not configured", icon="🔑")
            st.markdown(
                "To run live research, add `GROQ_API_KEY` and `GROQ_MODEL` to a `.env` file "
                "(copy `.env.example`), then restart the app. The offline demo needs no key."
            )
        st.caption(f"API key configured: {'yes' if settings.groq_api_key else 'no'}")
        st.divider()
        render_saved_runs(settings)
        st.divider()
        st.markdown("**Run settings**")
        st.caption(
            f"Revision limit: {settings.max_revision_rounds} per chapter  \n"
            f"Sources per chapter: up to {settings.max_sources_per_chapter}  \n"
            f"Outputs folder: `{settings.outputs_dir.name}/`"
        )


def render_saved_runs(settings: Settings) -> None:
    """Inspect earlier runs (and their failure diagnostics) without rerunning anything."""
    runs = list_saved_runs(settings.outputs_dir)
    st.markdown("**Saved runs**")
    if not runs:
        st.caption("No saved runs yet.")
        return
    labels = {r["run_id"]: f"{r['run_id'][4:19]} · {r['status']}" for r in runs}
    chosen = st.selectbox("Open a saved run", list(labels), format_func=labels.get, key="saved_run")
    run = next(r for r in runs if r["run_id"] == chosen)
    kind = "demo fixtures" if run.get("evidence_basis") == "test_fixtures" else "live research"
    st.caption(f"{run['status']} · {run.get('provider')} · {kind} · chapters passed: {run.get('chapters_passed')}")
    for filename in ("book.md", "failure_report.md", "run_report.json", "sources.json", "run.log"):
        path = run["path"] / filename
        if path.exists():
            st.download_button(
                f"Download {filename}", path.read_bytes(), file_name=f"{chosen}_{filename}",
                key=f"dl_{chosen}_{filename}",
            )


def render_brief_form(settings: Settings) -> None:
    groq_ready = settings.groq_api_key is not None and settings.groq_model is not None
    running = st.session_state.run_active
    with st.form("book_brief"):
        title = st.text_input("Book title", DEFAULT_TITLE, disabled=running)
        audience = st.text_input("Target audience", DEFAULT_AUDIENCE, disabled=running)
        col_chapters, col_min, col_max = st.columns(3)
        chapters = col_chapters.number_input("Chapters", 1, 10, 3, disabled=running)
        min_words = col_min.number_input("Minimum words per chapter", 100, 5000, 600, 50, disabled=running)
        max_words = col_max.number_input("Maximum words per chapter", 100, 5000, 900, 50, disabled=running)
        tone = st.text_input("Tone", DEFAULT_TONE, disabled=running)
        domains = st.text_input(
            "Preferred source domains",
            DEFAULT_DOMAINS,
            help="Results from these domains are fetched first. Comma-separated.",
            disabled=running,
        )
        mode = st.radio(
            "Run mode",
            [LIVE_MODE, DEMO_MODE],
            index=0 if groq_ready and not settings.is_mock_mode else 1,
            horizontal=True,
            disabled=running,
            help="The offline demo exercises every agent with fictional fixtures. It is not research.",
        )
        submitted = st.form_submit_button(
            "Running…" if running else "Generate book", type="primary", disabled=running
        )

    if submitted and not running:
        try:
            brief = BookBrief(
                title=title, audience=audience, chapter_count=int(chapters),
                min_words_per_chapter=int(min_words), max_words_per_chapter=int(max_words),
                tone=tone, preferred_source_domains=domains.split(","),
            )
        except ValidationError as error:
            st.error(_friendly_validation_message(error))
            return
        st.session_state.pending_run = {"brief": brief.model_dump(), "mock": mode == DEMO_MODE}
        st.session_state.run_active = True
        st.rerun()


def _friendly_validation_message(error: ValidationError) -> str:
    lines = [f"- {'.'.join(map(str, e['loc'])) or 'Brief'}: {e['msg']}" for e in error.errors()]
    return "Please fix the brief:\n\n" + "\n".join(lines)


def execute_pending_run(settings: Settings) -> None:
    """Run the queued workflow once. The queue is cleared before starting, so a
    rerun can never start the same request twice."""
    pending = st.session_state.pending_run
    st.session_state.pending_run = None
    if pending is None:
        st.session_state.run_active = False
        return

    run_settings = settings.model_copy(
        update={"llm_provider": "mock" if pending["mock"] else "groq"}
    )
    events: list[str] = []
    try:
        run_settings.require_live_provider_settings()
        with st.status("Starting run…", expanded=True) as status:

            def show_progress(event: ProgressEvent) -> None:
                line = f"`{event.elapsed_seconds:6.1f}s` {event.message}"
                events.append(line)
                status.update(label=event.message)
                st.markdown(line)

            workflow = create_book_workflow(run_settings, progress_callback=show_progress)
            result = workflow.generate_book(BookBrief.model_validate(pending["brief"]))
            status.update(
                label=f"Run finished: {result.publication_status}",
                state="complete" if result.publication_status == "published" else "error",
                expanded=False,
            )
        st.session_state.last_result = result
        st.session_state.last_events = events
        st.session_state.last_error = None
    except ConfigurationError as error:
        st.session_state.last_error = f"Cannot start a live run. {error}"
    except Exception as error:  # noqa: BLE001 - shown to the user; details are in run.log
        st.session_state.last_error = f"Unexpected {type(error).__name__}: {error}"
    finally:
        st.session_state.run_active = False
    st.rerun()


def render_run_banner(result: BookRunResult) -> None:
    is_demo = result.state.evidence_basis.value == "test_fixtures"
    if is_demo:
        st.warning(
            "DEMO OUTPUT, NOT REAL RESEARCH: this run used the offline mock provider and "
            "fictional fixtures. It demonstrates the workflow only; no AI model or real "
            "source was used.",
            icon="🧪",
        )
    else:
        real_sources = sum(1 for s in result.registry.sources.values() if s.is_real_world_evidence)
        st.info(
            f"Live run: {real_sources} real web source(s) retrieved; written by "
            f"{result.state.provider_name}/{result.state.model_name}. "
            "Automated checks do not prove every statement; review flagged claims.",
            icon="🌐",
        )
    if result.publication_status == "published" and is_demo:
        st.caption("Demo run completed its workflow checks; the content is fictional.")
    elif result.publication_status == "published":
        reviews = sum(len(c.fact_check.claims_needing_review) for c in result.state.chapters if c.fact_check)
        if reviews:
            st.warning(f"Book published with {reviews} claim(s) flagged for human review.", icon="⚠️")
        else:
            st.success("All chapters passed the mandatory checks. Book published.", icon="✅")
    elif result.publication_status == "blocked":
        st.error(
            "Publication blocked: at least one chapter still has critical issues after the "
            "revision limit. See the Research & quality tab and the failure report.",
            icon="⛔",
        )
    else:
        st.error("The run stopped before completion. " + " ".join(result.state.errors), icon="⛔")


def render_workflow_tab(result: BookRunResult | None) -> None:
    if result is None:
        st.markdown("No run yet. Fill in the brief and choose **Generate book**.")
        return
    state = result.state
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Status", result.publication_status.capitalize())
    col_b.metric("Chapters passed", f"{result.report['chapters_passed']} / {len(state.chapters)}")
    col_c.metric("Sources registered", len(result.registry.sources))
    col_d.metric("Elapsed", f"{state.elapsed_seconds:.1f} s")
    st.caption(f"Run `{state.run_id}` · provider `{state.provider_name}` · model `{state.model_name}`")

    rows = [
        {
            "Chapter": c.plan.chapter_number,
            "Title": c.draft.title if c.draft else c.plan.title,
            "Status": STATUS_LABELS[c.status],
            "Words": c.word_count,
            "Revisions": c.revision_count,
            "Sources cited": ", ".join(c.source_ids),
            "Fact-check": "performed" if c.fact_check else "not run",
            "Failure reason": c.failure_reason or "",
        }
        for c in state.chapters
    ]
    if rows:
        st.dataframe(rows, hide_index=True)
    for error in state.errors:
        st.error(error)
    with st.expander("Stage log for this run", expanded=False):
        st.markdown("\n\n".join(st.session_state.last_events) or "No events recorded.")


def render_research_tab(result: BookRunResult | None) -> None:
    if result is None:
        st.markdown("Sources, evidence and fact-check verdicts appear here after a run.")
        return
    registry = result.registry
    st.markdown("#### Source registry")
    source_rows = [
        {
            "ID": s.source_id,
            "Title": s.title,
            "Publisher": s.publisher,
            "URL": s.url,
            "Status": s.retrieval_status.value,
            "Excerpts": len(s.excerpts),
            "Published": s.publication_date.isoformat() if s.publication_date else "",
            "Note": s.failure_reason or "",
        }
        for s in sorted(registry.sources.values(), key=lambda s: int(s.source_id[1:]))
    ]
    if source_rows:
        st.dataframe(
            source_rows, hide_index=True,
            column_config={"URL": st.column_config.LinkColumn("URL")},
        )
    else:
        st.markdown("No sources were registered.")

    for chapter in result.state.chapters:
        number = chapter.plan.chapter_number
        label = f"Chapter {number}: {chapter.plan.title} — {STATUS_LABELS[chapter.status]}"
        with st.expander(label, expanded=chapter.status == ChapterStatus.FAILED):
            evidence = result.research.get(number)
            if evidence:
                st.caption(
                    f"{len(evidence.queries_run)} searches · {evidence.urls_considered} URLs tried · "
                    f"usable: {', '.join(evidence.usable_source_ids) or 'none'} · "
                    f"unusable: {', '.join(evidence.unusable_source_ids) or 'none'}"
                )
                for search_error in evidence.search_errors:
                    st.warning(f"Search problem: {search_error}")
            render_issue_list(chapter)
            render_claims(chapter, registry)


def render_issue_list(chapter: ChapterResult) -> None:
    if not chapter.validation_issues:
        st.markdown("No validation issues in the final round.")
        return
    for issue in chapter.validation_issues:
        prefix = "Heuristic check: " if issue.is_heuristic else ""
        message = f"{prefix}`{issue.code}` — {issue.message}"
        if issue.severity == Severity.CRITICAL:
            st.error(message)
        elif issue.severity == Severity.WARNING:
            st.warning(message)
        else:
            st.info(message)


def render_claims(chapter: ChapterResult, registry: SourceRegistry) -> None:
    if chapter.fact_check is None:
        if chapter.draft is None:
            st.caption("Fact-check not run: no draft was produced (see the failure reason above).")
        else:
            st.caption("Fact-check not run for the final draft (structural checks failed first).")
        return
    st.markdown("**Claims and evidence**")
    for claim in chapter.fact_check.claims:
        with st.container(border=True):
            st.markdown(f"{VERDICT_LABELS[claim.verdict]} — {claim.claim_text}")
            st.caption(
                f"Cites: {', '.join(claim.cited_source_ids) or 'nothing'} · {claim.explanation}"
            )
            for excerpt_id in claim.evidence_excerpt_ids:
                excerpt = registry.get_excerpt(excerpt_id)
                if excerpt:
                    st.markdown(f"> {excerpt.text}  \n> — *{excerpt_id}*")


def render_book_tab(result: BookRunResult | None) -> None:
    if result is None:
        st.markdown("The finished book and downloads appear here after a run.")
        return
    render_downloads(result)
    if result.book_markdown:
        for chapter in result.state.chapters:
            with st.container(key=f"chapter_preview_{chapter.plan.chapter_number}"):
                st.markdown(chapter.rendered_markdown or "")
            st.divider()
    elif result.failure_report:
        st.markdown(result.failure_report)


def render_downloads(result: BookRunResult) -> None:
    files = result.saved_files
    columns = st.columns(4)
    downloads = [
        ("book", "Download book.md", "text/markdown"),
        ("failure_report", "Download failure_report.md", "text/markdown"),
        ("sources", "Download sources.json", "application/json"),
        ("report", "Download run_report.json", "application/json"),
    ]
    slot = 0
    for key, label, mime in downloads:
        path = files.get(key)
        if path is None or not path.exists():
            continue
        columns[slot].download_button(
            label, data=path.read_bytes(), file_name=path.name, mime=mime,
            on_click="ignore", width="stretch",
        )
        slot += 1
    st.caption(f"Saved in `{result.run_dir}`")


def main() -> None:
    st.set_page_config(page_title="INK & EVIDENCE", page_icon="🖋️", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    initialize_session_state()
    settings = load_app_settings()
    if settings is None:
        return
    configure_logging(settings.log_level)

    render_sidebar(settings)
    st.title(APP_TITLE)
    st.caption("Plan, research, write, edit and fact-check a short book with traceable citations.")

    if st.session_state.run_active:
        render_brief_form(settings)
        execute_pending_run(settings)
        return

    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    result: BookRunResult | None = st.session_state.last_result
    if result is not None:
        render_run_banner(result)

    brief_tab, workflow_tab, research_tab, book_tab = st.tabs(
        ["Brief", "Workflow", "Research & quality", "Book"]
    )
    with brief_tab:
        render_brief_form(settings)
    with workflow_tab:
        render_workflow_tab(result)
    with research_tab:
        render_research_tab(result)
    with book_tab:
        render_book_tab(result)


main()
