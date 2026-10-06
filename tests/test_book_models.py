import pytest
from pydantic import ValidationError

from models.book_models import (
    BookBrief,
    BookOutline,
    BookRunState,
    ChapterPlan,
    ChapterResult,
    ChapterStatus,
    ClaimCheck,
    ClaimVerdict,
    EditorReview,
    EvidenceBasis,
    FactCheckReport,
)

UPI_TITLE = "Pay Me on UPI: How Digital Payments Changed Small Business in India"


def make_plan(number: int) -> ChapterPlan:
    return ChapterPlan(
        chapter_number=number,
        title=f"Chapter {number} title",
        purpose="Explain one part of the story clearly.",
        key_points=["first point", "second point"],
        research_questions=["What happened?"],
        search_queries=["upi merchant adoption"],
    )


class TestBookBrief:
    def test_defaults_are_three_chapters_of_600_to_900_words(self):
        brief = BookBrief(title=UPI_TITLE, audience="First-time small-business owners")
        assert brief.chapter_count == 3
        assert (brief.min_words_per_chapter, brief.max_words_per_chapter) == (600, 900)

    def test_title_is_stripped(self):
        assert BookBrief(title="  Pay Me  ", audience="owners").title == "Pay Me"

    def test_blank_title_is_rejected(self):
        with pytest.raises(ValidationError):
            BookBrief(title="     ", audience="owners")

    @pytest.mark.parametrize("count", [0, 11])
    def test_chapter_count_out_of_range_is_rejected(self, count):
        with pytest.raises(ValidationError):
            BookBrief(title=UPI_TITLE, audience="owners", chapter_count=count)

    def test_inverted_word_range_is_rejected(self):
        with pytest.raises(ValidationError, match="min_words_per_chapter"):
            BookBrief(
                title=UPI_TITLE,
                audience="owners",
                min_words_per_chapter=900,
                max_words_per_chapter=600,
            )

    def test_source_domains_are_normalized_and_deduplicated(self):
        brief = BookBrief(
            title=UPI_TITLE,
            audience="owners",
            preferred_source_domains=["WWW.RBI.org.in", "rbi.org.in", " ", "npci.org.in"],
        )
        assert brief.preferred_source_domains == ["rbi.org.in", "npci.org.in"]


class TestBookOutline:
    def test_consecutive_chapters_are_accepted(self):
        outline = BookOutline(
            book_title=UPI_TITLE,
            central_argument="UPI lowered the cost of accepting digital payments.",
            audience="owners",
            chapters=[make_plan(1), make_plan(2), make_plan(3)],
        )
        brief = BookBrief(title=UPI_TITLE, audience="owners")
        assert outline.matches_brief(brief)

    def test_gap_in_chapter_numbers_is_rejected(self):
        with pytest.raises(ValidationError, match="numbered 1..n"):
            BookOutline(
                book_title=UPI_TITLE,
                central_argument="UPI lowered the cost of accepting digital payments.",
                audience="owners",
                chapters=[make_plan(1), make_plan(3)],
            )

    def test_chapter_plan_requires_research_questions(self):
        with pytest.raises(ValidationError):
            ChapterPlan(
                chapter_number=1,
                title="Title",
                purpose="A clear purpose here.",
                key_points=["a", "b"],
                research_questions=[],
                search_queries=["q"],
            )


class TestEditorReview:
    def test_rejection_without_instructions_is_invalid(self):
        with pytest.raises(ValidationError, match="revision_instructions"):
            EditorReview(approved=False, summary="Needs work.")

    def test_approval_without_instructions_is_valid(self):
        assert EditorReview(approved=True, summary="Reads well.").approved


class TestClaimCheck:
    def test_supported_verdict_must_name_evidence(self):
        with pytest.raises(ValidationError, match="excerpt IDs"):
            ClaimCheck(
                claim_text="UPI launched in 2016.",
                cited_source_ids=["S01"],
                verdict=ClaimVerdict.SUPPORTED,
                explanation="The source says so.",
            )

    def test_contradicted_verdict_must_name_evidence(self):
        with pytest.raises(ValidationError):
            ClaimCheck(
                claim_text="UPI launched in 2010.",
                cited_source_ids=["S01"],
                verdict=ClaimVerdict.CONTRADICTED,
                explanation="The source gives a different year.",
            )

    def test_unsupported_verdict_may_have_no_evidence(self):
        claim = ClaimCheck(
            claim_text="Most shops prefer UPI.",
            cited_source_ids=["S02"],
            verdict=ClaimVerdict.UNSUPPORTED,
            explanation="No excerpt mentions shop preferences.",
        )
        assert claim.evidence_excerpt_ids == []

    def test_uncertain_verdict_always_needs_human_review(self):
        claim = ClaimCheck(
            claim_text="Adoption doubled.",
            cited_source_ids=["S01"],
            verdict=ClaimVerdict.UNCERTAIN,
            explanation="The excerpt is ambiguous about the time period.",
            needs_human_review=False,
        )
        assert claim.needs_human_review is True

    @pytest.mark.parametrize("bad_id", ["1", "S1", "source-1", "[S01]"])
    def test_malformed_source_ids_are_rejected(self, bad_id):
        with pytest.raises(ValidationError, match="invalid source ID"):
            ClaimCheck(
                claim_text="A claim.",
                cited_source_ids=[bad_id],
                verdict=ClaimVerdict.UNSUPPORTED,
                explanation="Not found.",
            )

    def test_report_separates_critical_and_review_claims(self):
        report = FactCheckReport(
            claims=[
                ClaimCheck(
                    claim_text="Supported claim.",
                    cited_source_ids=["S01"],
                    verdict=ClaimVerdict.SUPPORTED,
                    evidence_excerpt_ids=["S01-E1"],
                    explanation="Excerpt states it.",
                ),
                ClaimCheck(
                    claim_text="Contradicted claim.",
                    cited_source_ids=["S01"],
                    verdict=ClaimVerdict.CONTRADICTED,
                    evidence_excerpt_ids=["S01-E2"],
                    explanation="Excerpt says otherwise.",
                ),
                ClaimCheck(
                    claim_text="Uncertain claim.",
                    cited_source_ids=["S02"],
                    verdict=ClaimVerdict.UNCERTAIN,
                    explanation="Ambiguous wording.",
                ),
            ]
        )
        assert [c.claim_text for c in report.critical_claims] == ["Contradicted claim."]
        assert [c.claim_text for c in report.claims_needing_review] == ["Uncertain claim."]


class TestBookRunState:
    def make_state(self, statuses: list[ChapterStatus]) -> BookRunState:
        return BookRunState(
            run_id="run-test",
            brief=BookBrief(title=UPI_TITLE, audience="owners"),
            provider_name="mock",
            model_name="mock-deterministic",
            evidence_basis=EvidenceBasis.TEST_FIXTURES,
            chapters=[
                ChapterResult(plan=make_plan(i + 1), status=status)
                for i, status in enumerate(statuses)
            ],
        )

    def test_publishable_when_all_chapters_pass(self):
        state = self.make_state([ChapterStatus.PASSED, ChapterStatus.PASSED_WITH_WARNINGS])
        assert state.is_publishable

    def test_one_failed_chapter_blocks_publication(self):
        state = self.make_state([ChapterStatus.PASSED, ChapterStatus.FAILED])
        assert not state.is_publishable

    def test_run_with_no_chapters_is_not_publishable(self):
        assert not self.make_state([]).is_publishable

    def test_elapsed_time_is_non_negative(self):
        assert self.make_state([]).elapsed_seconds >= 0
