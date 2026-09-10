"""Create the current M0-M3A schema."""
from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("storage_path", sa.String(512), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("failure_reason", sa.String(500)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "document_texts",
        sa.Column("document_id", sa.String(36), sa.ForeignKey(
            "documents.id"), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey(
            "documents.id"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_document_chunks_document_id",
                    "document_chunks", ["document_id"])
    op.create_table(
        "document_embeddings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("chunk_id", sa.String(36), sa.ForeignKey(
            "document_chunks.id"), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("vector", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("chunk_id", "model_name",
                            name="uq_embedding_chunk_model"),
    )
    op.create_index("ix_document_embeddings_chunk_id",
                    "document_embeddings", ["chunk_id"])
    _create_assessment_tables()


def _create_assessment_tables() -> None:
    op.create_table(
        "assessment_generation_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_fingerprint", sa.String(128), nullable=False),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id")),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("skill", sa.String(255)),
        sa.Column("question_type", sa.String(20), nullable=False),
        sa.Column("difficulty", sa.String(20)),
        sa.Column("bloom_level", sa.String(20)),
        sa.Column("requested_count", sa.Integer(), nullable=False),
        sa.Column("retrieval_top_k", sa.Integer(), nullable=False),
        sa.Column("retrieval_candidate_k", sa.Integer()),
        sa.Column("embedding_model_name", sa.String(255), nullable=False),
        sa.Column("reranker_model_name", sa.String(255), nullable=False),
        sa.Column("llm_model_name", sa.String(255), nullable=False),
        sa.Column("prompt_version", sa.String(100), nullable=False),
        sa.Column("seed", sa.Integer()),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("failure_reason", sa.String(500)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("request_fingerprint",
                            name="uq_assessment_run_fingerprint"),
    )
    op.create_index("ix_assessment_generation_runs_request_fingerprint",
                    "assessment_generation_runs", ["request_fingerprint"])
    op.create_index("ix_assessment_generation_runs_document_id",
                    "assessment_generation_runs", ["document_id"])
    op.create_table(
        "assessment_questions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("generation_run_id", sa.String(36), sa.ForeignKey(
            "assessment_generation_runs.id"), nullable=False),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id")),
        sa.Column("question_type", sa.String(20), nullable=False),
        sa.Column("stem", sa.Text(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("difficulty", sa.String(20), nullable=False),
        sa.Column("bloom_level", sa.String(20), nullable=False),
        sa.Column("topic", sa.String(255), nullable=False),
        sa.Column("skill", sa.String(255), nullable=False),
        sa.Column("correct_option_key", sa.String(20)),
        sa.Column("content_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_assessment_questions_generation_run_id",
                    "assessment_questions", ["generation_run_id"])
    op.create_index("ix_assessment_questions_document_id",
                    "assessment_questions", ["document_id"])
    op.create_table(
        "assessment_question_options",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question_id", sa.String(36), sa.ForeignKey(
            "assessment_questions.id"), nullable=False),
        sa.Column("option_key", sa.String(20), nullable=False),
        sa.Column("option_text", sa.Text(), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("question_id", "option_key",
                            name="uq_assessment_option_key"),
        sa.UniqueConstraint("question_id", "display_order",
                            name="uq_assessment_option_order"),
    )
    op.create_index("ix_assessment_question_options_question_id",
                    "assessment_question_options", ["question_id"])
    op.create_table(
        "assessment_question_citations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("question_id", sa.String(36), sa.ForeignKey(
            "assessment_questions.id"), nullable=False),
        sa.Column("chunk_id", sa.String(36), sa.ForeignKey(
            "document_chunks.id"), nullable=False),
        sa.Column("citation_order", sa.Integer(), nullable=False),
        sa.Column("retrieval_rank", sa.Integer(), nullable=False),
        sa.Column("reranker_score", sa.Float(), nullable=False),
        sa.Column("hybrid_score", sa.Float(), nullable=False),
        sa.Column("chunk_text_snapshot", sa.Text(), nullable=False),
        sa.Column("page_number_snapshot", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("question_id", "citation_order",
                            name="uq_assessment_citation_order"),
    )
    op.create_index("ix_assessment_question_citations_question_id",
                    "assessment_question_citations", ["question_id"])
    op.create_index("ix_assessment_question_citations_chunk_id",
                    "assessment_question_citations", ["chunk_id"])


def downgrade() -> None:
    op.drop_table("assessment_question_citations")
    op.drop_table("assessment_question_options")
    op.drop_table("assessment_questions")
    op.drop_table("assessment_generation_runs")
    op.drop_index("ix_document_embeddings_chunk_id",
                  table_name="document_embeddings")
    op.drop_table("document_embeddings")
    op.drop_index("ix_document_chunks_document_id",
                  table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_table("document_texts")
    op.drop_table("documents")
