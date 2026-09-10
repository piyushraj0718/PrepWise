"""Add learner answer submissions table for M4B."""
from alembic import op
import sqlalchemy as sa

revision = "0005_learner_answer_submissions"
down_revision = "0004_learner_quiz_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "learner_answer_submissions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("learner_quiz_sessions.id"),
            nullable=False,
        ),
        sa.Column(
            "question_id",
            sa.String(36),
            sa.ForeignKey("assessment_questions.id"),
            nullable=False,
        ),
        sa.Column("learner_id", sa.String(255), nullable=False),
        sa.Column("submitted_option_key", sa.String(20), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "session_id",
            "question_id",
            name="uq_learner_submission_session_question",
        ),
    )
    op.create_index(
        "ix_learner_answer_submissions_session_id",
        "learner_answer_submissions",
        ["session_id"],
    )
    op.create_index(
        "ix_learner_answer_submissions_question_id",
        "learner_answer_submissions",
        ["question_id"],
    )
    op.create_index(
        "ix_learner_answer_submissions_learner_id",
        "learner_answer_submissions",
        ["learner_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_learner_answer_submissions_learner_id",
        table_name="learner_answer_submissions",
    )
    op.drop_index(
        "ix_learner_answer_submissions_question_id",
        table_name="learner_answer_submissions",
    )
    op.drop_index(
        "ix_learner_answer_submissions_session_id",
        table_name="learner_answer_submissions",
    )
    op.drop_table("learner_answer_submissions")
