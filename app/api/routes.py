import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.api.dependencies import (
    get_assessment_generation_service,
    get_assessment_repository,
    get_document_service,
    get_embedding_service,
    get_learner_repository,
    get_rag_service,
    get_search_service,
)
from app.schemas.documents import (
    DocumentChunkGenerationResponse,
    DocumentChunksResponse,
    DocumentProcessingResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    EmbeddingGenerationResponse,
    EmbeddingStatusItem,
    EmbeddingStatusResponse,
    AskRequest,
    AskResponse,
    SourceCitationItem,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.schemas.assessment_generation import QuestionGenerationResponse
from app.schemas.assessments import (
    GenerationRunResponse,
    QualityEvaluationListResponse,
    QualityEvaluationResponse,
    QuestionGenerationRequest,
    QuestionListResponse,
    QuestionResponse,
    QuestionListFilter,
)
from app.services.documents import (
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentProcessingStateError,
    DocumentService,
    UnsupportedDocumentError,
    UploadStorageError,
)
from app.services.embeddings import (
    EmbeddingPersistenceError,
    EmbeddingProviderError,
    EmbeddingService,
    InvalidEmbeddingError,
)
from app.services.retrieval import InvalidRetrievalQuery, RetrievalProviderError, RetrievalService
from app.services.reranking import (
    InvalidRerankingQuery,
    RerankerProviderError,
    RerankingService,
)
from app.services.rag import InvalidRAGQuery, RAGError, RAGService
from app.services.assessment_generation import (
    AssessmentGenerationError,
    AssessmentGenerationService,
    AssessmentNoContextError,
    AssessmentPersistenceError,
    AssessmentValidationError,
)
from app.repositories.assessments import AssessmentRepository
from app.repositories.learner import DuplicateSubmissionError, LearnerRepository
from app.ai.llm import LLMConfigurationError, LLMProviderError, LLMResponseError
from app.domain.assessment import (
    detect_weak_areas,
    score_mcq_answer,
    WEAK_TOPIC_MIN_ATTEMPTS,
    WEAK_TOPIC_ACCURACY_THRESHOLD,
)
from app.schemas.learner import (
    AnswerSubmissionRequest,
    AnswerSubmissionResponse,
    AreaPerformanceResponse,
    LearnerPerformanceResponse,
    LearnerQuestionResponse,
    LearnerQuestionOptionResponse,
    LearnerWeakTopicsResponse,
    QuizSessionCreateRequest,
    QuizSessionQuestionsResponse,
    QuizSessionResponse,
    SessionResultResponse,
    WeakAreaResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    try:
        document = await service.upload(file)
    except UnsupportedDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except UploadStorageError as error:
        logger.error("Unable to store uploaded document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be stored",
        ) from error

    return DocumentUploadResponse(
        document_id=document.id,
        status=document.status,
        created_at=document.created_at,
    )


@router.post("/documents/{document_id}/process", response_model=DocumentProcessingResponse)
def process_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentProcessingResponse:
    try:
        document = service.process(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except DocumentProcessingError as error:
        logger.warning("Unable to process document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The document could not be processed",
        ) from error

    return DocumentProcessingResponse(document_id=document.id, status=document.status)


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentStatusResponse:
    try:
        document = service.get(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    return DocumentStatusResponse(
        document_id=document.id,
        status=document.status,
        created_at=document.created_at,
    )


@router.post("/documents/{document_id}/chunks", response_model=DocumentChunkGenerationResponse)
def generate_document_chunks(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentChunkGenerationResponse:
    try:
        chunks = service.generate_chunks(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except DocumentProcessingError as error:
        logger.warning(
            "Unable to generate chunks for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The document chunks could not be generated",
        ) from error

    return DocumentChunkGenerationResponse(document_id=document_id, chunk_count=len(chunks))


@router.get("/documents/{document_id}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentChunksResponse:
    try:
        chunks = service.get_chunks(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return DocumentChunksResponse(
        document_id=document_id,
        chunks=[
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "page_number": chunk.page_number,
                "created_at": chunk.created_at,
            }
            for chunk in chunks
        ],
    )


@router.post("/documents/{document_id}/embeddings", response_model=EmbeddingGenerationResponse)
def generate_document_embeddings(
    document_id: str,
    rebuild: bool = Query(False),
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingGenerationResponse:
    try:
        summary = service.generate(document_id, rebuild=rebuild)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (DocumentProcessingError, InvalidEmbeddingError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except EmbeddingProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except EmbeddingPersistenceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error

    return EmbeddingGenerationResponse(
        document_id=summary.document_id,
        model_name=summary.model_name,
        dimension=summary.dimension,
        embedded_chunk_count=summary.embedded_chunk_count,
    )


@router.get("/documents/{document_id}/embeddings", response_model=EmbeddingStatusResponse)
def get_document_embedding_status(
    document_id: str,
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingStatusResponse:
    try:
        embeddings = service.get_status(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    grouped: dict[str, list] = {}
    for embedding in embeddings:
        grouped.setdefault(embedding.model_name, []).append(embedding)
    return EmbeddingStatusResponse(
        document_id=document_id,
        embeddings=[
            EmbeddingStatusItem(
                model_name=model_name,
                dimension=rows[0].dimension,
                embedded_chunk_count=len(rows),
            )
            for model_name, rows in grouped.items()
        ],
    )


@router.post("/search", response_model=SearchResponse)
def search_documents(
    request: SearchRequest,
    service: RerankingService = Depends(get_search_service),
) -> SearchResponse:
    try:
        results = service.search(
            request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            document_id=str(
                request.document_id) if request.document_id else None,
        )
    except RetrievalProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InvalidRetrievalQuery as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except RerankerProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except InvalidRerankingQuery as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        results=[SearchResultItem.model_validate(
            result) for result in results],
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    try:
        result = service.ask(
            request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            document_id=str(
                request.document_id) if request.document_id else None,
        )
    except (InvalidRAGQuery, InvalidRerankingQuery, InvalidRetrievalQuery) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (LLMConfigurationError, LLMProviderError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except LLMResponseError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except RAGError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    return AskResponse(
        query=request.query,
        answer=result.answer,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        citations=[
            SourceCitationItem(
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                chunk_index=citation.chunk_index,
                page_number=citation.page_number,
                chunk_text=citation.chunk_text,
                similarity=citation.similarity,
                bm25_score=citation.bm25_score,
                hybrid_score=citation.hybrid_score,
                reranker_score=citation.reranker_score,
            )
            for citation in result.citations
        ],
    )


@router.post(
    "/questions/generate",
    response_model=QuestionGenerationResponse,
    status_code=status.HTTP_201_CREATED,
)
def generate_questions(
    request: QuestionGenerationRequest,
    service: AssessmentGenerationService = Depends(
        get_assessment_generation_service),
) -> QuestionGenerationResponse:
    try:
        run, questions = service.generate(request)
    except (DocumentNotFoundError,) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (InvalidRerankingQuery, InvalidRetrievalQuery) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except (AssessmentNoContextError, AssessmentValidationError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except LLMResponseError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except AssessmentPersistenceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
    except AssessmentGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    return QuestionGenerationResponse(
        run=GenerationRunResponse.model_validate(run),
        questions=[QuestionResponse.model_validate(
            question) for question in questions],
    )


@router.get("/questions/{question_id}", response_model=QuestionResponse)
def get_question(
    question_id: str,
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> QuestionResponse:
    question = repository.get_question(question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Question was not found")
    return QuestionResponse.model_validate(question)


@router.get("/questions", response_model=QuestionListResponse)
def list_questions(
    filters: QuestionListFilter = Depends(),
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> QuestionListResponse:
    questions = repository.list_questions(
        document_id=str(filters.document_id) if filters.document_id else None,
        question_type=filters.question_type.value if filters.question_type else None,
        difficulty=filters.difficulty.value if filters.difficulty else None,
        bloom_level=filters.bloom_level.value if filters.bloom_level else None,
        topic=filters.topic,
        skill=filters.skill,
        limit=filters.limit,
        offset=filters.offset,
    )
    return QuestionListResponse(
        questions=[QuestionResponse.model_validate(
            question) for question in questions],
        limit=filters.limit,
        offset=filters.offset,
    )


@router.get(
    "/question-generation-runs/{run_id}",
    response_model=GenerationRunResponse,
)
def get_generation_run(
    run_id: str,
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> GenerationRunResponse:
    run = repository.get_generation_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question generation run was not found",
        )
    return GenerationRunResponse.model_validate(run)


@router.get(
    "/questions/{question_id}/quality",
    response_model=QualityEvaluationListResponse,
)
def get_question_quality(
    question_id: str,
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> QualityEvaluationListResponse:
    question = repository.get_question(question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question was not found",
        )
    evaluations = repository.get_question_quality(question_id)
    return QualityEvaluationListResponse(
        evaluations=[
            QualityEvaluationResponse.model_validate(item) for item in evaluations
        ],
    )


@router.get(
    "/question-generation-runs/{run_id}/quality",
    response_model=QualityEvaluationListResponse,
)
def get_generation_run_quality(
    run_id: str,
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> QualityEvaluationListResponse:
    run = repository.get_generation_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question generation run was not found",
        )
    evaluations = repository.get_run_quality(run_id)
    return QualityEvaluationListResponse(
        evaluations=[
            QualityEvaluationResponse.model_validate(item) for item in evaluations
        ],
    )


# ---------------------------------------------------------------------------
# M4A – Quiz sessions
# ---------------------------------------------------------------------------

@router.post(
    "/quiz-sessions",
    response_model=QuizSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_quiz_session(
    request: QuizSessionCreateRequest,
    learner_repo: LearnerRepository = Depends(get_learner_repository),
    assessment_repo: AssessmentRepository = Depends(get_assessment_repository),
) -> QuizSessionResponse:
    """Create a new quiz session for a learner over a set of question IDs."""
    # Validate every supplied question ID exists.
    missing = [
        qid for qid in request.question_ids
        if assessment_repo.get_question(qid) is None
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question IDs not found: {missing}",
        )
    session = learner_repo.create_session(
        learner_id=request.learner_id,
        question_ids=request.question_ids,
    )
    return QuizSessionResponse.model_validate(session)


@router.get(
    "/quiz-sessions/{session_id}",
    response_model=QuizSessionResponse,
)
def get_quiz_session(
    session_id: str,
    learner_repo: LearnerRepository = Depends(get_learner_repository),
) -> QuizSessionResponse:
    """Return quiz session metadata."""
    session = learner_repo.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz session was not found",
        )
    return QuizSessionResponse.model_validate(session)


@router.get(
    "/quiz-sessions/{session_id}/questions",
    response_model=QuizSessionQuestionsResponse,
)
def get_quiz_session_questions(
    session_id: str,
    learner_repo: LearnerRepository = Depends(get_learner_repository),
    assessment_repo: AssessmentRepository = Depends(get_assessment_repository),
) -> QuizSessionQuestionsResponse:
    """Return ordered learner-safe questions for a quiz session.

    correct_option_key and citations are intentionally excluded.
    """
    session = learner_repo.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz session was not found",
        )
    questions = []
    for qid in session.question_ids:
        question = assessment_repo.get_question(qid)
        if question is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {qid} was not found",
            )
        learner_options = [
            LearnerQuestionOptionResponse(
                id=opt.id,
                option_key=opt.option_key,
                option_text=opt.option_text,
                display_order=opt.display_order,
            )
            for opt in question.options
        ]
        questions.append(
            LearnerQuestionResponse(
                id=question.id,
                generation_run_id=question.generation_run_id,
                document_id=question.document_id,
                question_type=question.question_type,
                stem=question.stem,
                explanation=question.explanation,
                difficulty=question.difficulty,
                bloom_level=question.bloom_level,
                topic=question.topic,
                skill=question.skill,
                content_version=question.content_version,
                created_at=question.created_at,
                options=learner_options,
            )
        )
    return QuizSessionQuestionsResponse(session_id=session_id, questions=questions)


# ---------------------------------------------------------------------------
# M4B – Answer submission and session completion
# ---------------------------------------------------------------------------

@router.post(
    "/quiz-sessions/{session_id}/answers",
    response_model=AnswerSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_answer(
    session_id: str,
    request: AnswerSubmissionRequest,
    learner_repo: LearnerRepository = Depends(get_learner_repository),
    assessment_repo: AssessmentRepository = Depends(get_assessment_repository),
) -> AnswerSubmissionResponse:
    """Submit a learner's answer for one question in a quiz session.

    - 404 if the session or question does not exist.
    - 422 if the session is not active (already completed).
    - 409 if an answer for this question has already been submitted.
    """
    quiz_session = learner_repo.get_session(session_id)
    if quiz_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz session was not found",
        )
    if quiz_session.status != "active":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Answers can only be submitted to an active session",
        )
    if request.question_id not in quiz_session.question_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question does not belong to this session",
        )
    question = assessment_repo.get_question(request.question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question was not found",
        )
    is_correct = score_mcq_answer(
        request.submitted_option_key, question.correct_option_key
    )
    try:
        submission = learner_repo.submit_answer(
            session_id=session_id,
            question_id=request.question_id,
            learner_id=quiz_session.learner_id,
            submitted_option_key=request.submitted_option_key,
            is_correct=is_correct,
        )
    except DuplicateSubmissionError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)
        ) from error
    return AnswerSubmissionResponse.model_validate(submission)


@router.get(
    "/quiz-sessions/{session_id}/answers/{question_id}",
    response_model=AnswerSubmissionResponse,
)
def get_answer_submission(
    session_id: str,
    question_id: str,
    learner_repo: LearnerRepository = Depends(get_learner_repository),
) -> AnswerSubmissionResponse:
    """Return the submission for a specific question in a session, or 404."""
    submission = learner_repo.get_submission(session_id, question_id)
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission was not found",
        )
    return AnswerSubmissionResponse.model_validate(submission)


@router.post(
    "/quiz-sessions/{session_id}/complete",
    response_model=SessionResultResponse,
)
def complete_quiz_session(
    session_id: str,
    learner_repo: LearnerRepository = Depends(get_learner_repository),
) -> SessionResultResponse:
    """Mark a session as completed and return the aggregate score.

    - 404 if the session does not exist.
    - 409 if the session is already completed.
    """
    quiz_session = learner_repo.get_session(session_id)
    if quiz_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz session was not found",
        )
    if quiz_session.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Quiz session is already completed",
        )
    completed_session = learner_repo.complete_session(session_id)
    submissions = learner_repo.get_submissions_for_session(session_id)
    total = len(quiz_session.question_ids)
    correct_count = sum(1 for s in submissions if s.is_correct)
    score_percent = (correct_count / total * 100.0) if total > 0 else 0.0
    return SessionResultResponse(
        session_id=session_id,
        learner_id=completed_session.learner_id,
        status=completed_session.status,
        total_questions=total,
        correct_count=correct_count,
        score_percent=round(score_percent, 2),
        submissions=[
            AnswerSubmissionResponse.model_validate(s) for s in submissions
        ],
    )


# ---------------------------------------------------------------------------
# M4C – Learner performance and weak-topic detection
# ---------------------------------------------------------------------------

@router.get(
    "/learners/{learner_id}/performance",
    response_model=LearnerPerformanceResponse,
)
def get_learner_performance(
    learner_id: str,
    learner_repo: LearnerRepository = Depends(get_learner_repository),
) -> LearnerPerformanceResponse:
    """Return overall, per-topic, and per-skill performance for a learner.

    Aggregates all submitted answers for the learner across all sessions.
    Returns empty aggregates (zero counts) when no submissions exist.
    """
    submissions = learner_repo.get_all_submissions_for_learner(learner_id)
    total_attempts = len(submissions)
    total_correct = sum(1 for s in submissions if s.is_correct)

    from app.domain.assessment import compute_accuracy
    overall_accuracy = compute_accuracy(total_attempts, total_correct)

    topic_perfs = learner_repo.get_topic_performance_for_learner(learner_id)
    skill_perfs = learner_repo.get_skill_performance_for_learner(learner_id)

    return LearnerPerformanceResponse(
        learner_id=learner_id,
        total_attempts=total_attempts,
        total_correct=total_correct,
        overall_accuracy=overall_accuracy,
        by_topic=[
            AreaPerformanceResponse(
                label=p.label,
                attempts=p.attempts,
                correct=p.correct,
                accuracy=p.accuracy,
            )
            for p in topic_perfs
        ],
        by_skill=[
            AreaPerformanceResponse(
                label=p.label,
                attempts=p.attempts,
                correct=p.correct,
                accuracy=p.accuracy,
            )
            for p in skill_perfs
        ],
    )


@router.get(
    "/learners/{learner_id}/weak-topics",
    response_model=LearnerWeakTopicsResponse,
)
def get_learner_weak_topics(
    learner_id: str,
    learner_repo: LearnerRepository = Depends(get_learner_repository),
) -> LearnerWeakTopicsResponse:
    """Return weak topics and skills for a learner using deterministic policy.

    A topic or skill is weak when it has at least WEAK_TOPIC_MIN_ATTEMPTS
    attempts and accuracy <= WEAK_TOPIC_ACCURACY_THRESHOLD.
    Returns empty lists when there are no weak areas or no submissions.
    """
    topic_perfs = learner_repo.get_topic_performance_for_learner(learner_id)
    skill_perfs = learner_repo.get_skill_performance_for_learner(learner_id)

    weak_topics = detect_weak_areas(topic_perfs)
    weak_skills = detect_weak_areas(skill_perfs)

    return LearnerWeakTopicsResponse(
        learner_id=learner_id,
        min_attempts_threshold=WEAK_TOPIC_MIN_ATTEMPTS,
        accuracy_threshold=WEAK_TOPIC_ACCURACY_THRESHOLD,
        weak_topics=[
            WeakAreaResponse(
                label=w.label,
                attempts=w.attempts,
                correct=w.correct,
                accuracy=w.accuracy,
            )
            for w in weak_topics
        ],
        weak_skills=[
            WeakAreaResponse(
                label=w.label,
                attempts=w.attempts,
                correct=w.correct,
                accuracy=w.accuracy,
            )
            for w in weak_skills
        ],
    )
