from collections.abc import Generator
from importlib import import_module

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies import get_document_service
from app.db.base import Base
from app.db.session import get_db
from app.repositories.documents import DocumentRepository
from app.services.documents import DocumentService, LocalDocumentStorage
import app.models  # Registers models for test metadata.

main_module = import_module("app.main")
fastapi_app = main_module.app


@pytest.fixture
def session_factory(tmp_path) -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def client(session_factory, tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    monkeypatch.setattr(main_module, "engine", session_factory.kw["bind"])

    def override_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    def service_override() -> DocumentService:
        db = session_factory()
        return DocumentService(DocumentRepository(db), LocalDocumentStorage(tmp_path / "uploads"))

    fastapi_app.dependency_overrides[get_document_service] = service_override
    fastapi_app.dependency_overrides[get_db] = override_db
    with TestClient(fastapi_app) as test_client:
        yield test_client
    fastapi_app.dependency_overrides.clear()
