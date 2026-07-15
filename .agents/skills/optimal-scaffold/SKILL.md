---
name: optimal-scaffold
description: Use when creating new features, modules, routes, services, or agents in the any type of python project. Use when unsure how to place a new file, name a class, wire a dependency, add an endpoint, or write a test in this codebase. Triggers on "add endpoint", "new service", "new agent", "new route", "where does X go", "how to structure", or "follow the project pattern".
---

# Optimal python Scaffold

Reference for the layered API architecture used in a python project. Follow these patterns for every new feature.

## Architecture Layers

```
Streamlit or another type of UI  (src/app/frontend/)
      ↓
FastAPI Routes (src/app/api/routes/<feature>/)
      ↓
Services       (src/app/services/<feature>/)
      ↓
Domain         (src/app/domain/)
      ↓
ML Binaries if are needed  (src/app/ml_binaries/) via services/helper.py
```

Each layer has one job. Never skip layers or import upward.

## Directory Layout

```
src/app/
├── __init__.py          # version + configure_container()
├── __main__.py          # multiprocess launcher (API + UI)
├── settings.py          # Pydantic Settings — ports, paths, env vars
│
├── api/
│   ├── app.py           # FastAPI factory (include_router, add_exception_handler)
│   ├── dependencies.py  # Annotated DI aliases consumed by route functions
│   ├── schema.py        # BaseSchema(BaseModel + ExamplerMixin + CamelCase)
│   ├── error_handlers/  # One file per exception type → JSONResponse
│   └── routes/
│       └── <feature>/
│           ├── __init__.py
│           ├── endpoints.py   # route functions decorated with @inject
│           ├── schemas.py     # FeatureRequest / FeatureResponse
│           └── examples.py    # EXAMPLES dict for OpenAPI
│
├── domain/
│   ├── base.py          # BaseEntity (Pydantic + ExamplerMixin + CamelCase)
│   ├── ml_model.py      # MLModel Protocol (@runtime_checkable)
│   └── <entity>.py      # One file per domain entity
│
├── services/
│   ├── helper.py        # load_model / save_model (joblib)
│   └── <feature>/
│       ├── __init__.py
│       ├── service.py       # FeatureService class
│       └── exceptions.py    # Custom exception dataclasses
│
├── injections/
│   ├── __init__.py      # configure_container() with @lru_cache
│   ├── production.py    # Container(DeclarativeContainer) — wires services
│   └── test.py          # TestContainer — overrides for tests
│
├── frontend/
│   ├── __init__.py      # run_streamlit() subprocess launcher
│   ├── home.py          # st.navigation + page router
│   └── pages/
│       └── <page>.py    # One Streamlit page per feature
│
└── utils/
    └── exampler.py      # ExamplerMixin — create_example() / create_examples()
```

## Adding a New Feature (checklist)

### 1. Domain entity
```python
# src/app/domain/<entity>.py
from app.domain.base import BaseEntity

class MyEntity(BaseEntity):
    field_name: float = Field(gt=0)
```

### 2. Service
```python
# src/app/services/<feature>/service.py
class MyService:
    def do_thing(self, entity: MyEntity) -> MyOutput: ...

# src/app/services/<feature>/exceptions.py
@dataclass
class MyFeatureError(Exception):
    message: str
```

### 3. Wire into container
```python
# src/app/injections/production.py
my_service = providers.Factory(MyService)
```

### 4. DI alias for routes
```python
# src/app/api/dependencies.py
MyServiceDependency = Annotated[MyService, Depends(Provide["my_service"])]
```

### 5. Route schemas
```python
# src/app/api/routes/<feature>/schemas.py
class MyRequest(BaseSchema):
    field_name: float = Field(gt=0)

class MyResponse(BaseSchema):
    result: float
```

### 6. Endpoint
```python
# src/app/api/routes/<feature>/endpoints.py
router = APIRouter(prefix="/my-feature", tags=["My Feature"])

@router.post("/", response_model=MyResponse)
@inject
def my_endpoint(request: MyRequest, service: MyServiceDependency) -> MyResponse:
    result = service.do_thing(request.to_entity())
    return MyResponse(result=result)
```

### 7. Register router
```python
# src/app/api/routes/__init__.py
ROUTERS = [..., my_feature_router]
```

### 8. Error handler (if custom exception)
```python
# src/app/api/error_handlers/my_feature.py
def my_feature_error_handler(request: Request, exc: MyFeatureError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": exc.message})

# src/app/api/error_handlers/__init__.py
EXCEPTION_HANDLERS = {..., MyFeatureError: my_feature_error_handler}
```

### 9. Test
```python
# tests/services/<feature>/test_my_service.py
# tests/api/routes/<feature>/test_my_endpoint.py
```

## Model Conventions

| Base class | Where | Extras |
|---|---|---|
| `BaseEntity` | `domain/` | ExamplerMixin, CamelCase aliases |
| `BaseSchema` | `api/schema.py` | ExamplerMixin, CamelCase aliases |
| `MLModel` | Protocol in `domain/ml_model.py` | `fit()` + `predict()` |

All Pydantic models use `model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)` via the base classes.

Generate test data: `MySchema.create_example()` or `MySchema.create_examples(5)`.

## Testing Patterns

```python
# tests/conftest.py — override container for ALL tests
@pytest.fixture(scope="session", autouse=True)
def setup_container():
    container = TestContainer()
    container.wire(modules=[...])

# tests/api/conftest.py — shared test client
@pytest.fixture
def client(setup_container) -> TestClient:
    return TestClient(app)

# Snapshot testing (API contracts)
def test_health(client, snapshot):
    response = client.get("/health")
    assert response.json() == snapshot  # syrupy
```

Coverage must stay ≥ 80%. Omit `__main__.py`, `api/__init__.py`, `frontend/**`, `settings.py`.

## Dev Workflow

```bash
uv sync --all-groups          # install all deps

uv run poe serve              # API (:8000) + Streamlit (:10000)
uv run poe serve-api          # API only → /docs
uv run poe serve-ui           # Streamlit only

uv run poe test               # pytest + coverage
uv run poe check-coverage     # fail if < 80%
uv run poe format             # pre-commit (ruff, pylint, mypy, bandit, gitleaks)

uv run poe docker-build       # build image
uv run poe docker-run         # run with .env file

uv run poe version-bump       # semantic-release (no tag)
```

## Settings Pattern

```python
# src/app/settings.py
class _Settings(BaseSettings):
    MY_CONFIG: str = "default"

    @property
    def DERIVED_PATH(self) -> Path:
        return self.ROOT_PATH / "some/path"

settings = _Settings()
```

Import as `from app.settings import settings` everywhere. Never pass config as constructor args when `settings` suffices.

## Naming Quick Reference

| Thing | Convention | Example |
|---|---|---|
| Classes | PascalCase | `TrainingService` |
| Functions | snake_case | `load_model` |
| Constants/registries | UPPER_CASE | `ROUTERS`, `EXCEPTION_HANDLERS` |
| Private | `_` prefix | `_Settings` |
| Test files | `test_<module>.py` | `test_training_service.py` |
| Exception files | `exceptions.py` per service | `services/training/exceptions.py` |

## Common Mistakes

| Mistake | Fix |
|---|---|
| Importing a service directly into a route without DI | Add to container + create `Annotated` alias in `dependencies.py` |
| Putting business logic in endpoints | Move to a service method |
| Raising `HTTPException` inside a service | Raise a domain exception; handle it in `error_handlers/` |
| Skipping `@inject` on endpoint using `Provide` | Every endpoint that uses `Depends(Provide[...])` needs `@inject` |
| Adding a new router without registering it | Add to `ROUTERS` list in `api/routes/__init__.py` |
| Hardcoding paths | Use `settings.SOME_PATH` property |
