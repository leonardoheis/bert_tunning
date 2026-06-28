from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/")
@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "message": "The API is running smoothly.",
    }
