from fastapi import APIRouter

from ..contracts import API_CONTRACT_VERSION, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", api_contract_version=API_CONTRACT_VERSION)
