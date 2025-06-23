from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from config.errors.exceptions import VectorStoreNotFoundException, InvalidRequestException

def register_exception_handlers(app: FastAPI):
    """
    Register custom exception handlers with the FastAPI app.
    """
    @app.exception_handler(VectorStoreNotFoundException)
    async def vector_store_not_found_handler(request: Request, exc: VectorStoreNotFoundException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )

    @app.exception_handler(InvalidRequestException)
    async def invalid_request_handler(request: Request, exc: InvalidRequestException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred. Please try again later."},
        )