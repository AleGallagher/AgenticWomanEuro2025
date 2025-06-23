from fastapi import HTTPException, status

class VectorStoreNotFoundException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector store is not initialized or missing. Please check the configuration.",
        )

class InvalidRequestException(HTTPException):
    def __init__(self, message: str = "Invalid request"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )