class AppError(Exception):
    def __init__(self, message: str, code: str = "app_error"):
        super().__init__(message)
        self.code = code
        self.message = message


class ValidationAppError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="validation_error")


class RetrievalError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="retrieval_error")


class GenerationError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="generation_error")


class ExternalServiceError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="external_service_error")


class DocumentParseError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="document_parse_error")
