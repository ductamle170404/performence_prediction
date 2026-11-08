"""Application-level exceptions, mapped to HTTP responses in main.py."""


class AppError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404


class ForbiddenError(AppError):
    status_code = 403


class ValidationAppError(AppError):
    status_code = 422


class DataSchemaError(AppError):
    status_code = 500
