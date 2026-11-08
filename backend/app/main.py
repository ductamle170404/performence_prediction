from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.dependencies import get_adapter, get_settings
from app.presentation.routes import assessment_routes, course_routes, instructor_routes, prediction_routes

app = FastAPI(title="Performance Prediction Extension", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/health", tags=["system"])
def health():
    adapter = get_adapter()
    try:
        files = adapter.validate_schema()
        return {"status": "ok", "data_files": files}
    except AppError as exc:
        return JSONResponse(status_code=exc.status_code, content={"status": "error", "detail": exc.message})


for module in (course_routes, assessment_routes, prediction_routes, instructor_routes):
    app.include_router(module.router)
