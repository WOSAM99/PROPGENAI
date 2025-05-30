from app.utils.response import error_response, success_response
from fastapi import APIRouter
from loguru import logger

router = APIRouter()

@router.post("/success")
async def root():
    logger.info("Root endpoint was accessed")
    return success_response("Hello, FastAPI with Loguru!", 200)

@router.get("/error")
async def error():
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("An error occurred: Division by zero!")
        return error_response("Something went wrong!", 500)