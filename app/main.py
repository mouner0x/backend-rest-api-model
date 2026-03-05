from fastapi import FastAPI

from .routes.auth_routes import router as auth_router
from .routes.dataset_routes import router as dataset_router
from .routes.dashboard_routes import router as dashboard_router
from .routes.predict_routes import router as predict_router









app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # React Vite
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/api/v1")
app.include_router(dataset_router, prefix="/api/v1")
app.include_router(dashboard_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
