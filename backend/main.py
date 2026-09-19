"""
AI Compiler & Kernel Playground - FastAPI Unified Application
"""

import os
import sys
import argparse
import socket
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to Python sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.environment import detect_environment
from backend.routes.experiments import router as experiments_router
from backend.routes.code_runner import router as code_runner_router
from backend.routes.mlir import router as mlir_router
from backend.routes.max_routes import router as max_router
from backend.routes.history import router as history_router

app = FastAPI(
    title="AI Compiler & Kernel Playground API",
    description="High-Performance Computing, MLIR Dialects, Mojo Kernels & MAX Runtime Engine",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-Routers
app.include_router(experiments_router)
app.include_router(code_runner_router)
app.include_router(mlir_router)
app.include_router(max_router)
app.include_router(history_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI Compiler & Kernel Playground API", "version": "2.0.0"}


@app.get("/api/environment")
async def get_environment():
    """Returns detected host hardware, CPU SIMD capabilities, and compiler toolchains."""
    return detect_environment()


# Mount Static Frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_static_files(full_path: str):
        file_path = os.path.join(frontend_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dir, "index.html"))


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a given port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="AI Compiler & Kernel Playground Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    port = args.port
    # If chosen port is in use, attempt port 8000 or report clearly
    if is_port_in_use(port, args.host):
        alt_port = 8000 if port != 8000 else 8080
        if not is_port_in_use(alt_port, args.host):
            print(f"[NOTE] Port {port} is currently in use. Auto-switching to available port {alt_port}...")
            port = alt_port

    print(f"Starting AI Compiler & Kernel Playground FastAPI server on http://{args.host}:{port}")
    uvicorn.run("backend.main:app", host=args.host, port=port, reload=False)
