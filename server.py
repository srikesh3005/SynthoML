#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io
import os
import traceback
from datetime import datetime
import pandas as pd
import subprocess
import sys
from inference import generate, get_model_info

# Check if CTGAN is available
try:
    import ctgan
    CTGAN_AVAILABLE = True
except ImportError:
    try:
        import sdv
        CTGAN_AVAILABLE = True
    except ImportError:
        CTGAN_AVAILABLE = False
        print("⚠️  WARNING: CTGAN/SDV not available (likely Python 3.14+)")
        print("    Using simple statistical generator as fallback")
        print("    For deep learning quality, use Python 3.9-3.13")

app = FastAPI(
    title="CTGAN Synthetic Data API",
    description="Generate synthetic medical data using CTGAN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


training_status = {
    "is_training": False,
    "progress": 0,
    "message": "No training in progress",
    "current_epoch": 0,
    "total_epochs": 0
}

@app.get("/")
async def root():
    return {
        "message": "CTGAN Synthetic Data API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/generate": "Generate synthetic data (POST with query param 'n')",
            "/model-info": "Get model information",
            "/upload-train": "Upload CSV and train model (POST)",
            "/training-status": "Get training status (GET)"
        }
    }


@app.get("/health")
async def health_check():
    try:
        info = get_model_info()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": True,
            "library": info["library"],
            "ctgan_available": CTGAN_AVAILABLE,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": False,
            "error": str(e),
            "ctgan_available": CTGAN_AVAILABLE,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }


@app.get("/model-info")
async def model_info():
    try:
        info = get_model_info()
        return {
            "success": True,
            "data": info
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "data": None,
            "message": "No trained model found. Please upload and train a dataset first."
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": f"Failed to load model: {str(e)}"
        }


@app.post("/generate")
async def generate_synthetic_data(
    n: int = Query(
        default=1000,
        ge=1,
        le=100000,
        description="Number of synthetic samples to generate (1-100000)"
    ),
    seed: int = Query(
        default=None,
        description="Random seed for reproducibility (optional)"
    )
):
    try:
        print(f"Generating {n} samples (seed: {seed})...")
        synthetic_df = generate(n=n, seed=seed)
        
        csv_buffer = io.StringIO()
        synthetic_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_string = csv_buffer.getvalue()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_data_{n}rows_{timestamp}.csv"
        
        return StreamingResponse(
            iter([csv_string]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameters: {str(e)}"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Model file not found. Please train the model first using train_and_save_ctgan.py"
        )
    except Exception as e:
        print(f"Error generating synthetic data:")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate synthetic data: {str(e)}"
        )


def train_model_background(file_path: str, epochs: int):
    global training_status
    try:
        training_status["is_training"] = True
        training_status["message"] = f"Training started with {epochs} epochs..."
        training_status["total_epochs"] = epochs
        training_status["progress"] = 0
        
        print(f"Starting training: {file_path}, epochs: {epochs}")
        
        result = subprocess.run(
            [sys.executable, "train_and_save_ctgan.py", "--data", file_path, "--epochs", str(epochs)],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print(f"Training subprocess completed with return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            training_status["is_training"] = False
            training_status["progress"] = 100
            training_status["message"] = "Training completed successfully!"
            
            if os.path.exists("inference.py"):
                import importlib
                import inference
                importlib.reload(inference)
        else:
            training_status["is_training"] = False
            error_msg = result.stderr or result.stdout or "Unknown error"
            training_status["message"] = f"Training failed: {error_msg}"
            
    except Exception as e:
        training_status["is_training"] = False
        training_status["message"] = f"Training error: {str(e)}"


@app.post("/upload-train")
async def upload_and_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    epochs: int = Query(100, ge=10, le=1000)
):
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(
            status_code=400,
            detail="Training already in progress. Please wait for it to complete."
        )
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    try:
        contents = await file.read()
        # Try multiple encodings to read the uploaded file
        df = None
        for encoding in ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if df is None:
            raise HTTPException(
                status_code=400,
                detail="Unable to read CSV file. Please ensure it's a valid CSV with UTF-8 encoding."
            )
        
        if len(df) < 5:
            raise HTTPException(
                status_code=400,
                detail="Dataset must have at least 5 rows"
            )
        
        # Clean string data for Windows compatibility
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('').astype(str)
                df[col] = df[col].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii') if isinstance(x, str) else str(x))
                df[col] = df[col].str.strip()
                df[col] = df[col].replace('', 'Unknown')
        
        # Save with UTF-8-sig encoding for Windows compatibility
        upload_path = f"uploaded_data.csv"
        df.to_csv(upload_path, index=False, encoding='utf-8-sig')
        
        background_tasks.add_task(train_model_background, upload_path, epochs)
        
        return {
            "success": True,
            "message": f"Training started with {len(df)} rows, {len(df.columns)} columns",
            "rows": len(df),
            "columns": list(df.columns),
            "epochs": epochs
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/training-status")
async def get_training_status():
    return training_status


@app.get("/docs")
async def custom_docs():
    return {
        "message": "Visit /docs for interactive API documentation"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\nStarting CTGAN Synthetic Data API Server...")
    print("   Server will run at: http://localhost:8000")
    print("   API docs at: http://localhost:8000/docs")
    print("   Health check: http://localhost:8000/health")
    print("\n   Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
