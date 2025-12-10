"""
FastAPI Application for Transit Coverage Classification

This module provides REST API endpoints for model predictions.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import logging

from src.api.prediction_service import get_prediction_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Transit Coverage Classifier API",
    description="API for predicting transit coverage classification (well-served vs underserved areas)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Pydantic models for request/response validation
class Features(BaseModel):
    """Feature input model."""
    stop_count: float = Field(..., description="Number of transit stops in the cell")
    route_count: float = Field(..., description="Number of unique routes in the cell")
    daily_trips: float = Field(..., description="Total daily trips in the cell")
    stop_density: float = Field(..., description="Stops per square kilometer")
    route_diversity: float = Field(..., description="Route diversity metric")
    stop_count_norm: float = Field(..., description="Normalized stop count")
    route_count_norm: float = Field(..., description="Normalized route count")
    daily_trips_norm: float = Field(..., description="Normalized daily trips")
    
    class Config:
        json_schema_extra = {
            "example": {
                "stop_count": 5.0,
                "route_count": 3.0,
                "daily_trips": 450.0,
                "stop_density": 20.0,
                "route_diversity": 0.8,
                "stop_count_norm": 0.5,
                "route_count_norm": 0.4,
                "daily_trips_norm": 0.6
            }
        }


class PredictionRequest(BaseModel):
    """Single prediction request model."""
    features: Features = Field(..., description="Feature values for prediction")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    predictions: List[Dict[str, Any]] = Field(
        ..., 
        description="List of predictions with cell_id and features"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "cell_id": "cell_0_0",
                        "features": {
                            "stop_count": 5.0,
                            "route_count": 3.0,
                            "daily_trips": 450.0,
                            "stop_density": 20.0,
                            "route_diversity": 0.8,
                            "stop_count_norm": 0.5,
                            "route_count_norm": 0.4,
                            "daily_trips_norm": 0.6
                        }
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: int = Field(..., description="Predicted class (0=underserved, 1=well-served)")
    predicted_class: str = Field(..., description="Class label (underserved or well_served)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    confidence: float = Field(..., description="Confidence of prediction (max probability)")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions with cell_id")
    total_predictions: int = Field(..., description="Total number of predictions")
    avg_latency_ms: float = Field(..., description="Average latency per prediction")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    metadata_loaded: bool = Field(..., description="Whether metadata is loaded")
    model_version: str = Field(..., description="Model version")


class ModelInfoResponse(BaseModel):
    """Model information response model."""
    model_name: str
    model_type: str
    model_version: str
    feature_names: List[str]
    n_features: int
    target_classes: List[str]
    val_f1_score: Optional[float]
    export_date: Optional[str]


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Transit Coverage Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API and model.
    """
    try:
        service = get_prediction_service()
        health_status = service.health_check()
        
        if health_status['status'] == 'healthy':
            return health_status
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unhealthy"
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get model information.
    
    Returns metadata about the loaded model.
    """
    try:
        service = get_prediction_service()
        return service.get_model_info()
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Accepts feature values and returns a prediction with probabilities.
    """
    try:
        service = get_prediction_service()
        
        # Convert Pydantic model to dict
        features_dict = request.features.dict()
        
        # Make prediction
        result = service.predict(features_dict)
        
        # Check latency constraint
        if result['latency_ms'] > 200:
            logger.warning(f"Latency exceeded 200ms: {result['latency_ms']:.2f}ms")
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Accepts a list of feature dictionaries and returns predictions for all.
    """
    try:
        service = get_prediction_service()
        
        # Extract features from request
        features_list = []
        cell_ids = []
        
        for item in request.predictions:
            cell_ids.append(item.get('cell_id', f'cell_{len(cell_ids)}'))
            features_list.append(item['features'])
        
        # Make batch predictions
        import time
        start_time = time.time()
        
        results = service.predict_batch(features_list)
        
        total_time_ms = (time.time() - start_time) * 1000
        avg_latency_ms = total_time_ms / len(features_list)
        
        # Add cell_id to each result
        predictions_with_ids = []
        for cell_id, result in zip(cell_ids, results):
            result['cell_id'] = cell_id
            predictions_with_ids.append(result)
        
        return {
            'predictions': predictions_with_ids,
            'total_predictions': len(predictions_with_ids),
            'avg_latency_ms': round(avg_latency_ms, 2)
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Transit Coverage Classifier API...")
    logger.info("API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
