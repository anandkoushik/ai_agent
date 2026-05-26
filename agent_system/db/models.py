from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, JSON, Text, Float
from sqlalchemy.orm import declarative_base, relationship
import datetime
import uuid

Base = declarative_base()

def get_uuid():
    return str(uuid.uuid4())

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=get_uuid)
    dataset_hash = Column(String, unique=True, index=True, nullable=False) # Prevent duplicate ingestion
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    
    # Lineage tracking
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    
    id = Column(String, primary_key=True, default=get_uuid)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    version_tag = Column(String, nullable=False)
    
    # Lineage details
    parent_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=True)
    preprocessing_operations = Column(JSON, default=list)
    augmentations_applied = Column(JSON, default=list)
    train_split_ratio = Column(Float, nullable=True)
    validation_split_ratio = Column(Float, nullable=True)
    validation_status = Column(String, default="pending") # pending, valid, invalid
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    dataset = relationship("Dataset", back_populates="versions")

class Workflow(Base):
    __tablename__ = "workflows"
    
    id = Column(String, primary_key=True, default=get_uuid)
    workflow_id = Column(String, unique=True, index=True, nullable=False)
    status = Column(String, default="queued", index=True) # queued, running, completed, failed, paused
    dag_state = Column(JSON, default=dict) # Persist resumable state
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    jobs = relationship("Job", back_populates="workflow", cascade="all, delete-orphan")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=get_uuid)
    workflow_id = Column(String, ForeignKey("workflows.workflow_id"), nullable=False)
    job_type = Column(String, nullable=False)
    job_status = Column(String, default="pending", index=True)
    job_metadata = Column(JSON, default=dict) # retry checkpoints, hyperparams
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    workflow = relationship("Workflow", back_populates="jobs")

class MemoryEntry(Base):
    """Stores conversation history and reasoning, including compressed variants."""
    __tablename__ = "memory_entries"
    
    id = Column(String, primary_key=True, default=get_uuid)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False) # user, assistant, tool, system
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    
    is_summarized = Column(Boolean, default=False)
    is_critical_context = Column(Boolean, default=False) # e.g. constraints, active workflows
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

class FailureMemory(Base):
    """Structured metadata for adaptive self-healing. Linked to ChromaDB vectors."""
    __tablename__ = "failure_memory"
    
    id = Column(String, primary_key=True, default=get_uuid)
    chroma_vector_id = Column(String, unique=True, nullable=False) # Links to ChromaDB
    
    error_pattern = Column(Text, nullable=False)
    root_cause = Column(Text, nullable=True)
    applied_fix = Column(Text, nullable=True)
    outcome_success = Column(Boolean, nullable=True)
    
    # Validation contexts
    hardware_context = Column(JSON, nullable=True)
    model_type = Column(String, nullable=True)
    dataset_type = Column(String, nullable=True)
    framework_compatibility = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    id = Column(String, primary_key=True, default=get_uuid)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    
    # Derived lineage
    dataset_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=True)
    hyperparameters = Column(JSON, default=dict)
    
    status = Column(String, default="training") # training, ready, archived
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class TelemetryLog(Base):
    __tablename__ = "telemetry_logs"
    
    id = Column(String, primary_key=True, default=get_uuid)
    event_type = Column(String, nullable=False)
    payload = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
