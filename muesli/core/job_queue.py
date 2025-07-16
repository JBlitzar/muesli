"""
Job queue system for handling background processing tasks.

This module provides a job queue for handling asynchronous tasks such as
transcription and summarization. It manages job scheduling, execution,
status tracking, and result handling.
"""

import enum
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union, cast

logger = logging.getLogger(__name__)

# Type for job results
T = TypeVar("T")
# Type for job parameters
P = TypeVar("P")
# Type for progress updates
ProgressCallback = Callable[[float, Optional[str]], None]


class JobStatus(str, enum.Enum):
    """Status of a job in the queue."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    @property
    def is_active(self) -> bool:
        """Check if the job is still active (pending or running)."""
        return self in (JobStatus.PENDING, JobStatus.RUNNING)
    
    @property
    def is_finished(self) -> bool:
        """Check if the job has finished (completed, failed, or cancelled)."""
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class JobType(str, enum.Enum):
    """Types of jobs that can be processed by the queue."""
    
    TRANSCRIPTION = "transcription"
    SUMMARIZATION = "summarization"
    IMPORT = "import"
    EXPORT = "export"
    CUSTOM = "custom"


class JobError(Exception):
    """Base exception for job-related errors."""
    pass


class JobCancelledError(JobError):
    """Exception raised when a job is cancelled."""
    pass


class JobTimeoutError(JobError):
    """Exception raised when a job times out."""
    pass


@dataclass
class Job(Generic[P, T]):
    """
    Represents a job in the queue.
    
    A job is a unit of work that can be executed asynchronously.
    It has parameters, can report progress, and produces a result.
    """
    
    # Job metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: JobType = field(default=JobType.CUSTOM)
    status: JobStatus = field(default=JobStatus.PENDING)
    
    # Job parameters and result
    params: P = field(default_factory=dict)
    result: Optional[T] = field(default=None)
    error: Optional[Exception] = field(default=None)
    
    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = field(default=None)
    completed_at: Optional[datetime] = field(default=None)
    
    # Progress tracking
    progress: float = field(default=0.0)
    progress_message: Optional[str] = field(default=None)
    
    # Callbacks
    on_complete: Optional[Callable[[T], None]] = field(default=None)
    on_error: Optional[Callable[[Exception], None]] = field(default=None)
    on_progress: Optional[ProgressCallback] = field(default=None)
    
    # Execution settings
    timeout: Optional[float] = field(default=None)
    priority: int = field(default=0)  # Higher value = higher priority
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Initialize the job."""
        # Ensure params is a dict if not provided
        if self.params is None:
            self.params = cast(P, {})
    
    def start(self) -> None:
        """Mark the job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now()
        logger.debug(f"Job {self.id} ({self.job_type}) started")
    
    def complete(self, result: T) -> None:
        """Mark the job as completed with the given result."""
        self.status = JobStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()
        self.progress = 1.0
        logger.debug(f"Job {self.id} ({self.job_type}) completed")
        
        if self.on_complete:
            try:
                self.on_complete(result)
            except Exception as e:
                logger.error(f"Error in on_complete callback for job {self.id}: {e}")
    
    def fail(self, error: Exception) -> None:
        """Mark the job as failed with the given error."""
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
        logger.error(f"Job {self.id} ({self.job_type}) failed: {error}")
        
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as e:
                logger.error(f"Error in on_error callback for job {self.id}: {e}")
    
    def cancel(self) -> None:
        """Mark the job as cancelled."""
        if self.status.is_active:
            self.status = JobStatus.CANCELLED
            self.completed_at = datetime.now()
            self.error = JobCancelledError("Job was cancelled")
            logger.debug(f"Job {self.id} ({self.job_type}) cancelled")
            
            if self.on_error:
                try:
                    self.on_error(self.error)
                except Exception as e:
                    logger.error(f"Error in on_error callback for job {self.id}: {e}")
    
    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """
        Update the progress of the job.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message
        """
        self.progress = max(0.0, min(1.0, progress))  # Clamp to [0.0, 1.0]
        self.progress_message = message
        
        if self.on_progress:
            try:
                self.on_progress(self.progress, self.progress_message)
            except Exception as e:
                logger.error(f"Error in on_progress callback for job {self.id}: {e}")
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the job in seconds, or None if not started."""
        if self.started_at is None:
            return None
        
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the job to a dictionary for serialization."""
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "error": str(self.error) if self.error else None,
            "priority": self.priority,
            "depends_on": self.depends_on,
            "duration": self.duration,
        }


class JobQueue:
    """
    Queue for managing and executing background jobs.
    
    The job queue handles scheduling, execution, and status tracking of
    asynchronous jobs such as transcription and summarization.
    """
    
    def __init__(
        self, 
        max_workers: int = 4,
        max_queue_size: int = 100,
        poll_interval: float = 0.1
    ) -> None:
        """
        Initialize the job queue.
        
        Args:
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum number of jobs in the queue
            poll_interval: Interval in seconds to check for job completion
        """
        self.max_workers = max_workers
        self.poll_interval = poll_interval
        
        # Thread pool for executing jobs
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Priority queue for pending jobs
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Job storage
        self._jobs: Dict[str, Job] = {}
        self._active_jobs: Dict[str, Job] = {}
        self._job_futures: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Worker thread
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self._worker_thread.start()
        
        logger.info(f"Job queue initialized with {max_workers} workers")
    
    def add_job(
        self, 
        job_type: JobType,
        params: Any,
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_progress: Optional[ProgressCallback] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        depends_on: Optional[List[str]] = None
    ) -> Job:
        """
        Add a job to the queue.
        
        Args:
            job_type: Type of job to execute
            params: Parameters for the job
            on_complete: Callback when job completes successfully
            on_error: Callback when job fails
            on_progress: Callback for progress updates
            priority: Job priority (higher value = higher priority)
            timeout: Maximum execution time in seconds
            depends_on: List of job IDs this job depends on
            
        Returns:
            The created job
            
        Raises:
            queue.Full: If the queue is full
        """
        job = Job(
            job_type=job_type,
            params=params,
            on_complete=on_complete,
            on_error=on_error,
            on_progress=on_progress,
            priority=priority,
            timeout=timeout,
            depends_on=depends_on or []
        )
        
        with self._lock:
            self._jobs[job.id] = job
            
            # Check if dependencies are satisfied
            if all(self.get_job_status(dep_id) == JobStatus.COMPLETED for dep_id in job.depends_on):
                # Add to queue with negative priority for the priority queue
                # (lower values have higher priority in Python's PriorityQueue)
                self.job_queue.put((-priority, job.id))
            else:
                logger.debug(f"Job {job.id} waiting for dependencies: {job.depends_on}")
        
        logger.debug(f"Added job {job.id} ({job_type}) to queue with priority {priority}")
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID.
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            The job, or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get the status of a job.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            The job status, or None if the job is not found
        """
        job = self.get_job(job_id)
        return job.status if job else None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            True if the job was cancelled, False if not found or already completed
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or not job.status.is_active:
                return False
            
            # Cancel the job
            job.cancel()
            
            # Cancel the future if it's running
            future = self._job_futures.get(job_id)
            if future and not future.done():
                future.cancel()
            
            # Remove from active jobs
            self._active_jobs.pop(job_id, None)
            
            return True
    
    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Optional[Job]:
        """
        Wait for a job to complete.
        
        Args:
            job_id: ID of the job to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            The completed job, or None if the job was not found or timed out
        """
        start_time = time.time()
        while True:
            job = self.get_job(job_id)
            if not job:
                return None
            
            if job.status.is_finished:
                return job
            
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                return None
            
            # Wait a bit before checking again
            time.sleep(self.poll_interval)
    
    def get_all_jobs(self) -> List[Job]:
        """
        Get all jobs in the queue.
        
        Returns:
            List of all jobs
        """
        with self._lock:
            return list(self._jobs.values())
    
    def get_active_jobs(self) -> List[Job]:
        """
        Get all active jobs (pending or running).
        
        Returns:
            List of active jobs
        """
        with self._lock:
            return [job for job in self._jobs.values() if job.status.is_active]
    
    def clear_completed_jobs(self, max_age: Optional[float] = None) -> int:
        """
        Clear completed, failed, or cancelled jobs from the queue.
        
        Args:
            max_age: Maximum age in seconds to keep completed jobs
            
        Returns:
            Number of jobs cleared
        """
        with self._lock:
            now = datetime.now()
            to_remove = []
            
            for job_id, job in self._jobs.items():
                if not job.status.is_active:
                    # Check if the job is old enough to remove
                    if max_age is not None and job.completed_at:
                        age = (now - job.completed_at).total_seconds()
                        if age > max_age:
                            to_remove.append(job_id)
                    else:
                        to_remove.append(job_id)
            
            # Remove the jobs
            for job_id in to_remove:
                self._jobs.pop(job_id, None)
                self._job_futures.pop(job_id, None)
            
            return len(to_remove)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the job queue.
        
        Args:
            wait: Whether to wait for active jobs to complete
        """
        logger.info("Shutting down job queue")
        self._stop_event.set()
        
        if wait:
            if self._worker_thread.is_alive():
                self._worker_thread.join()
            
            # Wait for executor to finish
            self.executor.shutdown(wait=True)
        else:
            # Cancel all active jobs
            with self._lock:
                for job_id in list(self._active_jobs.keys()):
                    self.cancel_job(job_id)
            
            # Shutdown executor without waiting
            self.executor.shutdown(wait=False)
    
    def _process_jobs(self) -> None:
        """Worker thread that processes jobs from the queue."""
        while not self._stop_event.is_set():
            try:
                # Get the next job from the queue with a timeout
                try:
                    _, job_id = self.job_queue.get(timeout=0.5)
                except queue.Empty:
                    # Check for completed dependencies
                    self._check_dependencies()
                    continue
                
                with self._lock:
                    job = self._jobs.get(job_id)
                    if not job or job.status != JobStatus.PENDING:
                        self.job_queue.task_done()
                        continue
                    
                    # Check if dependencies are satisfied
                    deps_satisfied = True
                    for dep_id in job.depends_on:
                        dep_status = self.get_job_status(dep_id)
                        if dep_status != JobStatus.COMPLETED:
                            deps_satisfied = False
                            break
                    
                    if not deps_satisfied:
                        # Put the job back in the queue with reduced priority
                        self.job_queue.put((-job.priority + 1, job.id))
                        self.job_queue.task_done()
                        continue
                    
                    # Mark the job as active
                    self._active_jobs[job.id] = job
                
                # Submit the job to the executor
                future = self.executor.submit(self._execute_job, job)
                with self._lock:
                    self._job_futures[job.id] = future
                
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in job queue worker: {e}")
                time.sleep(1)  # Avoid tight loop on error
    
    def _execute_job(self, job: Job) -> None:
        """
        Execute a job.
        
        This method is run in a worker thread from the thread pool.
        
        Args:
            job: The job to execute
        """
        try:
            # Mark the job as started
            job.start()
            
            # Get the job handler based on job type
            handler = self._get_job_handler(job.job_type)
            if not handler:
                raise ValueError(f"No handler for job type: {job.job_type}")
            
            # Execute the job with timeout if specified
            if job.timeout:
                # TODO: Implement timeout handling
                # This would require more complex execution with a separate thread
                # and event for cancellation
                result = handler(job)
            else:
                result = handler(job)
            
            # Mark the job as completed
            job.complete(result)
            
            # Check for jobs that depend on this one
            self._check_dependencies()
            
        except Exception as e:
            # Mark the job as failed
            job.fail(e)
        finally:
            # Remove from active jobs
            with self._lock:
                self._active_jobs.pop(job.id, None)
    
    def _get_job_handler(self, job_type: JobType) -> Optional[Callable[[Job], Any]]:
        """
        Get the handler function for a job type.
        
        This method should be overridden by subclasses to provide
        specific handlers for different job types.
        
        Args:
            job_type: Type of job
            
        Returns:
            Handler function or None if not supported
        """
        # This is a placeholder - in a real implementation, this would
        # dispatch to actual handler methods or be overridden by subclasses
        logger.warning(f"No handler implemented for job type: {job_type}")
        return None
    
    def _check_dependencies(self) -> None:
        """
        Check for jobs with satisfied dependencies and add them to the queue.
        """
        with self._lock:
            for job_id, job in self._jobs.items():
                if job.status == JobStatus.PENDING and job_id not in self._active_jobs:
                    # Check if all dependencies are satisfied
                    if all(self.get_job_status(dep_id) == JobStatus.COMPLETED for dep_id in job.depends_on):
                        try:
                            # Try to add to queue without blocking
                            self.job_queue.put_nowait((-job.priority, job.id))
                        except queue.Full:
                            # Queue is full, will try again later
                            pass
