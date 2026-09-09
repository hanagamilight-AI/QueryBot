"""
Observability Module for QueryBot
Provides logging, tracing (LangSmith/Langfuse), and dashboards for monitoring agent behavior
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid
from loguru import logger
import os


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(debug: bool = False, log_file: str = "logs/querybot.log"):
    """
    Configure structured logging for the application
    
    Args:
        debug: Enable debug level logging
        log_file: Path to log file
    """
    import sys
    from pathlib import Path
    
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG" if debug else "INFO",
        colorize=True
    )
    
    # File handler with JSON format for production
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
        level="DEBUG" if debug else "INFO",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        serialize=True  # JSON format
    )
    
    logger.info("Logging configured successfully")


# ============================================================================
# TRACING INTEGRATION
# ============================================================================

class TraceContext:
    """Context manager for distributed tracing"""
    
    def __init__(self, trace_id: str = None, span_id: str = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())
        self.parent_span_id: Optional[str] = None
        self.attributes: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def set_attribute(self, key: str, value: Any):
        """Set trace attribute"""
        self.attributes[key] = value
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        if exc_type:
            self.set_attribute("error", str(exc_val))


class LangfuseTracer:
    """
    Langfuse integration for tracing agent operations
    https://langfuse.com/
    """
    
    def __init__(self, public_key: str = None, secret_key: str = None, host: str = None):
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self._client = None
        self._initialized = False
    
    def _init_client(self):
        """Initialize Langfuse client lazily"""
        if not self._initialized and self.public_key and self.secret_key:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
                self._initialized = True
                logger.info("Langfuse tracer initialized")
            except ImportError:
                logger.warning("Langfuse not installed. Install with: pip install langfuse")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
    
    def trace_generation(
        self,
        query: str,
        response: str,
        session_id: str,
        metadata: Dict[str, Any] = None
    ):
        """Trace a generation event"""
        self._init_client()
        
        if not self._client:
            return
        
        try:
            trace = self._client.trace(
                id=str(uuid.uuid4()),
                name="agent_generation",
                user_id=session_id,
                input=query,
                output=response,
                metadata=metadata or {},
                tags=["political-intelligence", "querybot"]
            )
            
            logger.debug(f"Trace sent to Langfuse: {trace.id}")
            return trace
        except Exception as e:
            logger.error(f"Failed to send trace to Langfuse: {e}")
    
    def score_generation(
        self,
        trace_id: str,
        score: float,
        name: str = "quality",
        comment: str = None
    ):
        """Add a score to a generation trace"""
        self._init_client()
        
        if not self._client:
            return
        
        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=score,
                comment=comment
            )
        except Exception as e:
            logger.error(f"Failed to add score to Langfuse: {e}")
    
    def flush(self):
        """Flush pending traces"""
        if self._client:
            self._client.flush()


class LangSmithTracer:
    """
    LangSmith integration for tracing and evaluation
    https://smith.langchain.com/
    """
    
    def __init__(self, api_key: str = None, project: str = None):
        self.api_key = api_key or os.getenv("LANGCHAIN_API_KEY")
        self.project = project or os.getenv("LANGCHAIN_PROJECT", "querybot")
        self._client = None
        self._initialized = False
    
    def _init_client(self):
        """Initialize LangSmith client lazily"""
        if not self._initialized and self.api_key:
            try:
                from langsmith import Client
                self._client = Client(api_key=self.api_key)
                self._initialized = True
                logger.info("LangSmith tracer initialized")
            except ImportError:
                logger.warning("LangSmith not installed. Install with: pip install langsmith")
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith: {e}")
    
    def trace_run(
        self,
        run_type: str,
        name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None
    ):
        """Trace a run"""
        self._init_client()
        
        if not self._client:
            return
        
        try:
            self._client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                tags=tags or [],
                project_name=self.project
            )
        except Exception as e:
            logger.error(f"Failed to create run in LangSmith: {e}")
    
    def evaluate_run(
        self,
        run_id: str,
        feedback_key: str,
        score: float,
        comment: str = None
    ):
        """Add feedback to a run"""
        self._init_client()
        
        if not self._client:
            return
        
        try:
            self._client.create_feedback(
                run_id=run_id,
                key=feedback_key,
                score=score,
                comment=comment
            )
        except Exception as e:
            logger.error(f"Failed to add feedback in LangSmith: {e}")


# ============================================================================
# METRICS COLLECTION
# ============================================================================

class MetricsCollector:
    """
    Collects and aggregates metrics for dashboards
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        key = self._make_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value
        
        # Store detailed record
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "name": name,
            "value": value,
            "labels": labels or {},
            "total": self.counters[key]
        }
        
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(record)
        
        logger.debug(f"Counter incremented: {key} = {self.counters[key]}")
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
        logger.debug(f"Gauge set: {key} = {value}")
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        key = self._make_key(name, labels)
        
        if key not in self.histograms:
            self.histograms[key] = []
        
        self.histograms[key].append(value)
        
        # Keep only last 1000 values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create metric key from name and labels"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {}
        }
        
        for key, values in self.histograms.items():
            if values:
                import statistics
                summary["histograms"][key] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": statistics.median(values),
                    "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else max(values),
                    "p99": sorted(values)[int(len(values) * 0.99)] if len(values) > 100 else max(values)
                }
        
        return summary
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            name = key.split("{")[0] if "{" in key else key
            lines.append(f"# TYPE {name} counter")
            if "{" in key:
                labels = key[key.index("{")+1:key.index("}")]
                lines.append(f"{name}{{{labels}}} {value}")
            else:
                lines.append(f"{name}_total {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            name = key.split("{")[0] if "{" in key else key
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{key} {value}")
        
        return "\n".join(lines)


# ============================================================================
# AGENT BEHAVIOR MONITORING
# ============================================================================

class AgentBehaviorMonitor:
    """
    Monitors agent behavior for anomalies and performance issues
    """
    
    def __init__(self, metrics_collector: MetricsCollector = None):
        self.metrics = metrics_collector or MetricsCollector()
        self.response_times: Dict[str, List[float]] = {}
        self.confidence_scores: List[float] = []
        self.tool_usage: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
    
    def record_query(
        self,
        session_id: str,
        query_length: int,
        response_time_ms: float,
        confidence_score: float,
        tools_used: List[str],
        error: str = None
    ):
        """Record a query event"""
        # Response time metrics
        self.metrics.record_histogram(
            "query_response_time",
            response_time_ms,
            {"session_id": session_id}
        )
        
        # Confidence score metrics
        self.metrics.record_histogram(
            "confidence_score",
            confidence_score
        )
        
        # Tool usage counter
        for tool in tools_used:
            self.tool_usage[tool] = self.tool_usage.get(tool, 0) + 1
            self.metrics.increment_counter("tool_usage", labels={"tool": tool})
        
        # Error tracking
        if error:
            error_type = error[:50]  # Truncate for grouping
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            self.metrics.increment_counter("errors", labels={"type": error_type})
        
        # Query volume
        self.metrics.increment_counter("queries_total")
        
        logger.debug(
            f"Query recorded: session={session_id}, "
            f"response_time={response_time_ms:.2f}ms, "
            f"confidence={confidence_score:.2f}"
        )
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in agent behavior"""
        anomalies = []
        
        # Check for high error rate
        total_queries = self.metrics.counters.get("queries_total", 0)
        total_errors = sum(self.error_counts.values())
        
        if total_queries > 10 and total_errors / total_queries > 0.2:
            anomalies.append({
                "type": "high_error_rate",
                "severity": "high",
                "details": f"Error rate: {total_errors / total_queries:.2%}"
            })
        
        # Check for low average confidence
        if self.confidence_scores:
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
            if avg_confidence < 0.4:
                anomalies.append({
                    "type": "low_confidence",
                    "severity": "medium",
                    "details": f"Average confidence: {avg_confidence:.2f}"
                })
        
        # Check for slow response times
        response_times = self.metrics.histograms.get("query_response_time", [])
        if response_times:
            import statistics
            avg_response = statistics.mean(response_times)
            if avg_response > 5000:  # 5 seconds
                anomalies.append({
                    "type": "slow_response",
                    "severity": "medium",
                    "details": f"Average response time: {avg_response:.2f}ms"
                })
        
        return anomalies
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""
        return {
            "metrics": self.metrics.get_metrics_summary(),
            "tool_usage": self.tool_usage,
            "error_counts": self.error_counts,
            "anomalies": self.detect_anomalies(),
            "last_updated": datetime.utcnow().isoformat()
        }


# ============================================================================
# DASHBOARD EXPORT
# ============================================================================

class DashboardExporter:
    """
    Exports monitoring data for dashboards (Grafana, etc.)
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def export_grafana_json(self, output_path: str = "dashboard.json"):
        """Export Grafana dashboard JSON"""
        dashboard = {
            "dashboard": {
                "title": "QueryBot Monitoring",
                "panels": [
                    {
                        "title": "Query Volume",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(queries_total[5m])",
                            "legendFormat": "Queries/sec"
                        }]
                    },
                    {
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, rate(query_response_time_bucket[5m]))",
                            "legendFormat": "P95 Response Time"
                        }]
                    },
                    {
                        "title": "Confidence Score",
                        "type": "gauge",
                        "targets": [{
                            "expr": "avg(confidence_score)",
                            "legendFormat": "Avg Confidence"
                        }]
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(errors_total[5m])",
                            "legendFormat": "Errors/sec"
                        }]
                    },
                    {
                        "title": "Tool Usage",
                        "type": "piechart",
                        "targets": [{
                            "expr": "sum by (tool) (tool_usage_total)",
                            "legendFormat": "{{tool}}"
                        }]
                    }
                ],
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"}
            },
            "overwrite": True
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info(f"Grafana dashboard exported to {output_path}")
        return dashboard
    
    def export_health_check(self) -> Dict[str, Any]:
        """Export health check data"""
        metrics_summary = self.metrics.get_metrics_summary()
        
        # Determine overall health status
        status = "healthy"
        errors = metrics_summary["counters"].get("errors", 0)
        queries = metrics_summary["counters"].get("queries_total", 1)
        
        if queries > 0 and errors / queries > 0.3:
            status = "degraded"
        elif queries > 0 and errors / queries > 0.5:
            status = "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics_summary,
            "version": "1.0.0"
        }


# ============================================================================
# OBSERVABILITY MANAGER
# ============================================================================

class ObservabilityManager:
    """
    Unified manager for all observability components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Setup logging
        setup_logging(
            debug=self.config.get("debug", False),
            log_file=self.config.get("log_file", "logs/querybot.log")
        )
        
        # Initialize tracers
        self.langfuse_tracer = LangfuseTracer(
            public_key=self.config.get("langfuse_public_key"),
            secret_key=self.config.get("langfuse_secret_key"),
            host=self.config.get("langfuse_host")
        )
        
        self.langsmith_tracer = LangSmithTracer(
            api_key=self.config.get("langsmith_api_key"),
            project=self.config.get("langsmith_project")
        )
        
        # Initialize metrics
        self.metrics_collector = MetricsCollector()
        
        # Initialize behavior monitor
        self.behavior_monitor = AgentBehaviorMonitor(self.metrics_collector)
        
        # Initialize dashboard exporter
        self.dashboard_exporter = DashboardExporter(self.metrics_collector)
        
        logger.info("Observability manager initialized")
    
    def record_agent_execution(
        self,
        session_id: str,
        query: str,
        response: str,
        response_time_ms: float,
        confidence: float,
        tools_used: List[str],
        metadata: Dict[str, Any] = None
    ):
        """Record complete agent execution"""
        # Record metrics
        self.behavior_monitor.record_query(
            session_id=session_id,
            query_length=len(query),
            response_time_ms=response_time_ms,
            confidence_score=confidence,
            tools_used=tools_used
        )
        
        # Send traces
        self.langfuse_tracer.trace_generation(
            query=query,
            response=response,
            session_id=session_id,
            metadata=metadata
        )
        
        self.langsmith_tracer.trace_run(
            run_type="chain",
            name="agent_query",
            inputs={"query": query},
            outputs={"response": response},
            metadata={
                "session_id": session_id,
                "confidence": confidence,
                "response_time_ms": response_time_ms
            },
            tags=["political-intelligence"]
        )
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error"""
        logger.error(f"Error occurred: {error}", extra=context or {})
        
        self.metrics_collector.increment_counter(
            "errors",
            labels={
                "type": type(error).__name__,
                **((context or {}).get("labels", {}))
            }
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        return self.dashboard_exporter.export_health_check()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        return self.behavior_monitor.get_dashboard_data()
    
    def export_dashboard(self, path: str = "grafana_dashboard.json"):
        """Export Grafana dashboard"""
        return self.dashboard_exporter.export_grafana_json(path)


# Singleton instance
_observability_manager: Optional[ObservabilityManager] = None


def get_observability_manager(config: Dict[str, Any] = None) -> ObservabilityManager:
    """Get or create observability manager singleton"""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager(config)
    return _observability_manager
