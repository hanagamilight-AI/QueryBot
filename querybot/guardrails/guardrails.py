"""
Guardrails for QueryBot Political Intelligence AI
Provides input/output validation, action confirmation, retry/rollback logic, and rate limiting
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ValidationError
from loguru import logger
import re
from collections import defaultdict
import asyncio


# ============================================================================
# INPUT VALIDATION GUARDRAILS
# ============================================================================

class InputValidationResult(BaseModel):
    """Result of input validation"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    sanitized_input: Optional[str] = None


class InputValidator:
    """
    Validates user inputs for safety and relevance
    """
    
    # Political domain-specific blocked patterns
    SENSITIVE_PATTERNS = [
        r'\b(vote\s+buying|ballot\s+stuffing|election\s+fraud)\b',
        r'\b(threaten|intimidate|harm)\s+(voter|candidate|official)\b',
        r'\b(incite\s+violence|riot|mob\s+action)\b',
    ]
    
    # Personal information patterns (to prevent doxxing)
    PII_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN-like
        r'\b\d{10}\b',  # Phone-like
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    ]
    
    MAX_INPUT_LENGTH = 2000
    MIN_INPUT_LENGTH = 5
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._compiled_sensitive = [re.compile(p, re.IGNORECASE) for p in self.SENSITIVE_PATTERNS]
        self._compiled_pii = [re.compile(p) for p in self.PII_PATTERNS]
    
    def validate(self, user_input: str, context: Dict[str, Any] = None) -> InputValidationResult:
        """
        Validate user input
        
        Args:
            user_input: Raw user input string
            context: Additional context (session_id, user_role, etc.)
            
        Returns:
            InputValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Length checks
        if len(user_input) < self.MIN_INPUT_LENGTH:
            errors.append(f"Input too short. Minimum {self.MIN_INPUT_LENGTH} characters required.")
        
        if len(user_input) > self.MAX_INPUT_LENGTH:
            errors.append(f"Input too long. Maximum {self.MAX_INPUT_LENGTH} characters allowed.")
            return InputValidationResult(is_valid=False, errors=errors)
        
        # Check for sensitive/harmful content
        for pattern in self._compiled_sensitive:
            if pattern.search(user_input):
                if self.strict_mode:
                    errors.append("Input contains potentially harmful or sensitive content.")
                else:
                    warnings.append("Input may contain sensitive political content. Response will be carefully moderated.")
        
        # Check for PII (in strict mode)
        if self.strict_mode:
            for pattern in self._compiled_pii:
                if pattern.search(user_input):
                    errors.append("Input appears to contain personal information. Please remove PII.")
        
        # Check for SQL injection attempts
        sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b.*\b(FROM|INTO|TABLE|DATABASE)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\bOR\b\s+\d+\s*=\s*\d+)",
        ]
        for pattern in sql_injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                errors.append("Invalid input format detected.")
                break
        
        # Sanitize input (basic cleanup)
        sanitized = user_input.strip()
        sanitized = re.sub(r'\s+', ' ', sanitized)  # Normalize whitespace
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.debug(f"Input validated successfully: {sanitized[:50]}...")
        else:
            logger.warning(f"Input validation failed: {errors}")
        
        return InputValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized
        )


# ============================================================================
# OUTPUT VALIDATION GUARDRAILS
# ============================================================================

class OutputValidationResult(BaseModel):
    """Result of output validation"""
    is_valid: bool
    response: str
    errors: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    requires_review: bool = False


class OutputValidator:
    """
    Validates agent responses before returning to user
    """
    
    def __init__(self):
        self.min_confidence_threshold = 0.3
        self.max_response_length = 4000
    
    def validate(
        self, 
        response: str, 
        confidence_score: float,
        sources: List[Dict] = None,
        context: Dict[str, Any] = None
    ) -> OutputValidationResult:
        """
        Validate agent output
        
        Args:
            response: Generated response text
            confidence_score: Model confidence score
            sources: Source documents used
            context: Additional context
            
        Returns:
            OutputValidationResult
        """
        errors = []
        requires_review = False
        
        # Check response length
        if len(response) > self.max_response_length:
            errors.append("Response exceeds maximum length. Truncating...")
            response = response[:self.max_response_length] + "..."
        
        # Check confidence threshold
        if confidence_score < self.min_confidence_threshold:
            errors.append("Low confidence response - verification recommended.")
            requires_review = True
        
        # Check for hallucination indicators
        hallucination_phrases = [
            "i believe that",
            "it seems likely",
            "probably means",
            "might suggest",
        ]
        response_lower = response.lower()
        uncertain_count = sum(1 for phrase in hallucination_phrases if phrase in response_lower)
        
        if uncertain_count > 3:
            warnings_msg = "Response contains multiple uncertainty markers."
            logger.warning(warnings_msg)
        
        # Check for source attribution (important for political data)
        if sources and len(sources) == 0 and confidence_score < 0.5:
            errors.append("No sources found for low-confidence response.")
            requires_review = True
        
        # Add disclaimer for low confidence
        if confidence_score < 0.4:
            disclaimer = "**Disclaimer:** This response has lower confidence. Please verify with official sources.\n\n"
            response = disclaimer + response
        
        is_valid = len([e for e in errors if "exceeds" not in e]) == 0
        
        return OutputValidationResult(
            is_valid=is_valid,
            response=response,
            errors=errors,
            confidence_score=confidence_score,
            requires_review=requires_review
        )


# ============================================================================
# ACTION CONFIRMATION GUARDRAILS
# ============================================================================

class ActionConfirmation(BaseModel):
    """Represents an action requiring confirmation"""
    action_name: str
    action_type: str  # read, write, delete, external_api
    description: str
    parameters: Dict[str, Any]
    risk_level: str  # low, medium, high, critical
    requires_confirmation: bool


class ActionConfirmator:
    """
    Manages action confirmation workflow for sensitive operations
    """
    
    # Actions that always require confirmation
    HIGH_RISK_ACTIONS = [
        "delete_data",
        "bulk_update",
        "export_all_data",
        "modify_schema",
        "external_api_write",
    ]
    
    MEDIUM_RISK_ACTIONS = [
        "database_write",
        "batch_insert",
        "update_constituency_data",
    ]
    
    def __init__(self, auto_confirm_low_risk: bool = True):
        self.auto_confirm_low_risk = auto_confirm_low_risk
        self.pending_confirmations: Dict[str, ActionConfirmation] = {}
    
    def check_action(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        user_role: str = "standard"
    ) -> ActionConfirmation:
        """
        Check if an action requires confirmation
        
        Args:
            action_name: Name of the action
            parameters: Action parameters
            user_role: User role (admin, analyst, standard)
            
        Returns:
            ActionConfirmation object
        """
        # Determine risk level
        if action_name in self.HIGH_RISK_ACTIONS:
            risk_level = "critical"
            requires_confirmation = True
        elif action_name in self.MEDIUM_RISK_ACTIONS:
            risk_level = "high"
            requires_confirmation = True
        elif "delete" in action_name.lower():
            risk_level = "high"
            requires_confirmation = True
        elif "write" in action_name.lower() or "update" in action_name.lower():
            risk_level = "medium"
            requires_confirmation = user_role == "standard"
        else:
            risk_level = "low"
            requires_confirmation = not self.auto_confirm_low_risk
        
        # Admin bypass for non-critical actions
        if user_role == "admin" and risk_level != "critical":
            requires_confirmation = False
        
        confirmation = ActionConfirmation(
            action_name=action_name,
            action_type=self._infer_action_type(action_name),
            description=f"Execute {action_name}",
            parameters=parameters,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation
        )
        
        if requires_confirmation:
            confirmation_id = f"conf_{datetime.utcnow().timestamp()}_{action_name}"
            self.pending_confirmations[confirmation_id] = confirmation
        
        return confirmation
    
    def _infer_action_type(self, action_name: str) -> str:
        """Infer action type from name"""
        if any(word in action_name.lower() for word in ["delete", "remove"]):
            return "delete"
        elif any(word in action_name.lower() for word in ["write", "update", "insert"]):
            return "write"
        elif any(word in action_name.lower() for word in ["api", "external"]):
            return "external_api"
        else:
            return "read"
    
    def confirm_action(self, confirmation_id: str, confirmed: bool) -> bool:
        """Process action confirmation"""
        if confirmation_id not in self.pending_confirmations:
            return False
        
        confirmation = self.pending_confirmations.pop(confirmation_id)
        
        if confirmed:
            logger.info(f"Action confirmed: {confirmation.action_name}")
            return True
        else:
            logger.warning(f"Action denied: {confirmation.action_name}")
            return False


# ============================================================================
# RETRY/ROLLBACK LOGIC
# ============================================================================

class RetryConfig(BaseModel):
    """Configuration for retry behavior"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryHandler:
    """
    Handles retry logic with exponential backoff
    """
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.retry_counts: Dict[str, int] = defaultdict(int)
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        operation_id: str = None,
        retryable_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Async function to execute
            *args: Function arguments
            operation_id: Unique operation identifier
            retryable_exceptions: Tuple of exceptions that trigger retry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        import random
        
        operation_id = operation_id or f"op_{datetime.utcnow().timestamp()}"
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Reset retry count on success
                self.retry_counts[operation_id] = 0
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                self.retry_counts[operation_id] += 1
                
                if attempt >= self.config.max_retries:
                    logger.error(f"Operation {operation_id} failed after {attempt + 1} attempts: {e}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                
                if self.config.jitter:
                    delay *= (0.5 + random.random() * 0.5)
                
                logger.warning(
                    f"Operation {operation_id} failed (attempt {attempt + 1}/{self.config.max_retries}). "
                    f"Retrying in {delay:.2f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def get_retry_count(self, operation_id: str) -> int:
        """Get current retry count for operation"""
        return self.retry_counts.get(operation_id, 0)


class RollbackManager:
    """
    Manages rollback operations for failed transactions
    """
    
    def __init__(self):
        self.compensation_actions: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_compensation(self, transaction_id: str, compensation_func: Callable):
        """Register a compensation (rollback) function"""
        self.compensation_actions[transaction_id].append(compensation_func)
        logger.debug(f"Registered compensation action for {transaction_id}")
    
    async def execute_rollback(self, transaction_id: str) -> bool:
        """
        Execute all compensation actions for a transaction
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            True if rollback successful, False otherwise
        """
        if transaction_id not in self.compensation_actions:
            logger.warning(f"No compensation actions registered for {transaction_id}")
            return False
        
        success = True
        for compensation_func in reversed(self.compensation_actions[transaction_id]):
            try:
                if asyncio.iscoroutinefunction(compensation_func):
                    await compensation_func()
                else:
                    compensation_func()
                logger.info(f"Compensation action executed for {transaction_id}")
            except Exception as e:
                logger.error(f"Compensation action failed for {transaction_id}: {e}")
                success = False
        
        # Clear compensation actions after execution
        del self.compensation_actions[transaction_id]
        
        return success
    
    def clear_transaction(self, transaction_id: str):
        """Clear compensation actions after successful completion"""
        if transaction_id in self.compensation_actions:
            del self.compensation_actions[transaction_id]
            logger.debug(f"Cleared compensation actions for {transaction_id}")


# ============================================================================
# RATE LIMITING GUARDRAILS
# ============================================================================

class RateLimiter:
    """
    Rate limiter for autonomous agent actions
    """
    
    def __init__(self):
        # Rate limits per action type (actions per minute)
        self.limits = {
            "database_read": 100,
            "database_write": 20,
            "vector_search": 60,
            "external_api": 30,
            "conversation_memory": 50,
            "default": 40,
        }
        
        # Track action counts per session
        self.action_counts: Dict[str, Dict[str, List[datetime]]] = defaultdict(lambda: defaultdict(list))
        self.lock = asyncio.Lock()
    
    def set_limit(self, action_type: str, limit_per_minute: int):
        """Set rate limit for action type"""
        self.limits[action_type] = limit_per_minute
    
    async def check_rate_limit(
        self, 
        session_id: str, 
        action_type: str
    ) -> tuple:
        """
        Check if action is within rate limit
        
        Args:
            session_id: Session identifier
            action_type: Type of action
            
        Returns:
            Tuple of (is_allowed, wait_time_seconds)
        """
        async with self.lock:
            now = datetime.utcnow()
            one_minute_ago = now - timedelta(minutes=1)
            
            # Clean old entries
            self.action_counts[session_id][action_type] = [
                ts for ts in self.action_counts[session_id][action_type]
                if ts > one_minute_ago
            ]
            
            # Get limit for this action type
            limit = self.limits.get(action_type, self.limits["default"])
            
            # Check if under limit
            current_count = len(self.action_counts[session_id][action_type])
            
            if current_count < limit:
                self.action_counts[session_id][action_type].append(now)
                return True, None
            else:
                # Calculate wait time
                oldest_ts = min(self.action_counts[session_id][action_type])
                wait_time = (oldest_ts + timedelta(minutes=1) - now).total_seconds()
                return False, max(0, wait_time)
    
    def get_usage_stats(self, session_id: str) -> Dict[str, int]:
        """Get current usage statistics for session"""
        stats = {}
        for action_type, timestamps in self.action_counts[session_id].items():
            stats[action_type] = len(timestamps)
        return stats


# ============================================================================
# POLITICAL DOMAIN-SPECIFIC GUARDRAILS
# ============================================================================

class PoliticalDomainGuardrails:
    """
    Domain-specific guardrails for political intelligence
    """
    
    # Constituencies that might require additional verification
    SENSITIVE_CONSTITUENCIES = [
        # Add constituency names that need extra care
    ]
    
    # Topics requiring careful handling
    SENSITIVE_TOPICS = [
        "election results",
        "voter demographics",
        "campaign finance",
        "polling data",
        "religious voting patterns",
        "caste-based analysis",
    ]
    
    def __init__(self):
        self.input_validator = InputValidator(strict_mode=True)
        self.output_validator = OutputValidator()
    
    def validate_query_context(
        self,
        query: str,
        entities: Dict[str, str],
        intent: str
    ) -> Dict[str, Any]:
        """
        Validate query in political context
        
        Args:
            query: User query
            entities: Extracted entities (constituency, party, etc.)
            intent: Query intent classification
            
        Returns:
            Validation result dictionary
        """
        result = {
            "is_safe": True,
            "warnings": [],
            "restrictions": [],
            "requires_disclaimer": False
        }
        
        # Check for sensitive constituency
        constituency = entities.get("constituency", "")
        if constituency in self.SENSITIVE_CONSTITUENCIES:
            result["warnings"].append(f"Query involves sensitive constituency: {constituency}")
            result["requires_disclaimer"] = True
        
        # Check for sensitive topics
        query_lower = query.lower()
        for topic in self.SENSITIVE_TOPICS:
            if topic in query_lower:
                result["warnings"].append(f"Query involves sensitive topic: {topic}")
                result["requires_disclaimer"] = True
                break
        
        # Check intent for potentially problematic queries
        if intent == "predictive":
            result["warnings"].append("Predictive queries should include uncertainty disclaimers")
            result["restrictions"].append("no_definitive_predictions")
        
        # Demographic analysis restrictions
        if any(term in query_lower for term in ["religion", "caste", "community"]):
            result["restrictions"].append("aggregate_data_only")
            result["warnings"].append("Demographic analysis limited to aggregate data")
        
        # Election result restrictions (before official declaration)
        if "result" in query_lower or "winner" in query_lower:
            result["restrictions"].append("official_sources_only")
            result["warnings"].append("Election results from official sources only")
        
        is_safe = len([w for w in result["warnings"] if "unsafe" in w.lower()]) == 0
        result["is_safe"] = is_safe
        
        return result
    
    def apply_response_restrictions(
        self,
        response: str,
        restrictions: List[str]
    ) -> str:
        """Apply restrictions to response"""
        modified_response = response
        
        if "no_definitive_predictions" in restrictions:
            # Add uncertainty language
            modified_response = "**Note:** Predictions are based on available data and carry inherent uncertainty.\n\n" + modified_response
        
        if "aggregate_data_only" in restrictions:
            # Ensure no individual-level data is shown
            modified_response += "\n\n*Data shown is aggregated and anonymized.*"
        
        if "official_sources_only" in restrictions:
            modified_response += "\n\n*Please verify with official election commission sources.*"
        
        return modified_response


# ============================================================================
# UNIFIED GUARDRAIL MANAGER
# ============================================================================

class GuardrailManager:
    """
    Unified manager for all guardrails
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.input_validator = InputValidator(strict_mode=self.config.get("strict_mode", False))
        self.output_validator = OutputValidator()
        self.action_confirmator = ActionConfirmator(
            auto_confirm_low_risk=self.config.get("auto_confirm_low_risk", True)
        )
        self.retry_handler = RetryHandler()
        self.rollback_manager = RollbackManager()
        self.rate_limiter = RateLimiter()
        self.domain_guardrails = PoliticalDomainGuardrails()
    
    async def validate_input(self, user_input: str, context: Dict[str, Any] = None) -> InputValidationResult:
        """Validate user input"""
        return self.input_validator.validate(user_input, context)
    
    def validate_output(
        self, 
        response: str, 
        confidence: float, 
        sources: List[Dict] = None
    ) -> OutputValidationResult:
        """Validate agent output"""
        return self.output_validator.validate(response, confidence, sources)
    
    def check_action(self, action_name: str, parameters: Dict, user_role: str = "standard") -> ActionConfirmation:
        """Check if action requires confirmation"""
        return self.action_confirmator.check_action(action_name, parameters, user_role)
    
    async def execute_with_retry(self, func: Callable, *args, operation_id: str = None, **kwargs) -> Any:
        """Execute function with retry logic"""
        return await self.retry_handler.execute_with_retry(func, *args, operation_id=operation_id, **kwargs)
    
    async def check_rate_limit(self, session_id: str, action_type: str) -> tuple:
        """Check rate limit for action"""
        return await self.rate_limiter.check_rate_limit(session_id, action_type)
    
    def validate_political_context(
        self,
        query: str,
        entities: Dict[str, str],
        intent: str
    ) -> Dict[str, Any]:
        """Validate query in political context"""
        return self.domain_guardrails.validate_query_context(query, entities, intent)


# Singleton instance
_guardrail_manager: Optional[GuardrailManager] = None


def get_guardrail_manager(config: Dict[str, Any] = None) -> GuardrailManager:
    """Get or create guardrail manager singleton"""
    global _guardrail_manager
    if _guardrail_manager is None:
        _guardrail_manager = GuardrailManager(config)
    return _guardrail_manager
