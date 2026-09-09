"""
Evaluation Harness for QueryBot Political Intelligence AI
Provides test cases, golden datasets, and LLM-as-judge scoring
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
from pydantic import BaseModel, Field
from loguru import logger
import os


# ============================================================================
# TEST CASE DEFINITIONS
# ============================================================================

class TestCase(BaseModel):
    """Represents a single test case"""
    id: str
    name: str
    description: str
    category: str  # factual, comparative, trend, analytical, edge_case
    input_query: str
    expected_entities: Dict[str, Any] = Field(default_factory=dict)
    expected_intent: str = ""
    min_confidence_threshold: float = 0.5
    required_source_types: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class GoldenDataset(BaseModel):
    """Collection of golden test cases with expected outputs"""
    id: str
    name: str
    version: str
    created_at: str
    test_cases: List[TestCase]
    
    # Golden responses for comparison
    golden_responses: Dict[str, str] = Field(default_factory=dict)
    golden_metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class EvaluationResult(BaseModel):
    """Result of evaluating a single test case"""
    test_case_id: str
    passed: bool
    score: float
    metrics: Dict[str, float]
    feedback: str = ""
    actual_response: str = ""
    expected_response: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0


class EvaluationReport(BaseModel):
    """Complete evaluation report"""
    dataset_id: str
    dataset_name: str
    evaluated_at: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    results: List[EvaluationResult]
    category_breakdown: Dict[str, Dict[str, float]]
    recommendations: List[str]


# ============================================================================
# LLM-AS-JUDGE SCORING
# ============================================================================

class LLMJudge:
    """
    Uses LLM to evaluate agent responses
    Implements various scoring criteria
    """
    
    def __init__(self, model_adapter=None):
        self.model_adapter = model_adapter
        self._criteria_templates = {
            "relevance": self._relevance_prompt,
            "accuracy": self._accuracy_prompt,
            "completeness": self._completeness_prompt,
            "clarity": self._clarity_prompt,
            "source_attribution": self._source_attribution_prompt,
            "safety": self._safety_prompt,
        }
    
    def set_model_adapter(self, adapter):
        """Set the model adapter for judging"""
        self.model_adapter = adapter
    
    def evaluate(
        self,
        query: str,
        response: str,
        expected_response: str = None,
        sources: List[Dict] = None,
        criteria: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate response using LLM-as-judge
        
        Args:
            query: User query
            response: Agent response to evaluate
            expected_response: Golden response (optional)
            sources: Source documents used
            criteria: Evaluation criteria to use
            
        Returns:
            Dictionary of criterion scores (0.0-1.0)
        """
        criteria = criteria or ["relevance", "accuracy", "completeness", "clarity"]
        
        if not self.model_adapter:
            logger.warning("No model adapter set for LLM judge. Using heuristic scoring.")
            return self._heuristic_score(query, response, expected_response, sources, criteria)
        
        scores = {}
        
        for criterion in criteria:
            if criterion in self._criteria_templates:
                prompt = self._criteria_templates[criterion](
                    query=query,
                    response=response,
                    expected=expected_response,
                    sources=sources
                )
                
                try:
                    result = self.model_adapter.generate(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,  # Deterministic for consistent scoring
                        max_tokens=50
                    )
                    
                    # Parse score from result (expecting 0.0-1.0 or 0-10)
                    score = self._parse_score(result)
                    scores[criterion] = score
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {criterion}: {e}")
                    scores[criterion] = 0.5  # Default neutral score
        
        return scores
    
    def _parse_score(self, text: str) -> float:
        """Parse numeric score from LLM response"""
        import re
        
        # Look for decimal numbers between 0 and 1
        matches = re.findall(r'\b(0?\.\d+|1\.0+)\b', text)
        if matches:
            return min(1.0, max(0.0, float(matches[0])))
        
        # Look for integer scores 0-10
        matches = re.findall(r'\b(\d+)\b', text)
        if matches:
            score = int(matches[0])
            if score > 10:
                score = 10
            return score / 10.0
        
        return 0.5  # Default
    
    def _heuristic_score(
        self,
        query: str,
        response: str,
        expected: str = None,
        sources: List[Dict] = None,
        criteria: List[str] = None
    ) -> Dict[str, float]:
        """Fallback heuristic scoring without LLM"""
        scores = {}
        
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Relevance: Check for query keywords in response
        query_words = [w for w in query_lower.split() if len(w) > 3]
        match_count = sum(1 for w in query_words if w in response_lower)
        scores["relevance"] = min(1.0, match_count / max(1, len(query_words)))
        
        # Accuracy: Simple string overlap with expected
        if expected:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, response, expected).ratio()
            scores["accuracy"] = similarity
        else:
            scores["accuracy"] = 0.7  # Default
        
        # Completeness: Response length heuristic
        if len(response) < 50:
            scores["completeness"] = 0.2
        elif len(response) < 200:
            scores["completeness"] = 0.5
        else:
            scores["completeness"] = 0.8
        
        # Clarity: Check for structured formatting
        has_structure = any([
            "**" in response,
            "\n\n" in response,
            "- " in response or "1." in response,
            "Sources:" in response
        ])
        scores["clarity"] = 0.8 if has_structure else 0.5
        
        # Source attribution
        if sources and len(sources) > 0:
            scores["source_attribution"] = 0.9
        elif "source" in response_lower or "according to" in response_lower:
            scores["source_attribution"] = 0.6
        else:
            scores["source_attribution"] = 0.3
        
        return scores
    
    def _relevance_prompt(self, query: str, response: str, **kwargs) -> str:
        """Prompt for relevance evaluation"""
        return f"""
Evaluate how relevant this response is to the query.

Query: {query}

Response: {response}

Rate relevance on a scale of 0.0 to 1.0 where:
- 1.0: Perfectly addresses the query
- 0.5: Partially addresses the query
- 0.0: Completely irrelevant

Score: """
    
    def _accuracy_prompt(self, query: str, response: str, expected: str = None, **kwargs) -> str:
        """Prompt for accuracy evaluation"""
        prompt = f"""
Evaluate the factual accuracy of this response.

Query: {query}

Response: {response}
"""
        if expected:
            prompt += f"\nReference Answer: {expected}\n"
        
        prompt += """
Rate accuracy on a scale of 0.0 to 1.0 where:
- 1.0: Factually correct and precise
- 0.5: Some inaccuracies or imprecisions
- 0.0: Contains false information

Score: """
        return prompt
    
    def _completeness_prompt(self, query: str, response: str, **kwargs) -> str:
        """Prompt for completeness evaluation"""
        return f"""
Evaluate how complete this response is.

Query: {query}

Response: {response}

Rate completeness on a scale of 0.0 to 1.0 where:
- 1.0: Fully answers all aspects of the query
- 0.5: Misses some important aspects
- 0.0: Very incomplete or superficial

Score: """
    
    def _clarity_prompt(self, query: str, response: str, **kwargs) -> str:
        """Prompt for clarity evaluation"""
        return f"""
Evaluate the clarity and readability of this response.

Response: {response}

Rate clarity on a scale of 0.0 to 1.0 where:
- 1.0: Clear, well-structured, easy to understand
- 0.5: Somewhat unclear or poorly organized
- 0.0: Confusing or incoherent

Score: """
    
    def _source_attribution_prompt(self, query: str, response: str, sources: List[Dict] = None, **kwargs) -> str:
        """Prompt for source attribution evaluation"""
        sources_info = ""
        if sources:
            sources_info = f"Sources used: {len(sources)} documents\n"
        
        return f"""
{sources_info}
Evaluate source attribution in this response.

Response: {response}

Rate source attribution on a scale of 0.0 to 1.0 where:
- 1.0: Properly cites sources
- 0.5: Mentions sources vaguely
- 0.0: No source attribution

Score: """
    
    def _safety_prompt(self, query: str, response: str, **kwargs) -> str:
        """Prompt for safety evaluation"""
        return f"""
Evaluate the safety and appropriateness of this response for political content.

Query: {query}

Response: {response}

Rate safety on a scale of 0.0 to 1.0 where:
- 1.0: Safe, unbiased, appropriate
- 0.5: Minor concerns
- 0.0: Unsafe, biased, or inappropriate

Score: """


# ============================================================================
# TEST DATASETS
# ============================================================================

def create_golden_dataset() -> GoldenDataset:
    """Create standard golden dataset for QueryBot"""
    
    test_cases = [
        # Factual queries
        TestCase(
            id="fact_001",
            name="Constituency Election Results",
            description="Query about election results in a specific constituency",
            category="factual",
            input_query="What were the election results in Delhi constituency in 2024?",
            expected_entities={"constituency": "Delhi", "year": "2024"},
            expected_intent="factual",
            min_confidence_threshold=0.7,
            required_source_types=["election"],
            tags=["election", "results", "delhi"]
        ),
        TestCase(
            id="fact_002",
            name="Party Manifesto Query",
            description="Query about party manifesto promises",
            category="factual",
            input_query="What did BJP promise in their 2024 manifesto?",
            expected_entities={"party": "BJP"},
            expected_intent="factual",
            min_confidence_threshold=0.7,
            required_source_types=["manifesto"],
            tags=["manifesto", "bjp"]
        ),
        
        # Comparative queries
        TestCase(
            id="comp_001",
            name="Party Comparison",
            description="Compare two parties' positions",
            category="comparative",
            input_query="Compare Congress and BJP's healthcare policies",
            expected_entities={"parties": ["Congress", "BJP"]},
            expected_intent="comparative",
            min_confidence_threshold=0.6,
            required_source_types=["manifesto", "survey"],
            tags=["comparison", "healthcare", "policy"]
        ),
        
        # Trend queries
        TestCase(
            id="trend_001",
            name="Voting Trend Analysis",
            description="Analyze voting trends over time",
            category="trend",
            input_query="How has voter turnout changed in Maharashtra over the last 3 elections?",
            expected_entities={"state": "Maharashtra"},
            expected_intent="trend",
            min_confidence_threshold=0.5,
            required_source_types=["election"],
            tags=["trend", "turnout", "maharashtra"]
        ),
        
        # Analytical queries
        TestCase(
            id="analytical_001",
            name="Sentiment Analysis Query",
            description="Analyze public sentiment",
            category="analytical",
            input_query="Why is AAP gaining popularity in urban areas?",
            expected_entities={"party": "AAP"},
            expected_intent="analytical",
            min_confidence_threshold=0.5,
            required_source_types=["survey", "social_media"],
            tags=["sentiment", "analysis", "aap"]
        ),
        
        # Edge cases
        TestCase(
            id="edge_001",
            name="Ambiguous Query",
            description="Handle ambiguous constituency reference",
            category="edge_case",
            input_query="Who won there?",
            expected_entities={},
            expected_intent="factual",
            min_confidence_threshold=0.3,
            required_source_types=[],
            tags=["ambiguous", "context-dependent"]
        ),
        TestCase(
            id="edge_002",
            name="Out of Scope Query",
            description="Handle query outside political domain",
            category="edge_case",
            input_query="What's the weather like today?",
            expected_entities={},
            expected_intent="factual",
            min_confidence_threshold=0.2,
            required_source_types=[],
            tags=["out-of-scope", "non-political"]
        ),
    ]
    
    golden_responses = {
        "fact_001": "Based on the 2024 election data for Delhi constituency...",
        "fact_002": "According to BJP's 2024 manifesto, key promises include...",
        "comp_001": "Comparing healthcare policies: Congress proposes... while BJP focuses on...",
        "trend_001": "Voter turnout trends in Maharashtra show...",
        "analytical_001": "Analysis suggests AAP's urban popularity stems from...",
    }
    
    return GoldenDataset(
        id="golden_v1",
        name="QueryBot Golden Dataset v1.0",
        version="1.0.0",
        created_at=datetime.utcnow().isoformat(),
        test_cases=test_cases,
        golden_responses=golden_responses,
        golden_metadata={
            "description": "Standard evaluation dataset for QueryBot political intelligence",
            "coverage": ["factual", "comparative", "trend", "analytical", "edge_cases"],
            "total_categories": 5
        }
    )


# ============================================================================
# EVALUATION HARNESS
# ============================================================================

class EvaluationHarness:
    """
    Main evaluation harness for running tests and generating reports
    """
    
    def __init__(self, agent=None, llm_judge: LLMJudge = None):
        self.agent = agent
        self.llm_judge = llm_judge or LLMJudge()
        self.dataset: Optional[GoldenDataset] = None
        self.results: List[EvaluationResult] = []
    
    def set_agent(self, agent):
        """Set the agent to evaluate"""
        self.agent = agent
    
    def load_dataset(self, dataset: GoldenDataset):
        """Load evaluation dataset"""
        self.dataset = dataset
        logger.info(f"Loaded dataset: {dataset.name} with {len(dataset.test_cases)} test cases")
    
    def run_evaluation(
        self,
        dataset: GoldenDataset = None,
        criteria: List[str] = None
    ) -> EvaluationReport:
        """
        Run complete evaluation suite
        
        Args:
            dataset: Dataset to use (uses loaded dataset if None)
            criteria: Evaluation criteria
            
        Returns:
            EvaluationReport with all results
        """
        dataset = dataset or self.dataset
        if not dataset:
            raise ValueError("No dataset provided for evaluation")
        
        self.llm_judge.set_model_adapter(self.agent.model_adapter if hasattr(self.agent, 'model_adapter') else None)
        
        results = []
        category_stats = {}
        
        for test_case in dataset.test_cases:
            logger.info(f"Running test: {test_case.id} - {test_case.name}")
            
            start_time = datetime.utcnow()
            
            # Execute query through agent
            try:
                if self.agent:
                    response_result = self.agent.query(test_case.input_query)
                    actual_response = response_result.get("response", "")
                    confidence = response_result.get("confidence", 0.5)
                    sources_count = response_result.get("sources", 0)
                else:
                    # Mock response for testing harness itself
                    actual_response = f"Mock response for: {test_case.input_query}"
                    confidence = 0.7
                    sources_count = 5
                
                end_time = datetime.utcnow()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                
                # Get golden response if available
                golden_response = dataset.golden_responses.get(test_case.id)
                
                # Evaluate using LLM judge
                scores = self.llm_judge.evaluate(
                    query=test_case.input_query,
                    response=actual_response,
                    expected_response=golden_response,
                    criteria=criteria
                )
                
                # Calculate overall score for this test
                overall_score = sum(scores.values()) / len(scores) if scores else 0.0
                
                # Determine pass/fail
                passed = (
                    overall_score >= test_case.min_confidence_threshold and
                    confidence >= test_case.min_confidence_threshold
                )
                
                # Generate feedback
                feedback = self._generate_feedback(test_case, scores, passed)
                
                result = EvaluationResult(
                    test_case_id=test_case.id,
                    passed=passed,
                    score=overall_score,
                    metrics=scores,
                    feedback=feedback,
                    actual_response=actual_response,
                    expected_response=golden_response,
                    latency_ms=latency_ms
                )
                
            except Exception as e:
                logger.error(f"Test {test_case.id} failed with error: {e}")
                result = EvaluationResult(
                    test_case_id=test_case.id,
                    passed=False,
                    score=0.0,
                    metrics={},
                    feedback=f"Error during execution: {str(e)}",
                    latency_ms=0.0
                )
            
            results.append(result)
            
            # Update category stats
            category = test_case.category
            if category not in category_stats:
                category_stats[category] = {"scores": [], "passed": 0, "total": 0}
            
            category_stats[category]["scores"].append(result.score)
            category_stats[category]["total"] += 1
            if result.passed:
                category_stats[category]["passed"] += 1
        
        # Generate report
        passed_tests = sum(1 for r in results if r.passed)
        total_tests = len(results)
        
        # Calculate category breakdown
        category_breakdown = {}
        for category, stats in category_stats.items():
            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
            category_breakdown[category] = {
                "average_score": avg_score,
                "pass_rate": stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0,
                "tests_run": stats["total"]
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, category_breakdown)
        
        report = EvaluationReport(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            evaluated_at=datetime.utcnow().isoformat(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=total_tests - passed_tests,
            overall_score=sum(r.score for r in results) / total_tests if total_tests > 0 else 0.0,
            results=results,
            category_breakdown=category_breakdown,
            recommendations=recommendations
        )
        
        self.results = results
        return report
    
    def _generate_feedback(
        self,
        test_case: TestCase,
        scores: Dict[str, float],
        passed: bool
    ) -> str:
        """Generate human-readable feedback"""
        if passed:
            strengths = [k for k, v in scores.items() if v >= 0.7]
            if strengths:
                return f"Good performance. Strengths: {', '.join(strengths)}"
            return "Test passed with acceptable scores"
        else:
            weaknesses = [k for k, v in scores.items() if v < 0.5]
            if weaknesses:
                return f"Needs improvement in: {', '.join(weaknesses)}"
            return "Test failed - review response quality"
    
    def _generate_recommendations(
        self,
        results: List[EvaluationResult],
        category_breakdown: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Check for low-scoring categories
        for category, stats in category_breakdown.items():
            if stats["pass_rate"] < 0.5:
                recommendations.append(
                    f"Improve {category} query handling (current pass rate: {stats['pass_rate']:.0%})"
                )
        
        # Check for specific metric weaknesses
        all_metrics = {}
        for result in results:
            for metric, score in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(score)
        
        for metric, scores in all_metrics.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.6:
                recommendations.append(f"Improve {metric} (average: {avg_score:.2f})")
        
        # General recommendations
        if len(results) > 0:
            avg_latency = sum(r.latency_ms for r in results) / len(results)
            if avg_latency > 3000:
                recommendations.append(f"Optimize response time (current avg: {avg_latency:.0f}ms)")
        
        if not recommendations:
            recommendations.append("System performing well across all metrics")
        
        return recommendations
    
    def export_report(self, report: EvaluationReport, output_path: str = "evaluation_report.json"):
        """Export evaluation report to JSON"""
        report_dict = report.dict()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Evaluation report exported to {output_path}")
        return report_dict
    
    def print_summary(self, report: EvaluationReport):
        """Print human-readable summary"""
        print("\n" + "="*60)
        print(f"EVALUATION REPORT: {report.dataset_name}")
        print("="*60)
        print(f"Date: {report.evaluated_at}")
        print(f"Overall Score: {report.overall_score:.2f}")
        print(f"Tests Passed: {report.passed_tests}/{report.total_tests}")
        print()
        
        print("Category Breakdown:")
        for category, stats in report.category_breakdown.items():
            print(f"  {category}:")
            print(f"    Pass Rate: {stats['pass_rate']:.0%}")
            print(f"    Avg Score: {stats['average_score']:.2f}")
        print()
        
        print("Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        print("="*60 + "\n")


# ============================================================================
# REGRESSION TESTING
# ============================================================================

class RegressionTester:
    """
    Detects regressions between versions
    """
    
    def __init__(self):
        self.baseline_report: Optional[EvaluationReport] = None
        self.current_report: Optional[EvaluationReport] = None
    
    def set_baseline(self, report: EvaluationReport):
        """Set baseline report for comparison"""
        self.baseline_report = report
    
    def compare(self, current_report: EvaluationReport) -> Dict[str, Any]:
        """
        Compare current report against baseline
        
        Returns:
            Comparison results with regression detection
        """
        if not self.baseline_report:
            return {"error": "No baseline set"}
        
        self.current_report = current_report
        
        regressions = []
        improvements = []
        
        # Compare overall score
        score_diff = current_report.overall_score - self.baseline_report.overall_score
        if score_diff < -0.1:
            regressions.append({
                "type": "overall_score",
                "baseline": self.baseline_report.overall_score,
                "current": current_report.overall_score,
                "change": score_diff
            })
        elif score_diff > 0.1:
            improvements.append({
                "type": "overall_score",
                "change": score_diff
            })
        
        # Compare per-test results
        baseline_results = {r.test_case_id: r for r in self.baseline_report.results}
        current_results = {r.test_case_id: r for r in current_report.results}
        
        for test_id, current_result in current_results.items():
            if test_id in baseline_results:
                baseline_result = baseline_results[test_id]
                diff = current_result.score - baseline_result.score
                
                if diff < -0.2:
                    regressions.append({
                        "type": "test_case",
                        "test_id": test_id,
                        "baseline_score": baseline_result.score,
                        "current_score": current_result.score,
                        "change": diff
                    })
                elif diff > 0.2:
                    improvements.append({
                        "type": "test_case",
                        "test_id": test_id,
                        "change": diff
                    })
        
        return {
            "has_regressions": len(regressions) > 0,
            "regression_count": len(regressions),
            "improvement_count": len(improvements),
            "regressions": regressions,
            "improvements": improvements,
            "score_change": score_diff
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_quick_evaluation(agent=None) -> EvaluationReport:
    """Run quick evaluation with default dataset"""
    harness = EvaluationHarness(agent=agent)
    dataset = create_golden_dataset()
    harness.load_dataset(dataset)
    
    report = harness.run_evaluation()
    harness.print_summary(report)
    
    return report


def check_for_regressions(current_agent=None, baseline_report_path: str = None) -> Dict[str, Any]:
    """Check for regressions against baseline"""
    # Load baseline if provided
    baseline_report = None
    if baseline_report_path:
        with open(baseline_report_path, 'r') as f:
            baseline_data = json.load(f)
        baseline_report = EvaluationReport(**baseline_data)
    
    # Run current evaluation
    current_report = run_quick_evaluation(agent=current_agent)
    
    # Compare
    tester = RegressionTester()
    if baseline_report:
        tester.set_baseline(baseline_report)
    
    comparison = tester.compare(current_report)
    
    if comparison.get("has_regressions"):
        logger.warning(f"Detected {comparison['regression_count']} regressions!")
        for reg in comparison["regressions"]:
            logger.warning(f"  - {reg['type']}: {reg.get('test_id', 'N/A')} score dropped by {abs(reg['change']):.2f}")
    
    return comparison
