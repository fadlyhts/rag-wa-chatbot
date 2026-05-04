"""Evaluation Reporting and Results Management"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import csv
import logging

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate and manage evaluation reports"""
    
    def __init__(self, results_dir: str = "evaluation_data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(
        self,
        results: Dict[str, Any],
        run_name: Optional[str] = None
    ) -> str:
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results dict
            run_name: Optional name for the run
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = run_name or f"eval_run_{timestamp}"
        
        file_path = self.results_dir / f"{run_name}.json"
        
        # Add metadata
        results_with_meta = {
            'run_name': run_name,
            'timestamp': timestamp,
            'results': results
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation results to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """Load evaluation results from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded results from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary report
        
        Args:
            results: Evaluation results
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("RAG EVALUATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Timestamp
        timestamp = results.get('timestamp', 'Unknown')
        report_lines.append(f"Timestamp: {timestamp}")
        report_lines.append("")
        
        # Dataset info
        if 'results' in results:
            results = results['results']
        
        dataset_size = results.get('dataset_size', 0)
        eval_time = results.get('evaluation_time_seconds', 0)
        report_lines.append(f"Dataset Size: {dataset_size} samples")
        report_lines.append(f"Evaluation Time: {eval_time:.2f} seconds")
        report_lines.append("")
        
        # Aggregated metrics
        report_lines.append("AGGREGATED METRICS")
        report_lines.append("-" * 70)
        
        aggregated = results.get('aggregated_metrics', {})
        
        # Group by metric
        metrics = {}
        for key, value in aggregated.items():
            if '_mean' in key:
                metric_name = key.replace('_mean', '')
                metrics[metric_name] = {'mean': value}
        
        for key, value in aggregated.items():
            for metric_name in metrics.keys():
                if key.startswith(metric_name):
                    stat = key.replace(f'{metric_name}_', '')
                    if stat != 'mean':
                        metrics[metric_name][stat] = value
        
        # Display metrics
        for metric_name, stats in metrics.items():
            report_lines.append(f"\n{metric_name.upper()}")
            report_lines.append(f"  Mean:   {stats.get('mean', 0):.4f}")
            report_lines.append(f"  Std:    {stats.get('std', 0):.4f}")
            report_lines.append(f"  Min:    {stats.get('min', 0):.4f}")
            report_lines.append(f"  Max:    {stats.get('max', 0):.4f}")
        
        # Quality assessment
        report_lines.append("")
        report_lines.append("QUALITY ASSESSMENT")
        report_lines.append("-" * 70)
        
        from app.rag.evaluation.config import evaluation_config
        
        thresholds = {
            'faithfulness': evaluation_config.min_faithfulness,
            'answer_relevancy': evaluation_config.min_answer_relevancy,
            'context_precision': evaluation_config.min_context_precision,
            'context_recall': evaluation_config.min_context_recall
        }
        
        for metric_name in metrics.keys():
            mean_score = metrics[metric_name]['mean']
            threshold = thresholds.get(metric_name, 0.7)
            status = "PASS" if mean_score >= threshold else "FAIL"
            report_lines.append(f"{metric_name}: {status} (threshold: {threshold:.2f})")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def export_to_csv(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Export per-sample results to CSV
        
        Args:
            results: Evaluation results
            output_path: Optional output path
            
        Returns:
            Path to CSV file
        """
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = str(self.results_dir / f"results_{timestamp}.csv")
        
        try:
            if 'results' in results:
                results = results['results']
            
            per_sample = results.get('per_sample_scores', [])
            
            if not per_sample:
                logger.warning("No per-sample scores to export")
                return ""
            
            # Write CSV
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=per_sample[0].keys())
                writer.writeheader()
                writer.writerows(per_sample)
            
            logger.info(f"Exported per-sample results to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def compare_runs(
        self,
        run1_path: str,
        run2_path: str
    ) -> Dict[str, Any]:
        """
        Compare two evaluation runs
        
        Args:
            run1_path: Path to first run results
            run2_path: Path to second run results
            
        Returns:
            Comparison results
        """
        try:
            run1 = self.load_results(run1_path)
            run2 = self.load_results(run2_path)
            
            results1 = run1.get('results', run1)
            results2 = run2.get('results', run2)
            
            agg1 = results1.get('aggregated_metrics', {})
            agg2 = results2.get('aggregated_metrics', {})
            
            # Calculate differences
            comparison = {
                'run1': {
                    'name': run1.get('run_name', 'Run 1'),
                    'timestamp': run1.get('timestamp', 'Unknown'),
                    'metrics': agg1
                },
                'run2': {
                    'name': run2.get('run_name', 'Run 2'),
                    'timestamp': run2.get('timestamp', 'Unknown'),
                    'metrics': agg2
                },
                'differences': {}
            }
            
            # Compare mean metrics
            for key in agg1.keys():
                if '_mean' in key and key in agg2:
                    diff = agg2[key] - agg1[key]
                    pct_change = (diff / agg1[key] * 100) if agg1[key] != 0 else 0
                    comparison['differences'][key] = {
                        'absolute_change': diff,
                        'percent_change': pct_change,
                        'improved': diff > 0
                    }
            
            logger.info(f"Compared {run1_path} vs {run2_path}")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing runs: {e}")
            raise
    
    def list_runs(self) -> List[Dict[str, str]]:
        """List all evaluation runs"""
        runs = []
        
        for file_path in sorted(self.results_dir.glob("*.json"), reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                runs.append({
                    'file_path': str(file_path),
                    'run_name': data.get('run_name', file_path.stem),
                    'timestamp': data.get('timestamp', 'Unknown')
                })
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        return runs


# Global reporter instance
evaluation_reporter = EvaluationReporter()
