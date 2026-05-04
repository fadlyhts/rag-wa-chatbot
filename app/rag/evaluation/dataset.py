"""Evaluation Dataset Management"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv
from datetime import datetime
import logging
from datasets import Dataset

logger = logging.getLogger(__name__)


class EvaluationDataset:
    """Manage evaluation datasets for RAGAS"""
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.data: List[Dict[str, Any]] = []
    
    def add_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a single evaluation sample
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Expected/correct answer (optional)
            metadata: Additional metadata (user_id, timestamp, etc.)
        """
        sample = {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth,
            'metadata': metadata or {}
        }
        self.data.append(sample)
        logger.debug(f"Added sample: {question[:50]}...")
    
    def load_from_json(self, file_path: str):
        """Load dataset from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Loaded {len(self.data)} samples from {file_path}")
        except Exception as e:
            logger.error(f"Error loading JSON dataset: {e}")
            raise
    
    def load_from_csv(self, file_path: str):
        """Load dataset from CSV file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse contexts (stored as JSON string in CSV)
                    contexts = json.loads(row.get('contexts', '[]'))
                    self.add_sample(
                        question=row['question'],
                        answer=row['answer'],
                        contexts=contexts,
                        ground_truth=row.get('ground_truth'),
                        metadata=json.loads(row.get('metadata', '{}'))
                    )
            logger.info(f"Loaded {len(self.data)} samples from {file_path}")
        except Exception as e:
            logger.error(f"Error loading CSV dataset: {e}")
            raise
    
    def save_to_json(self, file_path: Optional[str] = None):
        """Save dataset to JSON file"""
        file_path = file_path or self.dataset_path
        if not file_path:
            file_path = f"evaluation_data/test_sets/dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.data)} samples to {file_path}")
        except Exception as e:
            logger.error(f"Error saving JSON dataset: {e}")
            raise
    
    def save_to_csv(self, file_path: Optional[str] = None):
        """Save dataset to CSV file"""
        file_path = file_path or self.dataset_path
        if not file_path:
            file_path = f"evaluation_data/test_sets/dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                if not self.data:
                    logger.warning("No data to save")
                    return
                
                fieldnames = ['question', 'answer', 'contexts', 'ground_truth', 'metadata']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for sample in self.data:
                    row = {
                        'question': sample['question'],
                        'answer': sample['answer'],
                        'contexts': json.dumps(sample['contexts']),
                        'ground_truth': sample.get('ground_truth', ''),
                        'metadata': json.dumps(sample.get('metadata', {}))
                    }
                    writer.writerow(row)
            
            logger.info(f"Saved {len(self.data)} samples to {file_path}")
        except Exception as e:
            logger.error(f"Error saving CSV dataset: {e}")
            raise
    
    def to_ragas_dataset(self) -> Dataset:
        """
        Convert to RAGAS-compatible HuggingFace Dataset
        
        Returns:
            HuggingFace Dataset object
        """
        if not self.data:
            logger.warning("Empty dataset")
            return Dataset.from_dict({
                'question': [],
                'answer': [],
                'contexts': [],
                'ground_truth': []
            })
        
        # Format data for RAGAS
        formatted_data = {
            'question': [sample['question'] for sample in self.data],
            'answer': [sample['answer'] for sample in self.data],
            'contexts': [sample['contexts'] for sample in self.data],
            'ground_truth': [sample.get('ground_truth', '') for sample in self.data]
        }
        
        dataset = Dataset.from_dict(formatted_data)
        logger.info(f"Created RAGAS dataset with {len(dataset)} samples")
        return dataset
    
    def from_production_logs(
        self,
        conversations: List[Dict[str, Any]],
        min_score: float = 0.7
    ):
        """
        Build dataset from production conversation logs
        
        Args:
            conversations: List of conversation dicts with query/response/sources
            min_score: Minimum retrieval score to include
        """
        for conv in conversations:
            if not conv.get('sources') or not conv.get('query') or not conv.get('response'):
                continue
            
            # Filter contexts by score
            contexts = [
                src['content']
                for src in conv['sources']
                if src.get('score', 0) >= min_score
            ]
            
            if contexts:
                self.add_sample(
                    question=conv['query'],
                    answer=conv['response'],
                    contexts=contexts,
                    ground_truth=conv.get('ground_truth'),
                    metadata={
                        'user_id': conv.get('user_id'),
                        'timestamp': conv.get('timestamp'),
                        'retrieval_scores': [src.get('score') for src in conv['sources']]
                    }
                )
        
        logger.info(f"Created dataset from {len(self.data)} production conversations")
    
    def filter_by_metadata(self, key: str, value: Any) -> 'EvaluationDataset':
        """Filter dataset by metadata field"""
        filtered = EvaluationDataset()
        filtered.data = [
            sample for sample in self.data
            if sample.get('metadata', {}).get(key) == value
        ]
        logger.info(f"Filtered dataset: {len(filtered.data)}/{len(self.data)} samples")
        return filtered
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
