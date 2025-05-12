from typing import Dict, Any, Sequence, Generator
import os
import json
import pandas as pd
from .types import EventSentence

def load_file(filename: str)-> Dict[str, Any]:
    path = os.path.abspath(os.path.join("..", "data", "raw",  "ace2005", filename))
    cwd = os.getcwd()
    with open(path) as f:
        ds: Sequence[Dict[str, Any]] = json.load(f)
    return ds

def convert_events(ds: Dict[str, Any]) -> Generator[EventSentence, None, None]:
    for sentence_dict in ds:
        for event in sentence_dict["events"]:
            yield {
                "sent_id": sentence_dict["sent_id"],
                "text": sentence_dict["text"], 
                "type": event["event_type"],
                "modality": event["event_modality"], 
                "polarity": event["event_polarity"], 
                "genericity": event["event_genericity"],
                "trigger": event["trigger"][0][0]
            }

def bootstrap_metrics(model, X_test, y_test, n_iterations=1000) -> pd.DataFrame:
    # Initialize lists to hold the metrics
    metrics_data = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for i in range(n_iterations):
        # Bootstrap resampling
        X_resample, y_resample = resample(X_test, y_test, random_state=i)
        y_pred_resample = model.predict(X_resample)
        
        # Append metrics for this resample iteration
        metrics_data['accuracy'].append(accuracy_score(y_resample, y_pred_resample))
        metrics_data['precision'].append(precision_score(y_resample, y_pred_resample))
        metrics_data['recall'].append(recall_score(y_resample, y_pred_resample))
        metrics_data['f1_score'].append(f1_score(y_resample, y_pred_resample))
    
    # Convert to a dataframe
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df
