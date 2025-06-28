from diveye_utils import DivEyeUtils
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import xgboost as xgb
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

class DivEye:
    def __init__(self, model_name, train_dataset, test_dataset, logging=True):
        self.model = AutoModelForCausalLM.from_pretrained(model_name) # Recommended to be GPT-2 (for ease compute)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.utils = DivEyeUtils(self.model, self.tokenizer)
        self.train_dataset = train_dataset # Expected to be in a CSV format (columns must be in the format ['text', 'label'])
        self.test_dataset = test_dataset # Expected to be in a CSV format (columns must be in the format ['text', 'label'])
        
        self.logging = logging
        
    def _log(self, results, output_path="results.json"):
        # Convert all NumPy data types to native Python types
        def convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, (np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.int32, np.int64)):
                return int(o)
            elif isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [convert(i) for i in o]
            else:
                return o
    
        cleaned_results = [convert(item) for item in results]
    
        with open(output_path, "w") as f:
            json.dump(cleaned_results, f, indent=2)
    
        print(f"[LOGS] Results saved to {output_path}")
    
    def train(self):
        # Temporary variables
        X_train = []
        Y_train = []
        X_train_log = []
        
        # Parse CSV in a loop
        train_dict = pd.read_csv(self.train_dataset).to_dict(orient='records')
        
        for i, sample in enumerate(tqdm(train_dict)):
            features = self.utils.diveye_compute(sample['text'])
            if self.logging:
                X_train_log.append({'text': sample['text'], 'label': sample['label'], 'diveye': features})
                
            X_train.append(features)
            Y_train.append(sample['label'])
            
        # Train an XGB model, note that we have not set a config set for this - will be available in the camera-ready repository
        self.model = xgb.XGBClassifier(
            random_state=42, 
            scale_pos_weight=(len(Y_train) - sum(Y_train)) / sum(Y_train),
            max_depth=12,
            n_estimators=200,
            colsample_bytree=0.8,
            subsample=0.7,
            min_child_weight=5,
            gamma=1.0
        )
        self.model.fit(X_train, Y_train)
        
        if self.logging:
            self._log(X_train_log, output_path="train.json")
            print("[DivEye] Training & Model Training Completed.")
            
    def evaluate(self):
        # Temporary variables
        X_test = []
        Y_test = []
        X_test_log = []
        
        if self.model == None:
            raise RuntimeError("You need to train before evaluation!")
        
        # Parse CSV in a loop
        test_dict = pd.read_csv(self.test_dataset).to_dict(orient='records')
        
        for i, sample in enumerate(tqdm(test_dict)):
            features = self.utils.diveye_compute(sample['text'])
            X_test.append(features)
            Y_test.append(sample['label'])
            
            if self.logging:
                X_test_log.append({'text': sample['text'], 'label': sample['label'], 'diveye': features})
        
        # Evaluate the model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)
        precision = precision_score(Y_test, predictions)
        recall = recall_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)
        
        # Logging
        if self.logging:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            self._log(X_test_log, output_path="test.json")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model for computing DivEye")
    parser.add_argument('--train_dataset', type=str, required=True, help='Format: {name}.csv')
    parser.add_argument('--test_dataset', type=str, required=True, help='Format: {name.csv}')
    parser.add_argument('--logging', type=bool, default=True, help="Enable print statements to stdout & logging DivEye features")
    args = parser.parse_args()
    
    diveye = DivEye(model_name=args.model, train_dataset=args.train_dataset, test_dataset=args.test_dataset, logging=args.logging)
    diveye.train()
    diveye.evaluate()
    
# Initialize the DivEye class
if __name__ == "__main__":
    main()