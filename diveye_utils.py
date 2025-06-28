import numpy as np
import torch
from scipy.stats import skew, kurtosis, entropy

class DivEyeUtils:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.features = []
        
    def _log_likelihoods(self, text):
        tokens = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].squeeze(0)
        shift_labels = tokens[:, 1:].squeeze(0)
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        token_log_likelihoods = log_probs[range(shift_labels.shape[0]), shift_labels].cpu().numpy()
        
        return token_log_likelihoods
    
    def _surprisal(self, text):
        log_likelihoods = self._log_likelihoods(text)
        surprisals = -log_likelihoods
        return surprisals
        
    def diveye_compute(self, text):
        surprisals = self._surprisal(text)
        log_likelihoods = self._log_likelihoods(text)
        
        if len(surprisals) < 10 or len(log_likelihoods) < 3:
            return self.diveye_compute(text) # Recursively call diveye_compute if we note any error

        s = np.array(surprisals)
        mean_s, std_s, var_s, skew_s, kurt_s = np.mean(s), np.std(s), np.var(s), skew(s), kurtosis(s)
        diff_s = np.diff(s)
        mean_diff, std_diff = np.mean(diff_s), np.std(diff_s)
        first_order_diff = np.diff(log_likelihoods)
        second_order_diff = np.diff(first_order_diff)
        var_2nd, entropy_2nd = np.var(second_order_diff), entropy(np.histogram(second_order_diff, bins=20, density=True)[0])
        autocorr_2nd = np.corrcoef(second_order_diff[:-1], second_order_diff[1:])[0, 1] if len(second_order_diff) > 1 else 0

        return [mean_s, std_s, var_s, skew_s, kurt_s, mean_diff, std_diff, var_2nd, entropy_2nd, autocorr_2nd]