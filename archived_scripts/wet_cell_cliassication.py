def wet_cell_classification_accuracy(self, pred, ref_out):
        # Define threshold for wet cells (typically > 0.01m is considered wet)
        threshold = 0.01
        
        # Create binary masks
        pred_wet = (pred > threshold).float()
        ref_wet = (ref_out > threshold).float()
        
        # True positives: cells correctly predicted as wet
        true_positives = torch.sum((pred_wet == 1) & (ref_wet == 1)).float()
        
        # False positives: cells incorrectly predicted as wet
        false_positives = torch.sum((pred_wet == 1) & (ref_wet == 0)).float()
        
        # False negatives: wet cells incorrectly predicted as dry
        false_negatives = torch.sum((pred_wet == 0) & (ref_wet == 1)).float()
        
        # True negatives: correctly predicted dry cells
        true_negatives = torch.sum((pred_wet == 0) & (ref_wet == 0)).float()
        
        # Calculate metrics
        total = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Convert to Python scalars
        metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'true_positives': true_positives.item(),
            'false_positives': false_positives.item(),
            'false_negatives': false_negatives.item(),
            'true_negatives': true_negatives.item()
        }
        
        logger.info(f"Wet cell classification - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1']:.4f}")
        return metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']