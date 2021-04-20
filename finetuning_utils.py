from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import XLMRobertaForSequenceClassification

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    metrics = {}
    # metrics["precision"], metrics["recall"], metrics["f1"], _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary")
    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["mathews_correlation"] = matthews_corrcoef(labels, preds)
    return metrics

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base")
    return model
