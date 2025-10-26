import os
import pandas as pd
import torch
import random
import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from utils import get_dataframe_to_train, url_to_semantics

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CFG:
    # model_name_or_path = "intfloat/e5-base-v2"
    model_name_or_path = "sentence-transformers/all-mpnet-base-v2"
    data_path = "./input/"
    output_dir = "./e5-base-v2_model"

    EPOCHS = 10
    LEARNING_RATE = 2e-05
    MAX_LENGTH = 256
    BATCH_SIZE = 32

    # Cross-validation settings
    N_SPLITS = 5  # Number of folds for cross-validation
    RANDOM_STATE = 42

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def create_comprehensive_dashboard(cv_results, fold_predictions, filename='cv_dashboard.png'):
    """Create a comprehensive dashboard with all plots in one image"""

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # Define the grid layout
    gs = plt.GridSpec(3, 3, figure=fig)

    # Plot 1: CV Metrics across folds (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    folds = list(range(1, len(cv_results) + 1))
    metrics = ['f1', 'precision', 'recall', 'auc']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    x = np.arange(len(folds))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [result[metric] for result in cv_results]
        ax1.bar(x + i*width, values, width, label=metric.upper(), color=colors[i], alpha=0.8)

        # Add value labels
        for j, v in enumerate(values):
            ax1.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Score')
    ax1.set_title('Cross-Validation Metrics Across Folds', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width*1.5)
    ax1.set_xticklabels(folds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Plot 2: ROC Curves for all folds (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    for i, fold_pred in enumerate(fold_predictions):
        fpr, tpr, _ = roc_curve(fold_pred['true_labels'], fold_pred['probabilities'])
        auc_score = cv_results[i]['auc']
        ax2.plot(fpr, tpr, alpha=0.7, linewidth=2, label=f'Fold {i+1} (AUC = {auc_score:.3f})')

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves - All Folds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Metric distributions (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    metric_data = {metric: [result[metric] for result in cv_results] for metric in metrics}
    box_plot = ax3.boxplot([metric_data[metric] for metric in metrics],
                          labels=[m.upper() for m in metrics],
                          patch_artist=True)

    # Color the boxes
    colors_box = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel('Score')
    ax3.set_title('Metric Distributions Across Folds', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Plot 4: Performance summary table (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')

    # Calculate summary statistics
    summary_data = []
    for metric in metrics:
        values = [result[metric] for result in cv_results]
        summary_data.append([
            metric.upper(),
            f'{np.mean(values):.4f}',
            f'{np.std(values):.4f}',
            f'{np.min(values):.4f}',
            f'{np.max(values):.4f}'
        ])

    table = ax4.table(cellText=summary_data,
                     colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold')

    # Plot 5: Training loss curves if available (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    # This would require storing loss history during training
    ax5.text(0.5, 0.5, 'Training Loss Curves\n(Enable logging to see)',
             ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # Plot 6: Confusion matrix for best fold (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    best_fold_idx = np.argmax([result['f1'] for result in cv_results])
    best_fold_pred = fold_predictions[best_fold_idx]

    cm = confusion_matrix(best_fold_pred['true_labels'], best_fold_pred['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=['No Violation', 'Violation'],
                yticklabels=['No Violation', 'Violation'])
    ax6.set_title(f'Confusion Matrix - Best Fold (Fold {best_fold_idx + 1})',
                 fontsize=14, fontweight='bold')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Actual')

    # Plot 7: Class distribution (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    all_true_labels = np.concatenate([fp['true_labels'] for fp in fold_predictions])
    class_counts = [np.sum(all_true_labels == 0), np.sum(all_true_labels == 1)]
    colors_pie = ['#FF6B6B', '#4ECDC4']
    ax7.pie(class_counts, labels=['No Violation', 'Violation'], autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax7.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')

    # Plot 8: Fold-wise sample sizes (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    train_sizes = [result['train_size'] for result in cv_results]
    val_sizes = [result['val_size'] for result in cv_results]

    x = np.arange(len(folds))
    ax8.bar(x - 0.2, train_sizes, 0.4, label='Training', color='#2E86AB', alpha=0.8)
    ax8.bar(x + 0.2, val_sizes, 0.4, label='Validation', color='#A23B72', alpha=0.8)

    ax8.set_xlabel('Fold')
    ax8.set_ylabel('Number of Samples')
    ax8.set_title('Sample Sizes per Fold', fontsize=14, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(folds)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Add overall title
    plt.suptitle(f'{CFG.model_name_or_path} Cross-Validation Analysis Dashboard',
                fontsize=18, fontweight='bold', y=0.98)

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Comprehensive dashboard saved as {filename}")

def compute_metrics(eval_pred):
    """
    평가 시 사용할 메트릭을 계산합니다.

    Args:
        eval_pred: (predictions, labels) 튜플

    Returns:
        dict: 계산된 메트릭들
    """
    predictions, labels = eval_pred

    # 예측 확률 계산 (softmax 적용)
    probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=1)

    # 각 Column별 AUC 계산
    auc_scores = {}
    # TODO

    # 전체 AUC (클래스 1에 대한)
    try:
        overall_auc = roc_auc_score(labels, probabilities[:, 1])
        auc_scores['overall_auc'] = overall_auc
    except ValueError:
        auc_scores['overall_auc'] = 0.0

    return auc_scores

def evaluate_model(trainer, dataset, true_labels, rules, rule_names):
    """Comprehensive evaluation of the model"""

    predictions = trainer.predict(dataset)
    pred_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)
    pred_labels = np.argmax(predictions.predictions, axis=1)

    # Convert to numpy arrays for indexing
    true_labels = np.array(true_labels)
    rules = np.array(rules)

    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    auc_score = roc_auc_score(true_labels, pred_probs[:, 1].numpy())
    cm = confusion_matrix(true_labels, pred_labels)

    return {
        'predictions': pred_labels,
        'probabilities': pred_probs[:, 1].numpy(),
        'true_labels': true_labels,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'classification_report': classification_report(true_labels, pred_labels)
    }

def run_cross_validation(full_df, tokenizer):
    """Run k-fold cross-validation"""

    # Prepare the data
    full_df['body_with_url'] = full_df['body'].apply(lambda x: x + url_to_semantics(x))
    full_df['input_text'] = full_df['rule'] + "[SEP]" + full_df['body_with_url']

    # Initialize k-fold
    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.RANDOM_STATE)

    cv_results = []
    fold_predictions = []

    print(f"Starting {CFG.N_SPLITS}-fold cross-validation...")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df['rule']), 1):
        print(f"\n🎯 FOLD {fold}/{CFG.N_SPLITS}")
        print("-" * 40)

        # Split data
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Class balance - Train: {train_df['rule_violation'].value_counts().to_dict()}")
        print(f"Class balance - Val: {val_df['rule_violation'].value_counts().to_dict()}")

        # Tokenize
        train_encodings = tokenizer(
            train_df['input_text'].tolist(),
            truncation=True, padding=True, max_length=CFG.MAX_LENGTH
        )
        val_encodings = tokenizer(
            val_df['input_text'].tolist(),
            truncation=True, padding=True, max_length=CFG.MAX_LENGTH
        )

        train_dataset = JigsawDataset(train_encodings, train_df['rule_violation'].tolist())
        val_dataset = JigsawDataset(val_encodings, val_df['rule_violation'].tolist())

        # Initialize model for this fold
        model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)

        training_args = TrainingArguments(
            output_dir=f"{CFG.output_dir}_fold{fold}",
            num_train_epochs=CFG.EPOCHS,
            learning_rate=CFG.LEARNING_RATE,
            per_device_train_batch_size=CFG.BATCH_SIZE,
            per_device_eval_batch_size=CFG.BATCH_SIZE,
            warmup_ratio=0.1,
            weight_decay=0.01,
            report_to="none",
            # save a checkpoint each epoch so we can restore the best weights
            save_strategy="epoch",
            # evaluate each epoch to drive early stopping and model selection
            eval_strategy="epoch",
            logging_steps=10,
            metric_for_best_model="overall_auc",  # 최고 모델 선택 기준
            greater_is_better=True,                # AUC는 높을수록 좋음
            load_best_model_at_end=True,           # 終了時にベストモデルをロード
            save_total_limit=1,                    # ディスク節約: 最新/ベストのみ保持
            fp16=True,
        )
        es_callback = EarlyStoppingCallback(early_stopping_patience=3)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[es_callback],
        )

        # Train
        trainer.train()

        # Evaluate
        fold_results = evaluate_model(
            trainer, val_dataset,
            val_df['rule_violation'].tolist(),
            val_df['rule'].tolist(),
            val_df['rule'].unique()
        )

        # Store results
        cv_results.append({
            'fold': fold,
            'precision': fold_results['precision'],
            'recall': fold_results['recall'],
            'f1': fold_results['f1'],
            'auc': fold_results['auc'],
            'train_size': len(train_df),
            'val_size': len(val_df)
        })

        # Store predictions for this fold
        fold_predictions.append({
            'fold': fold,
            'true_labels': fold_results['true_labels'],
            'predictions': fold_results['predictions'],
            'probabilities': fold_results['probabilities'],
            'rules': val_df['rule'].tolist()
        })

        print(f"Fold {fold} Results:")
        print(f"  Precision: {fold_results['precision']:.4f}")
        print(f"  Recall:    {fold_results['recall']:.4f}")
        print(f"  F1-Score:  {fold_results['f1']:.4f}")
        print(f"  AUC-ROC:   {fold_results['auc']:.4f}")

        # Cleanup: free memory first
        del model, trainer, es_callback
        gc.collect()
        torch.cuda.empty_cache()

        # Remove saved checkpoints for this fold to save disk space.
        # load_best_model_at_end=True ensures the best weights were already
        # loaded into memory before deletion.
        try:
            shutil.rmtree(training_args.output_dir, ignore_errors=True)
        except Exception as e:
            print(f"[WARN] Failed to cleanup {training_args.output_dir}: {e}")

    return cv_results, fold_predictions

def print_cv_summary(cv_results):
    """Print comprehensive cross-validation summary"""

    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)

    metrics = ['precision', 'recall', 'f1', 'auc']

    for metric in metrics:
        values = [result[metric] for result in cv_results]
        mean_val = np.mean(values)
        std_val = np.std(values)

        print(f"\n📊 {metric.upper()}:")
        print(f"  Mean: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Range: {min(values):.4f} - {max(values):.4f}")
        print(f"  Fold values: {[f'{v:.4f}' for v in values]}")

    # Overall summary
    f1_scores = [result['f1'] for result in cv_results]
    auc_scores = [result['auc'] for result in cv_results]

    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"  Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"  Mean AUC-ROC:  {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

    stability = np.std(f1_scores)
    if stability < 0.03:
        stability_text = "Excellent"
    elif stability < 0.06:
        stability_text = "Good"
    elif stability < 0.1:
        stability_text = "Moderate"
    else:
        stability_text = "Variable"
    print(f"  Model Stability: {stability_text} (std: {stability:.4f})")

def main():
    gc.collect()
    torch.cuda.empty_cache()

    seed_everything(CFG.RANDOM_STATE)

    # Load data
    full_df = get_dataframe_to_train(CFG.data_path)

    print(f"📁 DATASET OVERVIEW:")
    print(f"Total samples: {len(full_df)}")
    print(f"Unique rules: {full_df['rule'].nunique()}")
    print(f"Class distribution: {full_df['rule_violation'].value_counts().to_dict()}")
    print(f"Positive rate: {full_df['rule_violation'].mean():.3f}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)

    # Run cross-validation
    cv_results, fold_predictions = run_cross_validation(full_df, tokenizer)

    # Print summary
    print_cv_summary(cv_results)

    # Create comprehensive dashboard
    print(f"\n🎨 GENERATING COMPREHENSIVE DASHBOARD...")
    create_comprehensive_dashboard(cv_results, fold_predictions, 'cv_dashboard.png')

    # Save detailed results
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv('cross_validation_results.csv', index=False)

    # Save all predictions
    all_predictions = []
    for fold_pred in fold_predictions:
        fold_df = pd.DataFrame({
            'fold': fold_pred['fold'],
            'true_label': fold_pred['true_labels'],
            'predicted_label': fold_pred['predictions'],
            'probability': fold_pred['probabilities'],
            'rule': fold_pred['rules']
        })
        all_predictions.append(fold_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv('cv_predictions.csv', index=False)

    print(f"\n✅ CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"📊 Dashboard saved as: cv_dashboard.png")
    print(f"📁 Results saved as: cross_validation_results.csv")
    print(f"📁 Predictions saved as: cv_predictions.csv")

if __name__ == "__main__":
    main()
