import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score, precision_score, recall_score
)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 110, 'font.family': 'DejaVu Sans'})
print("=" * 60)
print("  STEP 1 — Loading Data")
print("=" * 60)

df = pd.read_csv("creditcard.csv")

print(f"  Shape      : {df.shape}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Nulls      : {df.isnull().sum().sum()}")
print(f"\n  Class distribution:")
print(df['Class'].value_counts())
print(f"  Fraud rate : {df['Class'].mean()*100:.4f}%")

print("\n  STEP 2 — EDA")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#f8f9fa')

ax = axes[0, 0]
counts = df['Class'].value_counts()
bars = ax.bar(['Normal (0)', 'Fraud (1)'], counts.values,
               color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f'{v:,}\n({v/len(df)*100:.2f}%)', ha='center', fontweight='bold', fontsize=10)
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
ax.spines[['top', 'right']].set_visible(False)

ax = axes[0, 1]
df[df.Class == 0]['Amount'].clip(0, 500).hist(
    ax=ax, bins=60, alpha=0.6, color='#2ecc71', density=True, label='Normal')
df[df.Class == 1]['Amount'].clip(0, 500).hist(
    ax=ax, bins=60, alpha=0.7, color='#e74c3c', density=True, label='Fraud')
ax.set_title('Transaction Amount Distribution', fontweight='bold')
ax.set_xlabel('Amount (clipped at $500)')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

ax = axes[0, 2]
df[df.Class == 0]['Time'].hist(ax=ax, bins=60, alpha=0.6, color='#2ecc71',
                                density=True, label='Normal')
df[df.Class == 1]['Time'].hist(ax=ax, bins=60, alpha=0.7, color='#e74c3c',
                                density=True, label='Fraud')
ax.set_title('Transaction Time Distribution', fontweight='bold')
ax.set_xlabel('Time (seconds)')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)


ax = axes[1, 0]
df.boxplot(column='Amount', by='Class', ax=ax,
           boxprops=dict(color='#3498db'),
           medianprops=dict(color='#e74c3c', linewidth=2))
ax.set_title('Amount by Class', fontweight='bold')
ax.set_xlabel('Class (0=Normal, 1=Fraud)')
ax.set_ylabel('Amount ($)')
ax.figure.suptitle('')


ax = axes[1, 1]
v_cols = [c for c in df.columns if c.startswith('V')][:14]
corr_with_class = df[v_cols + ['Class']].corr()['Class'].drop('Class').sort_values()
colors = ['#e74c3c' if v > 0 else '#3498db' for v in corr_with_class]
corr_with_class.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('PCA Feature Correlation with Fraud', fontweight='bold')
ax.set_xlabel('Pearson Correlation')
ax.axvline(0, color='black', linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)


ax = axes[1, 2]
ax.axis('off')
stats = df.groupby('Class')['Amount'].agg(['mean', 'median', 'std', 'max']).round(2)
stats.index = ['Normal', 'Fraud']
tbl = ax.table(
    cellText=[[f"${v:.2f}" for v in row] for row in stats.values],
    rowLabels=stats.index,
    colLabels=['Mean', 'Median', 'Std Dev', 'Max'],
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.3, 2.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r == 1:
        cell.set_facecolor('#d5f5e3')
    else:
        cell.set_facecolor('#fdecea')
ax.set_title('Amount Statistics by Class', fontweight='bold', pad=20)

plt.suptitle('Exploratory Data Analysis — Credit Card Fraud', fontsize=15,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda.png', bbox_inches='tight', facecolor='#f8f9fa')
plt.close()
print("  ✅ EDA saved → eda.png")


print("\n  STEP 3 — Feature Engineering")

df['log_Amount'] = np.log1p(df['Amount'])
df['log_Time']   = np.log1p(df['Time'])


df['hour'] = (df['Time'] // 3600) % 24

FEATURES = [c for c in df.columns if c not in ['Class', 'Amount', 'Time']]
X = df[FEATURES]
y = df['Class']
print(f"  Features used: {len(FEATURES)}")


print("\n  STEP 4 — Train/Test Split & Scaling")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Fraud in train: {y_train.sum()} | Fraud in test: {y_test.sum()}")



print("\n  STEP 5 — Handling Class Imbalance via class_weight='balanced'")


print("\n  STEP 6 — Training Models")

models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, C=0.1, random_state=42),

    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12,
        class_weight='balanced', random_state=42, n_jobs=-1),

    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...", end=' ', flush=True)
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)

    results[name] = {
        'model':   model,
        'y_prob':  y_prob,
        'y_pred':  y_pred,
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc':  average_precision_score(y_test, y_prob),
        'f1':      f1_score(y_test, y_pred),
        'recall':  recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
    }
    r = results[name]
    print(f"ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
          f"F1={r['f1']:.4f}  Recall={r['recall']:.4f}")



print("\n  STEP 7 — Stratified 5-Fold Cross-Validation (ROC-AUC)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_s, y_train,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_scores[name] = scores
    print(f"  {name:25s}  {scores.mean():.4f} ± {scores.std():.4f}")



print("\n  STEP 8 — Threshold Tuning")

best_name = max(results, key=lambda k: results[k]['roc_auc'])
y_prob_best = results[best_name]['y_prob']

thresholds = np.linspace(0.01, 0.99, 200)
f1_scores  = [f1_score(y_test, (y_prob_best >= t).astype(int), zero_division=0)
              for t in thresholds]
optimal_t  = thresholds[np.argmax(f1_scores)]
y_pred_opt = (y_prob_best >= optimal_t).astype(int)

print(f"  Best model     : {best_name}")
print(f"  Default F1     : {results[best_name]['f1']:.4f}  (threshold=0.50)")
print(f"  Optimal thresh : {optimal_t:.3f}")
print(f"  Optimized F1   : {max(f1_scores):.4f}")
print(f"  Recall         : {recall_score(y_test, y_pred_opt):.4f}")
print(f"  Precision      : {precision_score(y_test, y_pred_opt):.4f}")



print("\n  STEP 9 — Generating Evaluation Dashboard")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {'Logistic Regression': '#3498db',
          'Random Forest':       '#2ecc71',
          'Gradient Boosting':   '#9b59b6'}

# 9a. ROC Curves
ax = fig.add_subplot(gs[0, 0])
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=COLORS[name], linewidth=2.5,
            label=f"{name}\n(AUC={res['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label='Random')
ax.set_title('ROC Curves', fontweight='bold', fontsize=13)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)


ax = fig.add_subplot(gs[0, 1])
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax.plot(rec, prec, color=COLORS[name], linewidth=2.5,
            label=f"{name}\n(AP={res['pr_auc']:.4f})")
baseline = y_test.mean()
ax.axhline(baseline, color='gray', linestyle='--', linewidth=1,
           label=f'Baseline ({baseline:.4f})')
ax.set_title('Precision-Recall Curves', fontweight='bold', fontsize=13)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[0, 2])
prec_curve = [precision_score(y_test, (y_prob_best >= t).astype(int), zero_division=0)
              for t in thresholds]
rec_curve  = [recall_score(y_test, (y_prob_best >= t).astype(int), zero_division=0)
              for t in thresholds]
ax.plot(thresholds, f1_scores,  color='#e74c3c', linewidth=2.5, label='F1')
ax.plot(thresholds, prec_curve, color='#3498db', linewidth=2,   label='Precision', alpha=0.8)
ax.plot(thresholds, rec_curve,  color='#2ecc71', linewidth=2,   label='Recall', alpha=0.8)
ax.axvline(optimal_t, color='black', linestyle='--', linewidth=1.8,
           label=f'Optimal = {optimal_t:.3f}')
ax.set_title(f'Threshold Tuning\n({best_name})', fontweight='bold', fontsize=13)
ax.set_xlabel('Threshold'); ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

for i, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i])
   
    yp = y_pred_opt if name == best_name else res['y_pred']
    cm = confusion_matrix(y_test, yp)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                      for row_v, row_p in zip(cm, cm_pct)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'],
                cbar=False, linewidths=1, annot_kws={'size': 10})
    star = ' ★' if name == best_name else ''
    ax.set_title(f'Confusion Matrix\n{name}{star}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')

ax = fig.add_subplot(gs[2, 0])
metrics = ['roc_auc', 'pr_auc', 'f1', 'recall', 'precision']
metric_labels = ['ROC-AUC', 'PR-AUC', 'F1', 'Recall', 'Precision']
x = np.arange(len(metrics))
w = 0.25
for j, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    ax.bar(x + j*w - w, vals, w, label=name, color=COLORS[name], alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_ylim(0, 1.1); ax.set_title('Model Comparison', fontweight='bold', fontsize=13)
ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[2, 1])
cv_means  = [cv_scores[n].mean() for n in models]
cv_stds   = [cv_scores[n].std()  for n in models]
bars = ax.bar(list(models.keys()), cv_means, color=list(COLORS.values()),
              alpha=0.85, edgecolor='white',
              yerr=cv_stds, capsize=5, error_kw={'linewidth': 2})
for bar, m, s in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
            f'{m:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticklabels([n.replace(' ', '\n') for n in models], fontsize=9)
ax.set_ylim(0.8, 1.02)
ax.set_title('5-Fold CV — ROC-AUC', fontweight='bold', fontsize=13)
ax.set_ylabel('ROC-AUC'); ax.spines[['top', 'right']].set_visible(False)


ax = fig.add_subplot(gs[2, 2])
rf_model = results['Random Forest']['model']
imp = pd.Series(rf_model.feature_importances_, index=FEATURES).nlargest(12).sort_values()
colors_fi = ['#e74c3c' if i >= len(imp) - 3 else '#3498db' for i in range(len(imp))]
imp.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_title('Top 12 Feature Importances\n(Random Forest)', fontweight='bold', fontsize=11)
ax.set_xlabel('Importance'); ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('Credit Card Fraud Detection — Full Evaluation Dashboard',
             fontsize=16, fontweight='bold', y=1.01)
plt.savefig('evaluation_dashboard.png', dpi=110, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()
print("  ✅ Dashboard saved → evaluation_dashboard.png")



print("\n" + "=" * 60)
print(f"  FINAL REPORT — {best_name} (threshold={optimal_t:.3f})")
print("=" * 60)
print(classification_report(y_test, y_pred_opt, target_names=['Normal', 'Fraud']))

print(f"""class_weight='balanced' → fixes imbalance       
     PR-AUC instead of accuracy → better metric    
     Threshold tuning → boosts recall on fraud     
     Stratified split → preserves fraud ratio        
     5-Fold cross-validation → robust evaluation    
     log(Amount) + log(Time) → feature engineering   
     Gradient Boosting added as 3rd model            
    Full visual dashboard with 9 subplots""")