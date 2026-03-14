## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- What companies evaluate in pair programming interviews (technical competence, problem-solving approach, communication, collaboration, code quality, ML judgment)
- The structure and time management of a 45-60 minute pair programming interview
- How to use AI tools appropriately during interviews that allow them

**Apply:**
- Execute the full workflow for common ML interview tasks: data pipeline construction, model training/evaluation, fine-tuning with QLoRA, debugging non-converging models, and building model serving endpoints
- Communicate your reasoning out loud while coding, explaining why (not just what) at each step

**Analyze:**
- Diagnose why a model is not converging by systematically checking data, learning rate, loss function, gradients, and overfitting a single batch

---

## Prerequisites

Before starting this lesson, you should be comfortable with:

- **Supervised learning** (../04-classical-ml/supervised/COURSE.md) -- logistic regression, gradient boosting, train/test splitting, cross-validation, and scikit-learn usage
- **Feature engineering** (../04-classical-ml/feature-engineering/COURSE.md) -- handling missing values, encoding categoricals, creating interaction and aggregation features, preventing data leakage
- **Evaluation metrics** (../04-classical-ml/evaluation-metrics/COURSE.md) -- precision, recall, F1, AUC-ROC, confusion matrices, and choosing the right metric for the problem

---

# Pair Programming Interview Prep

## What Companies Test

Pair programming interviews at top tech companies put you in your own IDE, working on a realistic ML engineering task with an interviewer who acts as your collaborator. This is not LeetCode. You will not be asked to implement a red-black tree or solve dynamic programming puzzles.

They are evaluating:

1. **Technical competence:** Can you write working code for ML tasks? Do you know the tools and libraries?
2. **Problem-solving approach:** Do you break problems down systematically? Do you start simple?
3. **Communication:** Do you think out loud? Do you explain your decisions? Do you ask clarifying questions?
4. **Collaboration:** Do you treat the interviewer as a teammate? Do you take suggestions well?
5. **Code quality:** Is your code readable? Do you handle edge cases? Do you test your assumptions?
6. **ML judgment:** Do you make sensible modeling decisions? Do you catch common pitfalls?

The interview is 45-60 minutes. You'll spend ~5 minutes on setup and clarification, ~35-45 minutes coding, and ~5-10 minutes discussing results and improvements.

---

## General Approach

### Before You Start Coding

**Ask clarifying questions.** This is the single most important thing you can do in the first 3 minutes.

- "What format is the data in? CSV, parquet, database?"
- "How large is the dataset? Thousands, millions, billions of rows?"
- "What's the success metric for this task? Accuracy? F1? Business metric?"
- "Are there any constraints on the tools I can use?"
- "Should I optimize for code quality or for getting a working solution quickly?"

**State your plan.** Before writing code, spend 60 seconds outlining your approach verbally:

"Here's my plan: First, I'll load and inspect the data to understand what we're working with. Then I'll do basic cleaning — handle missing values, check for data types. Next, I'll create a simple baseline model to establish a benchmark. Then I'll engineer features and build a better model. Finally, I'll evaluate properly with train/test split and look at the right metrics. Does that sound reasonable?"

The interviewer may redirect you ("We're more interested in the feature engineering part — spend less time on data cleaning"). This is valuable — it tells you where to focus.

### While Coding

**Think out loud constantly.** Narrate your decisions:

- "I'm checking the shape of the data first... 50,000 rows, 23 columns, that's manageable."
- "I see there are missing values in the 'income' column — about 12%. I'll impute with the median for now since income is likely skewed."
- "I'm going to start with a logistic regression as a baseline because it's fast and interpretable. We can move to gradient boosting after."

**Explain why, not just what.** Don't just type `df.dropna()` — say "I'm dropping rows with missing values for now to get a quick baseline, but in a production system I'd impute based on the distribution and investigate why these values are missing."

**When you're stuck, say so.** "I'm not sure about the right API for this — let me check the docs" is perfectly fine. "I know scikit-learn has a function for this but I can't remember the exact name" is honest and acceptable.

**Write comments.** Brief comments show the interviewer your intent:
```python
# Split before any feature engineering to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### After Getting Results

**Interpret the results.** Don't just print a number — explain what it means:

"Our baseline logistic regression gives us an F1 of 0.72. The precision is higher than recall (0.81 vs 0.64), which means we're being conservative — missing some positives. For this application where catching positives matters, I'd lower the threshold to boost recall, accepting some more false positives."

**Suggest improvements.** Even if time is up, mention what you'd do next:

"Given more time, I'd try: (1) gradient boosting which typically beats logistic regression on tabular data, (2) more feature engineering — interactions between customer tenure and purchase frequency, (3) hyperparameter tuning with cross-validation, and (4) looking at the confusion matrix to understand which segments the model struggles with."

---

## Task 1: "Build a Data Pipeline for This Dataset"

### What They're Evaluating

- Can you load and inspect data efficiently?
- Do you identify data quality issues?
- Do you handle missing values, outliers, and type mismatches appropriately?
- Do you engineer useful features?
- Is your pipeline reproducible and well-structured?

### How to Structure Your Approach

**Step 1: Load and inspect (3 minutes)**
```python
import pandas as pd

df = pd.read_csv('data.csv')
print(f"Shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic stats:\n{df.describe()}")
```

Say: "I always start by understanding the shape, types, and missing values before doing anything else."

**Step 2: Data quality checks (5 minutes)**
```python
# Check for duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Check target distribution
print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True)}")

# Check for suspicious values
for col in df.select_dtypes(include='number'):
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")
```

Say: "I'm checking for duplicates, target imbalance, and outliers. The target is 8% positive — that's imbalanced, we'll need to account for that."

**Step 3: Clean and transform (5 minutes)**
```python
# Handle missing values
# Numeric: impute with median (robust to outliers)
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')

# Categorical: impute with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')

# Encode categoricals
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))
```

Say: "I'm using median imputation because it's robust to outliers. For production, I'd investigate why these values are missing — MAR, MCAR, or MNAR patterns would change my approach."

**Step 4: Feature engineering (10 minutes)**

This is where you differentiate yourself. Create features that show ML intuition:

```python
# Interaction features
df['spend_per_visit'] = df['total_spend'] / (df['num_visits'] + 1)

# Recency features
df['days_since_last_purchase'] = (pd.Timestamp.now() - pd.to_datetime(df['last_purchase'])).dt.days

# Aggregation features
df['purchase_frequency'] = df['num_purchases'] / (df['account_age_days'] + 1)

# Log transform skewed features
import numpy as np
df['log_total_spend'] = np.log1p(df['total_spend'])
```

Say: "I'm creating ratio features because the model can learn from relative patterns better than raw counts. spend_per_visit is more informative than total_spend alone."

### Key Things to Say

- "I always split before feature engineering to prevent data leakage."
- "This feature is skewed, so I'll log-transform it."
- "I'd make this a proper sklearn Pipeline for reproducibility in production."

### Check Your Understanding

<details>
<summary>1. Why should you split the data BEFORE performing feature engineering, and what happens if you do not?</summary>

If you compute feature statistics (e.g., mean for imputation, min/max for scaling) on the full dataset before splitting, information from the test set leaks into the training process. The model indirectly learns about test data through the statistics, making your evaluation overly optimistic. In production, you will not have future data to compute statistics from, so the model's real-world performance will be worse than your evaluation suggests. Always split first, then compute statistics only on the training set.
</details>

<details>
<summary>2. You inspect a dataset and find the target column is 8% positive. What two things should you immediately consider?</summary>

(1) Class imbalance: standard ML algorithms will bias toward the majority class. You need to account for this with weighted loss, SMOTE, or threshold tuning. (2) Evaluation metric: accuracy is misleading for imbalanced data (a model predicting all negatives gets 92% accuracy). Use precision-recall AUC, F1 score, or AUC-ROC instead. You should also stratify your train/test split to ensure both sets have the same 8% positive rate.
</details>

---

## Task 2: "Train a Model and Evaluate It Properly"

### What They're Evaluating

- Do you split data correctly (no data leakage)?
- Do you start with a simple baseline?
- Do you choose appropriate metrics for the problem?
- Do you use cross-validation?
- Can you interpret results and suggest improvements?

### How to Structure Your Approach

**Step 1: Proper split (2 minutes)**
```python
from sklearn.model_selection import train_test_split

# Split FIRST, before any feature engineering that uses target info
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Say: "I'm stratifying the split because we have class imbalance — this ensures both train and test have the same positive rate."

**Step 2: Baseline model (3 minutes)**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)
y_prob = baseline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
```

Say: "I always start with logistic regression as a baseline. It's fast, interpretable, and gives us a floor to beat. Our AUC is 0.78 — decent but there's room for improvement."

**Step 3: Better model (5 minutes)**
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

# Cross-validation for robust estimate
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train on full training set and evaluate on test
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"Test AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

Say: "Gradient boosting almost always beats logistic regression on tabular data. I'm using cross-validation on the training set for model selection, and holding out the test set for final evaluation only."

**Step 4: Interpret and analyze (5 minutes)**
```python
# Feature importance
import matplotlib.pyplot as plt

importances = model.feature_importances_
sorted_idx = importances.argsort()[-15:]  # Top 15
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
plt.title('Top 15 Feature Importances')
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
```

Say: "Feature importance shows that 'days_since_last_purchase' and 'spend_per_visit' are the top predictors — this makes intuitive sense for churn prediction. The confusion matrix shows we're still missing 36% of churners. I'd tune the threshold to catch more."

### Key Things to Say

- "I never evaluate on training data — it tells you nothing about generalization."
- "Cross-validation gives a more robust estimate than a single train/test split."
- "AUC-ROC is better than accuracy here because the classes are imbalanced."
- "I'd look at the precision-recall tradeoff and adjust the threshold based on business needs."

### Check Your Understanding

<details>
<summary>1. Why do you start with logistic regression as a baseline before trying gradient boosting?</summary>

Logistic regression is fast to train, interpretable, and establishes a performance floor. If logistic regression achieves AUC 0.78, you know that any more complex model should beat that. If gradient boosting achieves AUC 0.79, the marginal improvement may not justify the added complexity. If it achieves AUC 0.92, the improvement is clearly worthwhile. Starting simple also helps you verify that your data pipeline and evaluation are correct before introducing model complexity -- it is much easier to debug a logistic regression than a gradient boosting model.
</details>

<details>
<summary>2. You trained a gradient boosting model and got a test AUC of 0.99. What should you immediately suspect?</summary>

Data leakage. A test AUC of 0.99 is suspiciously high for most real-world problems. Common causes: (1) a feature that directly encodes the target (e.g., "fraud_flag" as an input feature), (2) feature engineering computed on the full dataset before splitting, (3) temporal leakage where future information is used to predict past events, (4) duplicate rows appearing in both train and test sets. Investigate by checking feature importances -- if one feature dominates, inspect it carefully.
</details>

---

## Task 3: "Implement a Fine-Tuning Script"

### What They're Evaluating

- Do you know the HuggingFace ecosystem (transformers, datasets, PEFT)?
- Can you set up a training loop with proper configuration?
- Do you understand memory optimization (QLoRA, gradient accumulation)?
- Do you evaluate the fine-tuned model properly?

### How to Structure Your Approach

**Step 1: Setup and data preparation (5 minutes)**
```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load data
dataset = load_dataset('json', data_files='training_data.jsonl')

# Load tokenizer
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize
def tokenize(example):
    return tokenizer(
        example['text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )

tokenized = dataset.map(tokenize, batched=True)
```

Say: "I'm using the HuggingFace ecosystem because it's the standard for fine-tuning. Setting pad_token to eos_token is a common requirement for models that don't have a pad token by default."

**Step 2: Model setup with QLoRA (5 minutes)**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=16,                     # rank
    lora_alpha=32,            # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Say: "QLoRA lets us fine-tune an 8B model on a single GPU by keeping the base model in 4-bit and only training the LoRA adapters. With rank 16 and targeting the attention projections, we're training about 0.1% of the total parameters. I'm using NF4 quantization because it's optimized for normally-distributed weights."

**Step 3: Training (5 minutes)**
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch size = 16
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
)

trainer.train()
```

Say: "I'm using gradient accumulation of 4 to get an effective batch size of 16 while fitting in GPU memory. Learning rate 2e-4 is standard for LoRA fine-tuning. Cosine scheduler with 10% warmup gives smooth convergence."

### Key Things to Say

- "I always separate training and validation data for fine-tuning evaluation."
- "The learning rate for LoRA (1e-4 to 3e-4) is higher than full fine-tuning (1e-5 to 5e-5) because we're only updating a small number of parameters."
- "I'd evaluate with both loss and task-specific metrics — low loss doesn't always mean good task performance."

### Check Your Understanding

<details>
<summary>1. In the QLoRA setup, why is rank r=16 chosen and what are the target modules?</summary>

Rank r=16 is a common default that provides a good balance between model quality and parameter efficiency. It means each LoRA adapter adds matrices of shape d-by-16 and 16-by-d instead of updating the full d-by-d weight matrix. The target modules are the attention projection layers: q_proj, v_proj, k_proj, and o_proj. These are targeted because attention layers capture the most task-relevant patterns, and empirically, adapting attention projections gives the best quality per parameter. With r=16 targeting these 4 modules, you train roughly 0.1% of total parameters.
</details>

---

## Task 4: "Debug This Model That Isn't Converging"

### What They're Evaluating

- Can you systematically diagnose training problems?
- Do you know common causes of training failure?
- Can you read training logs and loss curves?

### Systematic Debugging Checklist

**Say this upfront:** "When a model isn't converging, I follow a systematic checklist. Let me walk through it."

**1. Check the data (first, always)**
```python
# Is the data loaded correctly?
print(f"Training samples: {len(train_dataset)}")
print(f"Sample input: {train_dataset[0]}")
print(f"Label distribution: {Counter(train_dataset['labels'])}")

# Are there NaN or infinite values?
print(f"NaN in features: {np.isnan(X_train).sum()}")
print(f"Inf in features: {np.isinf(X_train).sum()}")
```

Say: "The number one cause of training failure is bad data. I've seen models fail because of NaN values, wrong labels, or empty inputs."

**2. Check the learning rate**
```python
# Is the loss decreasing at all?
# If loss stays flat → learning rate too low
# If loss oscillates wildly → learning rate too high
# If loss explodes (NaN/Inf) → learning rate way too high

# Try a learning rate finder
# Start very small, increase exponentially, plot loss vs LR
```

Say: "If the loss isn't moving, I'll try 10x larger learning rate. If it's oscillating, I'll try 10x smaller. Learning rate is the most common hyperparameter to get wrong."

**3. Check the loss function**
```python
# Is the loss function appropriate for the task?
# Binary classification → BCEWithLogitsLoss (not MSE!)
# Multi-class → CrossEntropyLoss
# Regression → MSELoss or L1Loss

# Is the loss computed on the right outputs?
print(f"Model output shape: {output.shape}")
print(f"Target shape: {target.shape}")
```

Say: "I've seen cases where the loss function was applied to the wrong dimension, or the wrong loss was used for the task."

**4. Check for gradient issues**
```python
# Gradient magnitudes
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}, max={param.grad.abs().max():.6f}")

# Are gradients vanishing (all near zero)?
# Are gradients exploding (very large)?
```

Say: "Vanishing gradients mean the model can't learn — common in deep networks without skip connections. Exploding gradients mean the model diverges — use gradient clipping."

**5. Sanity check: can the model overfit one batch?**
```python
# If it can't memorize a single batch, something is fundamentally wrong
single_batch = next(iter(train_loader))
for i in range(100):
    loss = model.train_step(single_batch)
    if i % 10 == 0:
        print(f"Step {i}: loss = {loss:.4f}")
```

Say: "This is the most powerful debugging technique. If the model can't overfit a single batch, the problem is in the model architecture, loss function, or data format — not in the training hyperparameters."

### Key Things to Say

- "I debug data before model — 80% of training issues are data issues."
- "Can-it-overfit-one-batch is my first architectural sanity check."
- "I always visualize the loss curve — the shape tells me what's wrong."

---

## Task 5: "Write a Model Serving Endpoint"

### What They're Evaluating

- Can you wrap a model in a production-ready API?
- Do you handle errors, validation, and edge cases?
- Do you think about performance (batching, caching)?

### How to Structure Your Approach

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI()

# Load model once at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = torch.load("model.pt")
    model.eval()

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    if len(request.features) != 23:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 23 features, got {len(request.features)}"
        )

    # Run inference
    with torch.no_grad():
        input_tensor = torch.tensor([request.features], dtype=torch.float32)
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    return PredictionResponse(
        prediction=1 if probability > 0.5 else 0,
        confidence=probability if probability > 0.5 else 1 - probability
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
```

Say: "I'm using FastAPI because it's the standard for Python ML APIs — it gives us automatic validation via Pydantic, async support, and auto-generated docs. I load the model once at startup to avoid loading on every request. The health endpoint is essential for orchestration systems like Kubernetes."

### Key Things to Say

- "I always load the model once at startup, never per-request."
- "Input validation prevents cryptic errors from bad inputs."
- "The health endpoint is crucial for load balancers and container orchestration."
- "For production, I'd add request logging, monitoring, and rate limiting."
- "For GPU models, I'd add batching — accumulate requests and process them together for better GPU utilization."

---

## Using AI Tools in the Interview

### When the Interviewer Expects You to Use AI Tools

Modern ML engineering uses AI tools (Copilot, Claude, ChatGPT). Some interviewers explicitly allow or encourage them. If they say "use whatever tools you normally use," go ahead.

**Use AI tools for:**
- Remembering exact API signatures ("What's the argument name for stratified splitting in sklearn?")
- Boilerplate code (imports, configuration objects)
- Looking up library-specific syntax

**Don't use AI tools for:**
- Core algorithmic decisions (which model to use, what metrics to choose)
- Interpreting results
- Architectural decisions

**Always say what you're doing:** "I'm going to look up the exact arguments for BitsAndBytesConfig since I don't remember the parameter names." This shows the interviewer you know what you need — you just want the syntax.

### When You Should Demonstrate Knowledge

If the interviewer says "no AI tools" or you sense they want to test your knowledge:
- Write code from memory even if it's slightly wrong — show you know the patterns
- It's fine to say "I think the parameter is called `stratify` — let me check" and then check docs
- Focus on showing you understand the concepts, not perfect syntax

### Check Your Understanding

<details>
<summary>1. The "overfit one batch" technique is described as the most powerful debugging tool. Why?</summary>

If the model cannot memorize a single batch (loss does not approach zero after 100 training steps on the same batch), the problem is fundamental -- it is in the model architecture, loss function, or data format, NOT in hyperparameters. This eliminates entire categories of issues in one test. If the model CAN overfit one batch, the problem is likely in learning rate, regularization, or data quality across the full dataset. This single check narrows the debugging space dramatically.
</details>

---

## Universal Tips

### Start Simple

Always begin with the simplest version that works:
- Logistic regression before gradient boosting before neural networks
- Mean imputation before sophisticated imputation
- Random split before cross-validation
- Single feature before all features

Say: "Let me start with a simple baseline so we have something to compare against."

### Communicate Constantly

The interviewer can't read your mind. When you're silent for 30 seconds, they don't know if you're thinking deeply or stuck. Narrate:
- "I'm thinking about whether to impute or drop these missing values..."
- "I'm choosing between one-hot encoding and label encoding for this categorical..."
- "I know there's a faster way to do this, but let me get the simple version working first..."

### Test Your Assumptions

Before trusting any result, sanity-check it:
- "This accuracy seems too high — let me check for data leakage."
- "Let me verify the shapes match before training."
- "Let me print a few predictions to make sure they look reasonable."

### Handle Mistakes Gracefully

You will make mistakes. The interviewer expects this. What matters is how you respond:
- "Oh, I made a data leakage mistake — I computed feature statistics on the full dataset before splitting. Let me fix that by computing statistics only on the training set."
- "That error message tells me the tensor dimensions don't match. The issue is probably in my data preprocessing step — let me check the shapes."

### Time Management

- Don't spend 15 minutes on perfect data cleaning if the task is about modeling
- Don't hypertune hyperparameters — use reasonable defaults
- If you're stuck on a bug for > 3 minutes, ask the interviewer for a hint
- Leave 5 minutes at the end to discuss results and improvements

---

## Common Pitfalls

**1. Coding in silence.**
The interviewer cannot read your mind. If you are silent for 30 seconds, they do not know if you are thinking deeply or completely stuck. Narrate constantly: what you are doing, why you are doing it, what you expect to see. Even "I'm thinking about whether to impute or drop these missing values" is valuable communication.

**2. Jumping to a complex model without a baseline.**
Starting with a neural network or complex ensemble before trying logistic regression signals poor judgment. The baseline is not just for comparison -- it validates that your data pipeline and evaluation code are correct. If logistic regression gives AUC 0.99 or AUC 0.50, you know something is wrong before investing time in a complex model.

**3. Not splitting data before feature engineering.**
This is the most common data leakage mistake in pair programming interviews, and interviewers watch for it specifically. Computing feature statistics (imputation values, scaling parameters, target encodings) on the full dataset before splitting leaks test set information into training. Always call train_test_split first, then fit transformers on the training set only.

**4. Reporting accuracy on imbalanced datasets.**
If the dataset is 8% positive and you report 92% accuracy, the interviewer will immediately ask about the confusion matrix. A model that predicts all negatives achieves 92% accuracy. Always use precision, recall, F1, or AUC-ROC for imbalanced problems, and explain why you chose that metric.

---

## Hands-On Exercises

### Exercise 1: Timed Data Pipeline and Model Training (30 minutes)

Set a timer for 30 minutes. Using a publicly available dataset (e.g., the Kaggle Telco Customer Churn dataset or the UCI Adult Income dataset), complete the following from scratch:

1. Load and inspect the data (shape, types, missing values, target distribution) -- 3 minutes
2. Clean the data (handle missing values, encode categoricals) -- 5 minutes
3. Split the data (stratified, before any feature engineering that uses target info) -- 2 minutes
4. Create at least 3 engineered features (ratio features, log transforms, interaction features) -- 5 minutes
5. Train a logistic regression baseline and report AUC-ROC -- 3 minutes
6. Train a gradient boosting model with cross-validation and report AUC-ROC -- 5 minutes
7. Analyze feature importances and the confusion matrix -- 5 minutes
8. State 3 improvements you would make with more time -- 2 minutes

Practice this until you can complete it confidently within 30 minutes while narrating your reasoning out loud.

### Exercise 2: Timed Debugging Challenge (15 minutes)

Set a timer for 15 minutes. Write a deliberately broken training script (introduce one of these bugs: wrong loss function for the task, learning rate too high by 100x, NaN values in the input data, or target labels that are all the same class). Then, without looking at the bug you introduced, follow the systematic debugging checklist from Task 4:

1. Check the data (NaN, inf, label distribution)
2. Check the learning rate (is loss moving? oscillating? exploding?)
3. Check the loss function (is it appropriate for the task?)
4. Check gradients (vanishing or exploding?)
5. Sanity check: overfit one batch

Time how long it takes you to find and fix the bug. Repeat with different bugs until you can diagnose each one in under 5 minutes.

---

## Key Takeaways

1. Ask clarifying questions and state your plan before coding
2. Think out loud constantly — narrate your decisions and reasoning
3. Start with the simplest approach that works, then iterate
4. Always split data before any feature engineering (prevent leakage)
5. Use appropriate metrics for the problem (not just accuracy)
6. Can-it-overfit-one-batch is your best debugging tool
7. Interpret results — don't just print numbers
8. Suggest improvements even if you run out of time
9. Handle mistakes gracefully — diagnose and fix, don't panic
10. The interviewer is your collaborator, not your adversary

---

## Summary

This lesson covered the practical skills tested in pair programming ML interviews: building data pipelines (load, inspect, clean, engineer features), training and evaluating models (baseline first, then iterate; proper splitting and metric selection), implementing fine-tuning with QLoRA (HuggingFace ecosystem, memory optimization), debugging non-converging models (systematic checklist: data, learning rate, loss function, gradients, overfit-one-batch), and building model serving endpoints (FastAPI, input validation, health checks). The meta-skills are just as important: ask clarifying questions before coding, think out loud constantly, start simple, interpret results rather than just printing numbers, and handle mistakes gracefully.

## What's Next

You have completed the full course curriculum. At this point you should:

1. **Practice under timed conditions** -- the exercises in this lesson and the system design lesson should be repeated until you can complete them confidently within the time limits
2. **Review the 20 core concepts** (../concepts/COURSE.md) weekly using spaced repetition until you can deliver each explanation in 60 seconds
3. **Run mock interviews** -- pair with someone and alternate between interviewer and candidate roles using the system design worked examples and pair programming tasks
4. **Build portfolio projects** that demonstrate the full stack: an ML model integrated as an agent tool, evaluated end-to-end, and optimized for production -- this is the combination that differentiates you
