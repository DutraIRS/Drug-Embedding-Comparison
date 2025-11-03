# Drug Embedding Comparison - Classification Task Pipeline
# This script runs the complete training, analysis, and testing pipeline for CLASSIFICATION task

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Drug Embedding Comparison - CLASSIFICATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Task: Predicting presence/absence of side effects (binary)" -ForegroundColor Magenta
Write-Host "Loss: Weighted BCE with alpha=0.03" -ForegroundColor Magenta
Write-Host ""

# Step 1: Train models with grid search
Write-Host "[Step 1/3] Training models with grid search..." -ForegroundColor Yellow
Write-Host "This will train all model configurations with multiple runs per config." -ForegroundColor Gray
Write-Host ""

python src/train.py --task classification

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Training complete!" -ForegroundColor Green
Write-Host ""

# Step 2: Analyze results and find best configurations
Write-Host "[Step 2/3] Analyzing results and finding best configurations..." -ForegroundColor Yellow
Write-Host "This will generate best_configs.csv and visualization plots." -ForegroundColor Gray
Write-Host ""

python src/analyze_results.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Analysis failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Analysis complete!" -ForegroundColor Green
Write-Host ""

# Step 3: Test best models on test set
Write-Host "[Step 3/3] Testing best models on test set..." -ForegroundColor Yellow
Write-Host "This will retrain best configs on train+val and evaluate on test set." -ForegroundColor Gray
Write-Host ""

python src/test.py --task classification

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Testing failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CLASSIFICATION Pipeline Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - saved_models/all_results.csv" -ForegroundColor White
Write-Host "  - saved_models/best_configs.csv" -ForegroundColor White
Write-Host "  - saved_models/final_test_results.csv" -ForegroundColor White
Write-Host "  - saved_models/analysis_plots/" -ForegroundColor White
Write-Host "  - saved_models/final_test_plots/" -ForegroundColor White
Write-Host ""
Write-Host "Evaluation Metric: AUROC (Area Under ROC Curve)" -ForegroundColor Cyan
Write-Host ""
