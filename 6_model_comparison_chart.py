import matplotlib.pyplot as plt

# Replace these with your actual model scores
models = ['Linear Regression', 'Random Forest']
r2_scores = [0.6561, 0.7067]

# Plotting
plt.figure(figsize=(6, 4))
plt.bar(models, r2_scores, color=['skyblue', 'seagreen'])
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.ylim(0.6, 0.75)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save the chart
plt.tight_layout()
plt.savefig('outputs/model_comparison.png')
plt.show()
