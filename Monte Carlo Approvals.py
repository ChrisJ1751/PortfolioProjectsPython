import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def simulate_approval_rate(applications, initial_approval_rate, fico_scores, dti_ratios, k, new_policy=False):
    num_inputs = len(fico_scores)
    approval_rates = np.zeros(num_inputs)

    for i in range(num_inputs):
        X = np.array([[fico_scores[j], dti_ratios[j]] for j in range(num_inputs)])
        y = np.array([1 if (fico_scores[j] >= 700 and dti_ratios[j] <= 0.4) else 0 for j in range(num_inputs)])

        X_i = X[i]
        y_i = y[i]

        knn = KNeighborsClassifier(n_neighbors=min(k, num_inputs - 1))
        knn.fit(X, y)

        approved = 0
        for _ in range(applications):
            fico_score = fico_scores[i] + random.uniform(-20, 20)
            dti_ratio = dti_ratios[i] + random.uniform(-5, 5)

            if new_policy:
                if fico_score >= 650 and dti_ratio <= 0.6:
                    approved += 1
            else:
                if fico_score >= 700 and dti_ratio <= 0.4:
                    prediction = knn.predict([X_i])
                    if prediction == 1:
                        approved += 1

        approval_rates[i] = approved / applications * 100

    return approval_rates

# Input parameters
applications = 1000  # Number of applications to simulate
initial_approval_rate = 0.7  # Initial approval rate (without any changes)
fico_scores = [720, 680, 750]  # FICO scores for each input
dti_ratios = [0.3, 0.35, 0.4]  # DTI ratios for each input
k = 3  # Number of nearest neighbors to consider

# Simulate approval rates with current policy
current_approval_rates = simulate_approval_rate(applications, initial_approval_rate, fico_scores, dti_ratios, k, new_policy=False)

# Simulate approval rates with new policy
new_approval_rates = simulate_approval_rate(applications, initial_approval_rate, fico_scores, dti_ratios, k, new_policy=True)

# Calculate the percentage change in approvals
percentage_change = (new_approval_rates - current_approval_rates) / current_approval_rates * 100

# Output results
for i in range(len(fico_scores)):
    print(f"The approval rate for FICO score {fico_scores[i]} and DTI ratio {dti_ratios[i]} with the current policy is: {current_approval_rates[i]:.2f}%")
    print(f"The approval rate for FICO score {fico_scores[i]} and DTI ratio {dti_ratios[i]} with the new policy is: {new_approval_rates[i]:.2f}%")
    print(f"The percentage change in approvals for FICO score {fico_scores[i]} and DTI ratio {dti_ratios[i]} is: {percentage_change[i]:.2f}%\n")









