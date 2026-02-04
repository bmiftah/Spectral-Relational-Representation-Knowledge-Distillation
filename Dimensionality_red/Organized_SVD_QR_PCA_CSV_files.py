# generate_origin_csvs_consistent.py
import csv

data = [
    [20, 0.0267, 0.5429, 0.2727, 0.5433, 0.0055, 0.4951, 49.4, 0.048119, 85.30, 0.686819],
    [40, 0.0365, 0.8858, 0.2802, 0.8858, 0.0174, 0.8623, 16.1, 0.023441, 88.38, 0.372583],
    [60, 0.0389, 0.9998, 0.2833, 0.9998, 0.0300, 0.9998, 9.4, 0.000001, 89.11, 0.003405],
    [80, 0.0475, 0.9998, 0.2956, 0.9998, 0.0436, 0.9998, 6.8, 0.000003, 88.60, 0.004977]
]

# 1. Speedup comparison
with open('speedup_comparison.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k', 'PCA_Time', 'Full_SVD_Time', 'QR_Projection_Time', 'Speedup_vs_SVD'])
    for row in data:
        writer.writerow([row[0], row[1], row[3], row[5], row[7]])

# 2. Energy comparison
with open('energy_comparison.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k', 'PCA_Energy', 'Full_SVD_Energy', 'QR_Projection_Energy', 'Energy_Diff_vs_SVD'])
    for row in data:
        writer.writerow([row[0], row[2], row[4], row[6], row[8]])

# 3. Accuracy metrics
with open('accuracy_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k', 'Energy_Diff_vs_SVD', 'Max_Subspace_Angle', 'Rel_Recon_Error'])
    for row in data:
        writer.writerow([row[0], row[8], row[9], row[10]])

# 4. Method times grouped
with open('method_times_grouped.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Method', 'k20_Time', 'k40_Time', 'k60_Time', 'k80_Time'])
    writer.writerow(['PCA', data[0][1], data[1][1], data[2][1], data[3][1]])
    writer.writerow(['Full SVD', data[0][3], data[1][3], data[2][3], data[3][3]])
    writer.writerow(['QR Projection', data[0][5], data[1][5], data[2][5], data[3][5]])

print("✅ Generated 4 CSV files with consistent naming:")
print("   1. speedup_comparison.csv")
print("   2. energy_comparison.csv")
print("   3. accuracy_metrics.csv")
print("   4. method_times_grouped.csv")
print("\n✅ Consistent terminology:")
print("   - 'QR Projection' (our method)")
print("   - 'Full SVD' (baseline)")
print("   - 'PCA' (baseline)")