# demo_compare_dr_methods_fixed.py
import numpy as np
import torch
import time
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
import csv
import os


def save_to_csv(results_dict, filename):
    """Save results to CSV file"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['k', 'Method', 'Time(s)', 'Energy', 'Speedup_vs_SVD',
                         'Energy_Diff_vs_SVD', 'Max_Subspace_Angle', 'Rel_Recon_Error'])

        for k, results in results_dict.items():
            writer.writerow([k, 'PCA',
                             f"{results['pca']['time']:.4f}",
                             f"{results['pca']['energy']:.4f}",
                             '-',
                             f"{results['svd']['energy'] - results['pca']['energy']:.6f}",
                             '0.0',
                             '-'])

            writer.writerow([k, 'Full SVD',
                             f"{results['svd']['time']:.4f}",
                             f"{results['svd']['energy']:.4f}",
                             '1.0',
                             '0.000000',
                             '0.0',
                             '0.000000'])

            writer.writerow([k, 'Randomized SVD',
                             f"{results['rand_svd']['time']:.4f}",
                             f"{results['rand_svd']['energy']:.4f}",
                             f"{results['comparison']['speedup_vs_svd']:.1f}",
                             f"{results['comparison']['energy_difference']:.6f}",
                             f"{results['comparison']['max_subspace_angle']:.2f}",
                             f"{results['comparison']['reconstruction_error']:.6f}"])


def randomized_svd_qr(X, k, n_iters=2):
    """Fixed randomized SVD implementation"""
    n, d = X.shape

    # Step 1: Random Gaussian matrix
    np.random.seed(42)  # For reproducibility
    Omega = np.random.randn(d, k)

    # Step 2: Power iterations
    Y = X @ Omega
    for _ in range(n_iters):
        Y = X @ (X.T @ Y)

    # Step 3: QR orthogonalization
    Q, R = np.linalg.qr(Y, mode='reduced')

    # Step 4: Project X onto Q
    B = Q.T @ X

    # Step 5: SVD of small matrix
    U_b, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 6: Reconstruct
    U = Q @ U_b

    return U[:, :k], S[:k], Vt[:k, :]


def compare_methods(embeddings, k=30):
    """Compare methods without plotting"""
    n, d = embeddings.shape
    print(f"\n=== DR Method Comparison (n={n}, d={d}, k={k}) ===")

    results = {}

    # 1. PCA
    print("1. Running PCA...")
    start = time.time()
    pca = PCA(n_components=k, random_state=42)
    pca_proj = pca.fit_transform(embeddings)
    pca_time = time.time() - start
    pca_energy = np.sum(pca.explained_variance_ratio_)

    # 2. Full SVD
    print("2. Running Full SVD...")
    start = time.time()
    U_full, S_full, Vt_full = np.linalg.svd(embeddings, full_matrices=False)
    svd_time = time.time() - start
    svd_energy = np.sum(S_full[:k] ** 2) / np.sum(S_full ** 2)

    # 3. Randomized SVD
    print("3. Running Randomized SVD...")
    start = time.time()
    U_r, S_r, Vt_r = randomized_svd_qr(embeddings, k=k, n_iters=2)
    rand_svd_time = time.time() - start

    # Calculate energy for randomized SVD
    X_norm = np.linalg.norm(embeddings, 'fro') ** 2
    rand_recon = U_r @ np.diag(S_r) @ Vt_r
    rand_energy = 1 - (np.linalg.norm(embeddings - rand_recon, 'fro') ** 2 / X_norm)

    # 4. Compute metrics
    svd_recon = U_full[:, :k] @ np.diag(S_full[:k]) @ Vt_full[:k, :]
    recon_error = np.linalg.norm(svd_recon - rand_recon, 'fro') / np.linalg.norm(svd_recon, 'fro')

    # Subspace angles
    from scipy.linalg import subspace_angles
    angles = subspace_angles(Vt_full[:k, :].T, Vt_r.T)
    max_angle = np.max(np.degrees(angles))

    # Compile results
    results = {
        'pca': {'time': pca_time, 'energy': pca_energy},
        'svd': {'time': svd_time, 'energy': svd_energy},
        'rand_svd': {'time': rand_svd_time, 'energy': rand_energy},
        'comparison': {
            'speedup_vs_svd': svd_time / rand_svd_time,
            'speedup_vs_pca': pca_time / rand_svd_time,
            'energy_difference': svd_energy - rand_energy,
            'reconstruction_error': recon_error,
            'max_subspace_angle': max_angle
        }
    }

    # Print summary
    print(f"\n   Time - PCA: {pca_time:.3f}s, SVD: {svd_time:.3f}s, RandSVD: {rand_svd_time:.3f}s")
    print(f"   Energy - PCA: {pca_energy:.4f}, SVD: {svd_energy:.4f}, RandSVD: {rand_energy:.4f}")
    print(f"   Speedup vs SVD: {svd_time / rand_svd_time:.1f}x")
    print(f"   Energy diff (SVD-Rand): {svd_energy - rand_energy:.6f}")
    print(f"   Max subspace angle: {max_angle:.2f}°")

    return results


def main():
    # Generate realistic data (replace with your actual embeddings)
    n_samples = 2560
    d = 512
    np.random.seed(42)

    # Create low-rank data with noise
    rank_true = 50
    U = np.random.randn(n_samples, rank_true)
    V = np.random.randn(rank_true, d)
    signal = U @ V
    noise = 0.1 * np.random.randn(n_samples, d)
    embeddings = signal + noise

    print(f"Generated {n_samples} embeddings of dimension {d}")
    print(f"True rank: {rank_true}")

    # Test multiple k values
    k_values = [20, 40, 60, 80]
    all_results = {}

    for k in k_values:
        print("\n" + "=" * 70)
        print(f"Testing with k = {k}")
        print("=" * 70)

        results = compare_methods(embeddings, k=k)
        all_results[k] = results

    # Save to CSV
    save_to_csv(all_results, 'dr_comparison_results.csv')
    print(f"\n✅ Results saved to 'dr_comparison_results.csv'")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'k':>4} {'RandSVD Time':<12} {'Speedup vs SVD':<15} {'Energy Diff':<12} {'Max Angle':<10}")
    print("-" * 70)

    for k in k_values:
        res = all_results[k]
        print(f"{k:>4} {res['rand_svd']['time']:<12.3f} {res['comparison']['speedup_vs_svd']:<15.1f}x "
              f"{res['comparison']['energy_difference']:<12.6f} {res['comparison']['max_subspace_angle']:<10.2f}°")


if __name__ == "__main__":
    main()