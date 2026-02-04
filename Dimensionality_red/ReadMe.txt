Run first :-Code_Compare_dr_methods_real.py which will save file given below :- 
it will give the below output in console :- 

demo_compare_dr_methods_real.py 
Generated 2560 embeddings of dimension 512  for CICDDOS2019
True rank: 50

======================================================================
Testing with k = 20
======================================================================

=== DR Method Comparison (n=2560, d=512, k=20) ===
1. Running PCA...
2. Running Full SVD...
3. Running Randomized SVD...

   Time - PCA: 0.070s, SVD: 0.329s, RandSVD: 0.007s
   Energy - PCA: 0.5429, SVD: 0.5433, RandSVD: 0.4951
   Speedup vs SVD: 46.2x
   Energy diff (SVD-Rand): 0.048119
   Max subspace angle: 85.30°

======================================================================
Testing with k = 40
======================================================================

=== DR Method Comparison (n=2560, d=512, k=40) ===
1. Running PCA...
2. Running Full SVD...
3. Running Randomized SVD...

   Time - PCA: 0.037s, SVD: 0.266s, RandSVD: 0.016s
   Energy - PCA: 0.8858, SVD: 0.8858, RandSVD: 0.8623
   Speedup vs SVD: 16.1x
   Energy diff (SVD-Rand): 0.023441
   Max subspace angle: 88.38°

======================================================================
Testing with k = 60
======================================================================

=== DR Method Comparison (n=2560, d=512, k=60) ===
1. Running PCA...
2. Running Full SVD...
3. Running Randomized SVD...

   Time - PCA: 0.044s, SVD: 0.264s, RandSVD: 0.028s
   Energy - PCA: 0.9998, SVD: 0.9998, RandSVD: 0.9998
   Speedup vs SVD: 9.4x
   Energy diff (SVD-Rand): 0.000001
   Max subspace angle: 89.11°

======================================================================
Testing with k = 80
======================================================================

=== DR Method Comparison (n=2560, d=512, k=80) ===
1. Running PCA...
2. Running Full SVD...
3. Running Randomized SVD...

   Time - PCA: 0.055s, SVD: 0.266s, RandSVD: 0.041s
   Energy - PCA: 0.9998, SVD: 0.9998, RandSVD: 0.9998
   Speedup vs SVD: 6.4x
   Energy diff (SVD-Rand): 0.000003
   Max subspace angle: 88.60°

 Results saved to 'dr_comparison_results.csv'

======================================================================
FINAL SUMMARY
======================================================================
   k RandSVD Time Speedup vs SVD  Energy Diff  Max Angle 
----------------------------------------------------------------------
  20 0.007        46.2           x 0.048119     85.30     °
  40 0.016        16.1           x 0.023441     88.38     °
  60 0.028        9.4            x 0.000001     89.11     °
  80 0.041        6.4            x 0.000003     88.60     °

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Run Organized_SVD_QR_PCA_CSV_files.py after running Code_Compare_dr_methods_real.py and will save the following iles 
 Generated 4 CSV files with consistent naming:
   1. speedup_comparison.csv
   2. energy_comparison.csv
   3. accuracy_metrics.csv
   4. method_times_grouped.csv

 Consistent terminology we used in the manuscript for back and forth check:
   - 'QR Projection' (our method)
   - 'Full SVD' (baseline)
   - 'PCA' (baseline)