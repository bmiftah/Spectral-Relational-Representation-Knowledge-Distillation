Spectral Relational Distillation for Intrusion Detection
Overview
This repository implements Spectral Relational Knowledge Distillation (KD) methods for network intrusion detection. The proposed approach distills knowledge from a large teacher model to a compact student model using spectral decomposition of feature relations, offering superior noise robustness and faster convergence compared to standard distillation baselines.

🚀 Key Features
Spectral RRD: Projects embeddings to low-dimensional spectral subspace for efficient relation matching

Weighted Spectral RRD: Adaptive weighting of spectral components based on energy retention

Baseline Implementations: VanillaKD (logit-based) and CorrKD (correlation-based) for fair comparison

Multi-Dataset Support: CICDDoS2019 and CICIoT2023 datasets

Comprehensive Evaluation: Noise robustness, convergence speed, and efficiency analysis

📁 Repository Structure
text
.
├── models/
│   ├── model_2019.py              # Teacher/Student architecture definitions
│   ├── baseline_students.py       # Baseline model implementations
│   └── Feedback_KD/               # Trained model checkpoints
├── training_scripts/
│   ├── spectral_rrd_train.py      # Proposed Spectral RRD training
│   ├── weighted_spectral_train.py # Weighted Spectral RRD training
│   ├── vanilla_kd_train.py        # Vanilla KD baseline
│   └── corr_kd_train.py           # Correlation KD baseline
├── data_loaders/
│   ├── data_loader_2019.py        # CICDDoS2019 dataset loader
│   └── data_loader_2023.py        # CICIoT2023 dataset loader
├── evaluation/
│   ├── evaluation_utils.py         # Comprehensive evaluation metrics
│   ├── noise_robustness.py         # Noise sensitivity analysis
│   ├── convergence_analysis.py     # Training convergence analysis
│   └── ablation_studies/          # Ablation experiment scripts
├── results/
│   ├── noise_robustness/          # Noise robustness results
│   ├── convergence/               # Convergence speed results
│   └── ablation/                  # Ablation study results
└── configs/                       # Configuration files
📊 Models
Teacher Model
Architecture: 3-layer Bidirectional GRU with 512 hidden units

Parameters: ~1.2M parameters

Input: Time-series network traffic features

Output: 8-class intrusion classification

Student Models
Proposed Spectral RRD: Single Bi-GRU (64 hidden) with spectral projection (k=40)

Weighted Spectral RRD: Adaptive spectral weighting based on energy retention

VanillaKD Baseline: Standard logit distillation with temperature scaling

CorrKD Baseline: Correlation alignment of second-order statistics

📈 Results
Key Findings
Noise Robustness: Weighted Spectral RRD shows 76% lower accuracy degradation under Gaussian noise compared to VanillaKD

Convergence Speed: Spectral methods converge 33% faster than baselines (2 vs 3 epochs to 99% accuracy)

Memory Efficiency: 94% reduction in covariance computation memory (900 vs 16,384 elements)

Performance Comparison
Method	CICDDoS2019 Accuracy	CICIoT2023 Accuracy	Noise Robustness	Convergence Speed
Teacher	99.96%	99.35%	-	-
Spectral RRD	99.50%	99.26%	✅ High	✅ Fastest
Weighted Spectral	99.89%	99.25%	✅ Best	✅ Fast
VanillaKD	99.88%	99.23%	⚠️ Moderate	⚠️ Medium
CorrKD	99.89%	99.26%	✅ Good	⚠️ Slowest
🛠️ Installation
bash
# Clone repository
git clone https://github.com/yourusername/spectral-kd-ids.git
cd spectral-kd-ids

# Create environment
conda create -n spectral-kd python=3.8
conda activate spectral-kd

# Install dependencies
pip install -r requirements.txt
🚦 Quick Start
Prepare Data: Download and preprocess CICDDoS2019/CICIoT2023 datasets

Train Teacher: python training_scripts/train_teacher.py

Train Student: python training_scripts/spectral_rrd_train.py

Evaluate: python evaluation/evaluate_model.py --model spectral_rrd

📚 Datasets
CICDDoS2019: Link to dataset

8 attack types + benign traffic

Time-series features from network flows

CICIoT2023: Link to dataset

IoT-specific attack scenarios

8-class classification

📝 Citation
If you use this code in your research, please cite:

bibtex
@article{yourpaper2024,
  title={Spectral Relational Knowledge Distillation for Robust Intrusion Detection},
  author={Your Name},
  journal={Journal of Network Security},
  year={2024}
}
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Canadian Institute for Cybersecurity for datasets

PyTorch community for excellent deep learning tools

Reviewers for valuable feedback
