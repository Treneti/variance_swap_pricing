# Variance Swap Pricing & Greeks

This repository contains a solution for pricing an SPX Variance Swap and calculating its Greeks. The solution implements three pricing methodologies:
1. **Analytical Approach**
2. **Monte Carlo: Geometric Brownian Motion**
3. **Monte Carlo: Heston Model** 

## 📂 Project Structure

* **`Variance_Swap_Solution.ipynb`**: The main entry point. Contains the execution, results, and **the detailed write-up (approach, assumptions, and findings)**.
* **`utils.py`**: Contains the core logic, pricing algorithms, and helper functions.
* **`requirements.txt`**: Dependencies.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook to view code and narrative
jupyter notebook Variance_Swap_Solution.ipynb