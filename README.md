# Statistical-distributions-analysis-of-gene-expression-levels
This repository contains a Python script for analyzing gene expression statistical distributions using histograms. The script processes input data (CSV format) and generates PDF plots based on statistical model fitting.  
## Usage
python gene_expression_distribution.py --steinii steinii_data.csv --inflata inflata_data.csv --boundary_margin 1.0 --gap 2.0 --fit_only  
* --steinii: Path to the CSV file containing gene expression data for Colpoda steinii (log2-transformed).  
* --inflata: Path to the CSV file containing gene expression data for Colpoda inflata (log2-transformed).  
* --boundary_margin: (Optional) Margin for the boundary when plotting. Default is 0.8.  
* --gap: (Optional) Base gap between curves in the 3D plot. Default is 1.0.  
* --fit_only: (Optional) If set, only the fitted curves will be shown (without histogram data).  
* --combined_colormap: (Optional) Color map for combined plots. Default is plasma.  
## Example Output:  
After running the script, you should see the following output files in your directory:  
* **PDF** plot files visualizing the gene expression distributions.  
* **CSV** files with detailed classification and statistical analysis results.  
## Additional Information  
The script uses a variety of statistical models for fitting the gene expression data, including:  
* Normal Distribution  
* Skewed Normal Distribution  
* Student's t-distribution  
* Log-normal Distribution  
* Gamma Distribution  
* Gaussian Mixture Model (GMM)  
