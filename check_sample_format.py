import sys
import os
import anndata as ad

def check_sample_format(sample_path):
    adata = ad.read_h5ad(sample_path)
    print(adata.var)
    print(adata.obs)
    print(adata)

if __name__ == "__main__":
    check_sample_format(sys.argv[1])    
    print("Sample format checked successfully")

