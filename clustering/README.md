# Clustering

This directory is reserved for the next phase of the project: clustering the specialized building embeddings with additional datasets.

## Status

🚧 **Under Development** - This component is not yet implemented.

## Planned Functionality

This phase will focus on:
- Downloading and preprocessing additional building/spatial datasets
- Aligning embeddings across different data sources
- Clustering buildings based on their specialized embeddings
- Generating clusters for spatial analysis and building similarity
- Exporting clustered results for downstream applications

## Future Structure

The planned structure for this directory will include:

```
clustering/
├── download/           # Scripts to download additional datasets
├── preprocessing/      # Data alignment and normalization
├── algorithms/         # Clustering algorithms (K-means, HDBSCAN, etc.)
├── evaluation/         # Cluster quality metrics
├── visualization/      # Tools for visualizing clusters
└── export/            # Export clustered results
```

## Dependencies

This phase depends on:
- Completed embedding pipeline (in `embedding_pipeline/`)
- Specialized embeddings from trained model
- Additional spatial datasets (to be determined)

## Configuration

When implemented, clustering workflows will use the root `config.toml` file with a new `[clustering]` section for parameters such as:
- Clustering algorithm selection
- Number of clusters or density parameters
- Distance metrics
- Additional dataset paths
- Output formats

---

**Note**: Do not add code to this directory until the embedding pipeline is fully validated and published.
