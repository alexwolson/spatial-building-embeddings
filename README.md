## Spatial Building Embeddings

- **Dataset scale**: 1,006,160 unique buildings and 5,772,534 unique images across the combined train, validation, and test partitions.

### Pipeline Execution Order

The following graph shows the dependency relationships between all `submit_*.sh` scripts. Execute them in the order indicated by the arrows (scripts at the top/left must complete before those below/right):

```mermaid
flowchart TD
    A[submit_tar_jobs.sh<br/>Process raw tar files]
    
    B[submit_fingerprint_jobs.sh<br/>Compute fingerprints]
    C[submit_embedding_jobs.sh<br/>Generate embeddings]
    
    D[submit_merge.sh<br/>Merge & create splits]
    
    E[submit_visual_neighbors.sh<br/>Compute visual neighbors]
    
    F[submit_optuna_tuning.sh<br/>Hyperparameter tuning]
    G[submit_best_training.sh<br/>Train with best params]
    
    A --> B
    A --> C
    A --> D
    C --> D
    
    B --> E
    
    D --> F
    D --> G
    
    E --> F
    E --> G
    
    style A fill:#e1f5e1
    style B fill:#e1f0f5
    style C fill:#e1f0f5
    style D fill:#fff4e1
    style E fill:#f5e1f0
    style F fill:#f0e1f5
    style G fill:#f0e1f5
```

