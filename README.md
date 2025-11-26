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
    
    subgraph Metadata [Difficulty Metadata (Choose One)]
        E[submit_difficulty_metadata.sh<br/>Geographic neighbors]
        F[submit_visual_neighbors.sh<br/>Visual neighbors]
    end
    
    G[submit_optuna_tuning.sh<br/>Hyperparameter tuning]
    H[submit_best_training.sh<br/>Train with best params]
    
    A --> B
    A --> C
    A --> D
    C --> D
    
    D --> E
    B --> F
    
    D --> G
    D --> H
    
    E --> G
    E --> H
    F --> G
    F --> H
    
    style A fill:#e1f5e1
    style B fill:#e1f0f5
    style C fill:#e1f0f5
    style D fill:#fff4e1
    style E fill:#f5e1f0
    style F fill:#f5e1f0
    style G fill:#f0e1f5
    style H fill:#f0e1f5
```

**Note**: Training requires difficulty metadata, which can be generated **either** geographically (`submit_difficulty_metadata.sh`) **or** visually (`submit_visual_neighbors.sh`). You do not need to run both; choose the method that suits your experiment.

