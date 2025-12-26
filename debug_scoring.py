import torch
from src.eggroll.scoring import VectorizedScorer, ChunkedScorer

def debug_scoring():
    B, P, D_in, D_out, M = 1, 1, 4, 2, 4
    
    torch.manual_seed(42)
    H_q = torch.randn(B, D_in)
    H_d = torch.randn(B, P, D_in)
    W = torch.randn(D_out, D_in)
    A = torch.randn(M, D_out)
    B_noise = torch.randn(M, D_in)
    
    full_scorer = VectorizedScorer(sigma=0.1)
    chunked_scorer = ChunkedScorer(sigma=0.1, chunk_size=2)
    
    full_scores = full_scorer.compute_all_scores(H_q, H_d, W, A, B_noise)
    chunked_scores = chunked_scorer.compute_all_scores_chunked(H_q, H_d, W, A, B_noise)
    
    print(f"Full scores shape: {full_scores.shape}")
    print(f"Chunked scores shape: {chunked_scores.shape}")
    
    if full_scores.shape == chunked_scores.shape:
        diff = (full_scores - chunked_scores).abs()
        print(f"Max diff: {diff.max().item()}")
        if diff.max() > 1e-5:
            print("Scores mismatch!")
            print("Full scores:\n", full_scores)
            print("Chunked scores:\n", chunked_scores)
    else:
        print("Shape mismatch!")

if __name__ == "__main__":
    debug_scoring()
