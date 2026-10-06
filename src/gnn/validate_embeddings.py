import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

EXPECTED_SHAPE = (20, 32)

print("=" * 60)
print("GNN EMBEDDING VALIDATION")
print("=" * 60)

# ==================================================
# Check directory
# ==================================================

if not os.path.exists(EMBEDDINGS_DIR):
    raise FileNotFoundError(
        f"Embedding directory not found: {EMBEDDINGS_DIR}"
    )

# ==================================================
# Find embedding files
# ==================================================

files = sorted(
    [
        f
        for f in os.listdir(EMBEDDINGS_DIR)
        if f.endswith(".npy")
    ]
)

print("Embedding files:", len(files))

if len(files) == 0:
    raise FileNotFoundError(
        "No embedding files found."
    )

# ==================================================
# Validation counters
# ==================================================

valid = 0
invalid_shape = 0
invalid_values = 0

first_shape = None
first_embedding = None

# ==================================================
# Validate every embedding
# ==================================================

for i, filename in enumerate(files):

    path = os.path.join(
        EMBEDDINGS_DIR,
        filename
    )

    try:
        embedding = np.load(path)

        # Check shape
        if embedding.shape != EXPECTED_SHAPE:
            invalid_shape += 1
            print(
                f"Invalid shape: {filename} "
                f"{embedding.shape}"
            )
            continue

        # Check NaN / Inf
        if not np.isfinite(embedding).all():
            invalid_values += 1
            print(
                f"Invalid values: {filename}"
            )
            continue

        valid += 1

        if first_embedding is None:
            first_embedding = embedding
            first_shape = embedding.shape

    except Exception as e:
        print(
            f"Error reading {filename}: {e}"
        )

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1}/{len(files)}"
        )

# ==================================================
# Results
# ==================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print("Total files:", len(files))
print("Valid embeddings:", valid)
print("Invalid shapes:", invalid_shape)
print("Invalid values:", invalid_values)

print("\nExpected shape:", EXPECTED_SHAPE)
print("First embedding shape:", first_shape)

if first_embedding is not None:

    print(
        "\nFirst stock embedding:"
    )

    print(
        first_embedding[0]
    )

# ==================================================
# Final status
# ==================================================

if (
    valid == len(files)
    and invalid_shape == 0
    and invalid_values == 0
):
    print("\n===== VALIDATION PASSED =====")
else:
    print("\n===== VALIDATION FAILED =====")

print("=" * 60)