from torch.utils.data import DataLoader, Dataset
import torch

# Default number of workers (can be tuned)
DEFAULT_NUM_WORKERS = torch.get_num_threads() // 2 if torch.get_num_threads() > 1 else 0


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = True,
    **kwargs,  # Allow passing other DataLoader arguments
) -> DataLoader:
    """Creates a PyTorch DataLoader for the RNA3DDataset.

    Args:
        dataset (Dataset): An instance of RNA3DDataset (or any PyTorch Dataset).
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
                           0 means that the data will be loaded in the main process.
        pin_memory (bool): If True, the data loader will copy Tensors
                           into device/CUDA pinned memory before returning them.
        **kwargs: Additional keyword arguments to pass to the DataLoader constructor.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("'dataset' must be an instance of torch.utils.data.Dataset")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("'batch_size' must be a positive integer")

    # Note: Using default collate_fn for now. This will create batches where
    # each field in the dataset's output dictionary becomes a list (or stacked tensor
    # if possible) in the batch dictionary. Variable length sequences/coordinates
    # will likely result in lists of tensors/strings, requiring handling in the model
    # or training loop.
    # If padding is needed, a custom collate_fn should be implemented and passed.

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers
        > 0,  # Keep workers alive between epochs if using >0
        **kwargs,
    )

    print(
        f"Created DataLoader with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}"
    )
    return dataloader


# Example Usage (optional):
if __name__ == "__main__":
    from pathlib import Path

    # Assuming datasets is in the same package or accessible
    try:
        from datasets import RNA3DDataset

        # Adjust the path to where your data actually is
        DATA_ROOT = Path("../data/stanford-rna-3d-folding")  # Example path adjustment

        if DATA_ROOT.exists():
            print(f"Attempting to load data from: {DATA_ROOT}")
            try:
                train_dataset = RNA3DDataset(data_dir=DATA_ROOT, split="train")
                print(f"Loaded training dataset with {len(train_dataset)} samples.")

                if len(train_dataset) > 0:
                    # Create DataLoader
                    train_loader = get_dataloader(
                        train_dataset, batch_size=4, shuffle=True
                    )
                    print(f"DataLoader created: {train_loader}")

                    # Iterate over one batch to see the structure
                    print("\nIterating over one batch:")
                    try:
                        first_batch = next(iter(train_loader))
                        print("Batch keys:", first_batch.keys())
                        print("Target IDs in batch:", first_batch.get("target_id"))
                        # Sequences will be a list of strings
                        print(
                            "Sequences in batch (lengths):",
                            [len(s) for s in first_batch.get("sequence", [])],
                        )
                        # Coordinates will likely be a list of Tensors (if present and lengths vary)
                        if "coordinates" in first_batch:
                            print(
                                "Coordinates in batch (types/shapes):",
                                [
                                    (
                                        type(c),
                                        (
                                            c.shape
                                            if isinstance(c, torch.Tensor)
                                            else None
                                        ),
                                    )
                                    for c in first_batch["coordinates"]
                                ],
                            )
                        else:
                            print(
                                "Coordinates key not present in this batch (check dataset filtering/labels)"
                            )

                    except StopIteration:
                        print("Could not retrieve a batch from the DataLoader.")
                    except Exception as e:
                        print(f"Error iterating over DataLoader: {e}")

            except FileNotFoundError as e:
                print(f"\nError: Could not find necessary files in {DATA_ROOT}. {e}")
            except Exception as e:
                print(f"\nAn unexpected error occurred during dataset loading: {e}")

        else:
            print(
                f"Error: Example data directory {DATA_ROOT} not found. Cannot run example usage."
            )

    except ImportError:
        print("Could not import RNA3DDataset. Ensure it's in the correct path.")
    except Exception as e:
        print(f"An unexpected error occurred during example execution: {e}")
