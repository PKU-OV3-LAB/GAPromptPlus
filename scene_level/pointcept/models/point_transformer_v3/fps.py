import torch

def fps(x, ratio=None, max_num_samples=None, batch=None, random_start=True):
    if max_num_samples is None and ratio is None:
        raise ValueError("ratio or max_num_samples must be provided")

    if batch is None:
        return _fps_single(x, ratio, max_num_samples, random_start)
    else:
        return _fps_batch(x, ratio, max_num_samples, batch, random_start)


def _fps_single(x, ratio, max_num_samples, random_start):
    n_points = x.size(0)

    if max_num_samples is not None:
        n_samples = min(max_num_samples, n_points)
    else:
        n_samples = min(int(ratio * n_points), n_points)

    if n_samples == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)
    if n_samples == n_points:
        return torch.arange(n_points, dtype=torch.long, device=x.device)

    indices = torch.zeros(n_samples, dtype=torch.long, device=x.device)

    if random_start:
        start_idx = torch.randint(0, n_points, (1,), device=x.device)
    else:
        start_idx = torch.tensor([0], device=x.device)

    indices[0] = start_idx

    distances = torch.norm(x - x[start_idx], dim=1, p=2)

    for i in range(1, n_samples):
        farthest_idx = torch.argmax(distances)
        indices[i] = farthest_idx

        new_distances = torch.norm(x - x[farthest_idx], dim=1, p=2)
        distances = torch.min(distances, new_distances)

    return indices


def _fps_batch(x, ratio, max_num_samples, batch, random_start):
    batch_size = batch.max().item() + 1
    all_indices = []

    for i in range(batch_size):
        mask = (batch == i)
        x_batch = x[mask]

        n_points_batch = x_batch.size(0)

        if max_num_samples is not None:
            n_samples_batch = min(max_num_samples, n_points_batch)
        else:
            n_samples_batch = min(int(ratio * n_points_batch), n_points_batch)

        if n_samples_batch == 0:
            continue

        indices_batch = _fps_single(x_batch, None, n_samples_batch, random_start)

        global_indices = torch.where(mask)[0][indices_batch]
        all_indices.append(global_indices)

    if len(all_indices) == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)

    return torch.cat(all_indices)
