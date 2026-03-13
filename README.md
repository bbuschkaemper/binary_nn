# binary_nn

## Setup

### Devcontainer

1. On the devcontainer host machine, create a named volume for persistent data storage called `data`.
For example, to create a volume that binds to a local directory `/path/to/local/data`, run:

    ```bash
    docker volume create --name data --opt type=none --opt device=/path/to/local/data --opt o=bind
    ```

2. Create a `.env` file in the repository directory containing the keys of `.env.example`

3. (Optional, for GPU support) Install the NVIDIA Container Toolkit using the script `setup-gpu-host.sh`.

4. Open the workspace in a devcontainer using VSCode.
