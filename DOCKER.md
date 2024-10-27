# Docker Setup

Leverage Docker to create a consistent and reproducible environment for running **LayerSkip** without requiring GPU support. This setup ensures that all dependencies are managed efficiently and secrets like the HuggingFace token are handled securely.

## Prerequisites

1. **Docker Installed**: Ensure Docker is installed on your machine. [Get Docker](https://docs.docker.com/get-docker/)

2. **HuggingFace Token**: Obtain your HuggingFace access token. [HuggingFace Tokens](https://huggingface.co/docs/hub/security-tokens)

### 1. Building the Docker Image

Follow these steps to build the Docker image for **LayerSkip**:

1. **Clone the Repository**:

   ```bash
    git clone git@github.com:facebookresearch/LayerSkip.git
    cd LayerSkip
    ```

2. **Ensure Dockerfile and Entrypoint Script are Present**:  
   Make sure the `Dockerfile`, `entrypoint.sh`, and `.dockerignore` are located in the root directory of your project as shown below:
      
    ```
    .
    ├── Dockerfile
    ├── entrypoint.sh
    ├── .dockerignore
    ├── arguments.py
    ├── benchmark.py
    ├── CODE_OF_CONDUCT.md
    ├── CONTRIBUTING.md
    ├── correctness.py
    ├── data.py
    ├── DOCKER.md
    ├── eval.py
    ├── generate.py
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── self_speculation
    │   ├── autoregressive_generator.py
    │   ├── generator_base.py
    │   ├── llama_model_utils.py
    │   ├── self_speculation_generator.py
    │   └── speculative_streamer.py
    ├── sweep.py
    └── utils.py
    ```

3. **Build the Docker Image**:
    ```bash
    docker build -t layerskip:latest .
    ```
    - **Explanation**:
        - `-t layerskip:latest`: Tags the image as `layerskip` with the `latest` tag.
        - `.`: Specifies the current directory as the build context.

    **Note**: The build process may take several minutes as it installs all dependencies.

### 2. Running the Docker Container

Once the Docker image is built, you can run your **LayerSkip** scripts inside the container. Below are instructions and examples for executing different scripts.

#### Basic Command Structure

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    layerskip:latest \
    python your_script.py --help
```

- **Flags and Arguments**:
    - `-it`: Runs the container in interactive mode with a pseudo-TTY.
    - `--rm`: Automatically removes the container when it exits.
    - `-e HUGGINGFACE_TOKEN=your_huggingface_token_here`: Sets the `HUGGINGFACE_TOKEN` environment variable inside the container. **Replace** `your_huggingface_token_here` with your actual token.
    - `layerskip:latest`: Specifies the Docker image to use.
    - `python your_script.py --help`: The command to execute inside the container. Replace `your_script.py --help` with your desired script and arguments.

#### Examples

##### 2.1. Generate Text

Run the `generate.py` script in interactive mode using regular autoregressive decoding:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    layerskip:latest \
    python generate.py --model facebook/layerskip-llama2-7B \
                        --sample True \
                        --max_steps 512
```

To observe speedup with self-speculative decoding, specify `--exit_layer` and `--num_speculations`:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    layerskip:latest \
    python generate.py --model facebook/layerskip-llama2-7B \
                        --sample True \
                        --max_steps 512 \
                        --generation_strategy self_speculative \
                        --exit_layer 8 \
                        --num_speculations 6
```

##### 2.2. Benchmark

Benchmark the model on a specific dataset:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/logs:/app/logs \
    layerskip:latest \
    python benchmark.py --model facebook/layerskip-llama2-7B \
                         --dataset cnn_dm_summarization \
                         --num_samples 100 \
                         --generation_strategy self_speculative \
                         --exit_layer 8 \
                         --num_speculations 6 \
                         --output_dir /app/logs
```

- **Explanation**:
    - `-v /path/on/host/logs:/app/logs`: Mounts the host directory `/path/on/host/logs` to the container's `/app/logs` directory, ensuring that logs are saved on the host.

##### 2.3. Evaluate

Evaluate the model using the Eleuther Language Model Evaluation Harness:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/logs:/app/logs \
    layerskip:latest \
    python eval.py --model facebook/layerskip-llama2-7B \
                    --tasks gsm8k \
                    --limit 10 \
                    --generation_strategy self_speculative \
                    --exit_layer 8 \
                    --num_speculations 6 \
                    --output_dir /app/logs
```

##### 2.4. Sweep

Perform a sweep over different `exit_layer` and `num_speculations` hyperparameters:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/sweep:/app/sweep \
    layerskip:latest \
    python sweep.py --model facebook/layerskip-llama2-7B \
                     --dataset human_eval \
                     --generation_strategy self_speculative \
                     --num_samples 150 \
                     --max_steps 256 \
                     --output_dir /app/sweep \
                     --sample False
```

##### 2.5. Correctness Check

Verify the correctness of self-speculative decoding:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/correctness:/app/correctness \
    layerskip:latest \
    python correctness.py --model facebook/layerskip-llama2-7B \
                            --dataset human_eval \
                            --generation_strategy self_speculative \
                            --num_speculations 6 \
                            --exit_layer 4 \
                            --num_samples 10 \
                            --sample False \
                            --output_dir /app/correctness
```

- **Explanation**:
    - `-v /path/on/host/correctness:/app/correctness`: Mounts the host directory `/path/on/host/correctness` to the container's `/app/correctness` directory, ensuring that correctness metrics are saved on the host.

### 3. Handling Output and Logs

To persist outputs and logs generated by your scripts, mount host directories to the corresponding directories inside the Docker container using the `-v` flag. This ensures that all results are stored on your host machine and are not lost when the container is removed.

**Example: Mounting Logs Directory**

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/logs:/app/logs \
    layerskip:latest \
    python benchmark.py --model facebook/layerskip-llama2-7B \
                         --dataset human_eval \
                         --num_samples 100 \
                         --generation_strategy self_speculative \
                         --exit_layer 8 \
                         --num_speculations 6 \
                         --output_dir /app/logs
```

### 4. Environment Variables Security

**Important**: Never hardcode sensitive information like the HuggingFace token directly into the `Dockerfile` or your scripts. Always pass them securely at runtime using environment variables.

#### Passing Environment Variables Securely

When running the Docker container, pass the HuggingFace token using the `-e` flag:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    layerskip:latest \
    python generate.py --help
```

#### Using Docker Secrets (Advanced)

For enhanced security, especially in production environments, consider using Docker secrets to manage sensitive data. This approach is more secure than passing environment variables directly.

**Example Using Docker Secrets (Docker Swarm):**

1. **Create a Secret:**

    ```bash
    echo "your_huggingface_token_here" | docker secret create huggingface_token -
    ```

2. **Update `entrypoint.sh` to Read the Secret:**

    ```bash
    #!/bin/bash
    # entrypoint.sh

    # Activate the Conda environment
    source /opt/conda/etc/profile.d/conda.sh
    conda activate layer_skip

    # Read HuggingFace token from Docker secret
    export HUGGINGFACE_TOKEN=$(cat /run/secrets/huggingface_token)

    # Execute the passed command
    exec "$@"
    ```

3. **Deploy the Service with the Secret:**

    ```bash
    docker service create --name layerskip_service \
        --secret huggingface_token \
        layerskip:latest \
        python generate.py --help
    ```

**Note**: Docker secrets are primarily designed for use with Docker Swarm. If you're not using Swarm, passing environment variables securely as shown earlier is the recommended approach.

### 5. Additional Recommendations

#### 5.1. Caching HuggingFace Models

To avoid re-downloading models every time you run the container, mount the HuggingFace cache directory:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/huggingface_cache:/root/.cache/huggingface \
    layerskip:latest \
    python generate.py --help
```

- **Explanation**:
    - `-v /path/on/host/huggingface_cache:/root/.cache/huggingface`: Mounts the host directory to the container's HuggingFace cache directory, speeding up model loading times.

#### 5.2. Optimizing Docker Layers

- **Leverage Docker Caching**: By copying `requirements.txt` and installing dependencies before copying the rest of the code, Docker can cache these layers and speed up subsequent builds when dependencies haven't changed.

- **Combine RUN Commands**: Reduce the number of Docker layers by combining multiple `RUN` commands where possible.

    **Example:**
    ```dockerfile
    RUN conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch -y && \
        pip install --upgrade pip && \
        pip install --no-cache-dir -r /app/requirements.txt
    ```

#### 5.3. Port Exposures

If any of your scripts run a web server or need specific ports exposed, add the `EXPOSE` directive in the Dockerfile and map the ports when running the container.

**Example: Exposing Port 8000**

1. **Update Dockerfile:**

    ```dockerfile
    EXPOSE 8000
    ```

2. **Run the Container with Port Mapping:**

    ```bash
    docker run -it --rm \
        -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
        -p 8000:8000 \
        layerskip:latest \
        python your_web_server_script.py
    ```

---

## Testing the Docker Container Without a HuggingFace Token

### 1. Verify Python and PyTorch Installation

First, ensure that Python and PyTorch are correctly installed in your Docker image.

```bash
docker run -it --rm layerskip:latest python -c "import torch; print(torch.__version__)"
```

**Expected Output:**

```
2.2.1
```

This confirms that PyTorch version 2.2.1 is installed.

### 2. Check Available Scripts and Help Messages

Ensure that your scripts are accessible and functioning as expected by checking their help messages. This helps verify that all dependencies are correctly installed.

- **Generate Script:**

    ```bash
    docker run -it --rm layerskip:latest python generate.py --help
    ```

    **Expected Output:**

    Displays the help message for `generate.py`, listing available arguments and usage instructions.

- **Benchmark Script:**

    ```bash
    docker run -it --rm layerskip:latest python benchmark.py --help
    ```

    **Expected Output:**

    Displays the help message for `benchmark.py`.

- **Evaluate Script:**

    ```bash
    docker run -it --rm layerskip:latest python eval.py --help
    ```

    **Expected Output:**

    Displays the help message for `eval.py`.

- **Sweep Script:**

    ```bash
    docker run -it --rm layerskip:latest python sweep.py --help
    ```

    **Expected Output:**

    Displays the help message for `sweep.py`.

- **Correctness Script:**

    ```bash
    docker run -it --rm layerskip:latest python correctness.py --help
    ```

    **Expected Output:**

    Displays the help message for `correctness.py`.

### 3. Check Environment Variables

Ensure that environment variables are correctly set up within the Docker container.

```bash
docker run -it --rm layerskip:latest bash -c 'echo $HUGGINGFACE_TOKEN'
```

**Expected Output:**

If you haven't set the `HUGGINGFACE_TOKEN` when running the container, this will likely be empty or show a default placeholder. This is expected since you don't have the token yet.

### 4. Test with a Dummy HuggingFace Token (Optional)

Even without a valid token, you can pass a dummy value to ensure that environment variables are handled correctly. This won't allow you to access models, but it confirms that the token is being set.

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=dummy_token \
    layerskip:latest \
    python generate.py --help
```

**Expected Output:**

Displays the help message for `generate.py` without attempting to access any models, thereby avoiding authentication errors.

## Running Scripts Inside the Docker Container Once You Have a HuggingFace Token

Once you obtain your HuggingFace token, you can run your scripts with proper authentication. Here's how to proceed:

### 1. Obtain Your HuggingFace Token

Follow these steps to get your HuggingFace token:

1. **Create a HuggingFace Account**: If you haven't already, create an account on [HuggingFace](https://huggingface.co/).

2. **Generate a Token**:
    - Navigate to your account settings.
    - Go to the "Access Tokens" section.
    - Click on "New Token" and follow the prompts to generate a token.
    - **Note**: Keep your token secure and **do not** share it publicly.

### 2. Run Your Scripts with the Token

Replace `your_huggingface_token_here` with your actual token in the following commands.

#### 2.1. Generate Text with Self-Speculative Decoding

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    layerskip:latest \
    python generate.py --model facebook/layerskip-llama2-7B \
                        --sample True \
                        --max_steps 512 \
                        --generation_strategy self_speculative \
                        --exit_layer 8 \
                        --num_speculations 6
```

#### 2.2. Benchmarking

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/logs:/app/logs \
    layerskip:latest \
    python benchmark.py --model facebook/layerskip-llama2-7B \
                         --dataset cnn_dm_summarization \
                         --num_samples 100 \
                         --generation_strategy self_speculative \
                         --exit_layer 8 \
                         --num_speculations 6 \
                         --output_dir /app/logs
```

- **Explanation:**
    - `-v /path/on/host/logs:/app/logs`: Mounts the host directory `/path/on/host/logs` to the container's `/app/logs` directory, ensuring logs are saved on the host.

#### 2.3. Evaluate

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/logs:/app/logs \
    layerskip:latest \
    python eval.py --model facebook/layerskip-llama2-7B \
                    --tasks gsm8k \
                    --limit 10 \
                    --generation_strategy self_speculative \
                    --exit_layer 8 \
                    --num_speculations 6 \
                    --output_dir /app/logs
```

#### 2.4. Sweep

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/sweep:/app/sweep \
    layerskip:latest \
    python sweep.py --model facebook/layerskip-llama2-7B \
                     --dataset human_eval \
                     --generation_strategy self_speculative \
                     --num_samples 150 \
                     --max_steps 256 \
                     --output_dir /app/sweep \
                     --sample False
```

#### 2.5. Correctness Check

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/correctness:/app/correctness \
    layerskip:latest \
    python correctness.py --model facebook/layerskip-llama2-7B \
                            --dataset human_eval \
                            --generation_strategy self_speculative \
                            --num_speculations 6 \
                            --exit_layer 4 \
                            --num_samples 10 \
                            --sample False \
                            --output_dir /app/correctness
```

## Summary

- **Without a HuggingFace Token**:
    - Run help commands to ensure scripts are accessible.
    - Verify Python and PyTorch installations.
    - Check environment variable settings.

- **With a HuggingFace Token**:
    - Run your scripts as intended by passing the token via the `-e` flag.
    - Mount host directories for logs and outputs as needed.

## Additional Recommendations

### 1. Caching HuggingFace Models

To avoid re-downloading models every time you run the container, mount the HuggingFace cache directory:

```bash
docker run -it --rm \
    -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
    -v /path/on/host/huggingface_cache:/root/.cache/huggingface \
    layerskip:latest \
    python generate.py --help
```

- **Explanation**:
    - `-v /path/on/host/huggingface_cache:/root/.cache/huggingface`: Mounts the host directory to the container's HuggingFace cache directory, speeding up model loading times.

### 2. Environment Variables Security

- **Avoid Hardcoding Tokens**:  
  Never hardcode your HuggingFace tokens in the Dockerfile or scripts. Always pass them as environment variables at runtime.

- **Using Docker Secrets (Advanced)**:  
  For enhanced security, especially in production environments, consider using Docker secrets or other secret management tools.

    **Example Using Docker Secrets (Docker Swarm):**

    1. **Create a Secret:**

        ```bash
        echo "your_huggingface_token_here" | docker secret create huggingface_token -
        ```

    2. **Update `entrypoint.sh` to Read the Secret:**

        ```bash
        #!/bin/bash
        # entrypoint.sh

        # Activate the Conda environment
        source /opt/conda/etc/profile.d/conda.sh
        conda activate layer_skip

        # Read HuggingFace token from Docker secret
        export HUGGINGFACE_TOKEN=$(cat /run/secrets/huggingface_token)

        # Execute the passed command
        exec "$@"
        ```

    3. **Deploy the Service with the Secret:**

        ```bash
        docker service create --name layerskip_service \
            --secret huggingface_token \
            layerskip:latest \
            python generate.py --help
        ```

    **Note**: Docker secrets are primarily designed for use with Docker Swarm. If you're not using Swarm, passing environment variables securely as shown earlier is the recommended approach.

### 3. Optimizing Docker Layers

- **Leverage Docker Caching**:  
  By copying `requirements.txt` and installing dependencies before copying the rest of the code, Docker can cache these layers and speed up subsequent builds when dependencies haven't changed.

- **Combine `RUN` Commands**:  
  Reduce the number of Docker layers by combining multiple `RUN` commands where possible.

    **Example:**

    ```dockerfile
    RUN conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch -y && \
        pip install --upgrade pip && \
        pip install --no-cache-dir -r /app/requirements.txt
    ```

### 4. Port Exposures

If any of your scripts run a web server or need specific ports exposed, add the `EXPOSE` directive in the Dockerfile and map the ports when running the container.

**Example: Exposing Port 8000**

1. **Update Dockerfile:**

    ```dockerfile
    EXPOSE 8000
    ```

2. **Run the Container with Port Mapping:**

    ```bash
    docker run -it --rm \
        -e HUGGINGFACE_TOKEN=your_huggingface_token_here \
        -p 8000:8000 \
        layerskip:latest \
        python your_web_server_script.py
    ```

