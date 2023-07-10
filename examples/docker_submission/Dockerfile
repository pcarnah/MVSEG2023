FROM condaforge/mambaforge

# Set working directory for the project
WORKDIR /app

# Create Conda environment from the YAML file
COPY env_docker.yml .
RUN mamba env create -f env_docker.yml
 
# Override default shell and use bash
SHELL ["conda", "run", "-n", "monai10", "/bin/bash", "-c"]
 
# Activate Conda environment and check if it is working properly
RUN echo "Making sure PyTorch is installed correctly..."
RUN python -c "import torch"
 
# Python program to run in the container
COPY run_model.py .
COPY final_model_mvseg.pt model.md
ENTRYPOINT ["conda", "run", "-n", "monai10", "python", "run_model.py"]
