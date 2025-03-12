#!/bin/bash

# Remote server details
REMOTE_HOST="172.31.26.4"
REMOTE_USER="ec2-user"
APP_DIR="/home/ec2-user/csd-analyser-deployment"
SSH_KEY="/Users/ldominc/.id_rsa"
SSH_OPTS="-i $SSH_KEY"

# Function to display usage information
usage() {
    echo "Usage: $0 [-c|--cleanup]"
    echo "  -c, --cleanup    Clean up the deployment (stop service and remove files)"
    echo "  -h, --help      Display this help message"
    exit 1
}

# Function to calculate MD5 hash of requirements.txt
get_requirements_hash() {
    if [ -f "requirements.txt" ]; then
        md5sum requirements.txt | cut -d' ' -f1
    else
        echo ""
    fi
}

# Function to check if requirements have changed
check_requirements_changed() {
    local current_hash=$(get_requirements_hash)
    local remote_hash_file="$APP_DIR/.requirements_hash"
    
    # If no current requirements file exists
    if [ -z "$current_hash" ]; then
        echo "No requirements.txt found"
        return 1
    fi
    
    # Check if hash file exists on remote server
    if ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "[ -f $remote_hash_file ]"; then
        local stored_hash=$(ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "cat $remote_hash_file")
        if [ "$current_hash" != "$stored_hash" ]; then
            echo "Requirements have changed. Old hash: $stored_hash, New hash: $current_hash"
            return 0
        else
            echo "Requirements unchanged"
            return 1
        fi
    else
        echo "No previous requirements hash found"
        return 0
    fi
}

# Function to update requirements hash on remote server
update_requirements_hash() {
    local current_hash=$(get_requirements_hash)
    if [ ! -z "$current_hash" ]; then
        echo "$current_hash" | ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "cat > $APP_DIR/.requirements_hash"
    fi
}

# Function to perform cleanup
cleanup() {
    echo "Cleaning up deployment..."
    
    # Stop and disable the service
    echo "Stopping and disabling service..."
    ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "\
        systemctl --user stop csd-analyser && \
        systemctl --user disable csd-analyser && \
        rm -f ~/.config/systemd/user/csd-analyser.service && \
        systemctl --user daemon-reload"
    
    # Remove the application directory
    echo "Removing application directory..."
    ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "rm -rf $APP_DIR"
    
    echo "Cleanup completed!"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cleanup)
            cleanup
            ;;
        -h|--help)
            usage
            ;;
        *)
            usage
            ;;
    esac
    shift
done

# Create the application directory on the remote server
ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "mkdir -p $APP_DIR"

# Copy necessary files to the remote server
echo "Copying files to remote server..."
rsync -avz -e "ssh $SSH_OPTS" \
          --exclude '.git' \
          --exclude '.venv' \
          --exclude '__pycache__' \
          --exclude '*.pyc' \
          --exclude '.pytest_cache' \
          --exclude 'htmlcov' \
          --exclude '*.csv' \
          --exclude '*.xlsx' \
          ./ $REMOTE_USER@$REMOTE_HOST:$APP_DIR/

# Setup the environment and install dependencies on the remote server
echo "Setting up environment on remote server..."
ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "cd $APP_DIR && \
    if [ ! -d '.venv' ]; then \
        echo 'Creating new virtual environment...' && \
        python3 -m venv .venv; \
    fi"

# Check if requirements have changed and install if necessary
if check_requirements_changed; then
    echo "Installing updated requirements..."
    ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "cd $APP_DIR && \
        source .venv/bin/activate && \
        pip install -r requirements.txt && \
        python3 -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"averaged_perceptron_tagger\")'"
    
    # Update the hash after successful installation
    update_requirements_hash
else
    echo "Skipping pip install as requirements haven't changed"
fi

# Create .streamlit directory if it doesn't exist
ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "mkdir -p $APP_DIR/.streamlit"

# Copy the secrets.toml file separately (if it exists)
if [ -f ".streamlit/secrets.toml" ]; then
    echo "Copying secrets file..."
    scp $SSH_OPTS .streamlit/secrets.toml $REMOTE_USER@$REMOTE_HOST:$APP_DIR/.streamlit/
fi

# Create user's systemd directory if it doesn't exist
echo "Setting up user service directory..."
ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "mkdir -p ~/.config/systemd/user/"

# Create the user service file for running the app
echo "Creating user service..."
cat << EOF | ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "tee ~/.config/systemd/user/csd-analyser.service"
[Unit]
Description=CSD Analyser Streamlit App
After=network.target

[Service]
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/.venv/bin"
ExecStart=$APP_DIR/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=default.target
EOF

# Enable and start the user service
echo "Starting the service..."
ssh $SSH_OPTS $REMOTE_USER@$REMOTE_HOST "\
    systemctl --user daemon-reload && \
    systemctl --user enable csd-analyser && \
    systemctl --user start csd-analyser && \
    loginctl enable-linger $REMOTE_USER"  # Allow service to run even when user logs out

echo "Deployment completed! The app should be running at http://$REMOTE_HOST:8501"
echo
echo "Available commands:"
echo "systemctl --user status csd-analyser   # Check status"
echo "systemctl --user restart csd-analyser  # Restart"
echo "systemctl --user stop csd-analyser     # Stop"
echo
echo "To clean up the deployment, run: $0 --cleanup" 