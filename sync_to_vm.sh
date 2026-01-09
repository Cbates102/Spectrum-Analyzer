#!/bin/bash

# Sync Traffic Monitoring repository to VM with temporal-aware changes

VM_IP="209.20.156.138"
VM_USER="ubuntu"
SSH_KEY="~/.ssh/lambda-cloud-ssh.pem"
REMOTE_PATH="/home/ubuntu/RadDet/ems-yolo"
LOCAL_PATH="."

echo "Syncing ems-yolo files to VM..."

# Create directory structure on VM if it doesn't exist
ssh -i $SSH_KEY $VM_USER@$VM_IP "mkdir -p $REMOTE_PATH"

# Sync files excluding large directories and cache
rsync -avz --update \
  --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.git' \
  --exclude '.vscode' \
  --exclude '.DS_Store' \
  --exclude '.gitignore' \
  --exclude '.dockerignore' \
  --exclude 'data/raddet_gt' \
  --exclude 'runs/detect' \
  --exclude 'runs/train' \
  --exclude 'runs/val' \
  --exclude 'runs/test' \
  --exclude 'runs/speed' \
  --exclude 'runs/study' \
  --exclude 'runs/logs' \
  --exclude 'runs/results' \
  -e "ssh -i $SSH_KEY" \
  $LOCAL_PATH/ \
  $VM_USER@$VM_IP:$REMOTE_PATH/

echo "Sync complete!"