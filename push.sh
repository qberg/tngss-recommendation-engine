#!/bin/bash
set -e

echo "+++ TNGSS AI LAYER FASTAPI APP DEPLOYMENT (AUTOMATED) +++"
echo "Starting automated deployment at $(date)"

SERVER_IP="13.127.19.137"
SERVER_USER="ubuntu"
SSH_KEY="$HOME/.ssh/tngss_pem.pem"
SERVER_PATH="/opt/fastapi-app"
LOCAL_PATH="."
SUPERVISOR_APP_NAME="fastapi-workers:fastapi"

echo -e "${YELLOW}>> Step 1: Syncing source code to server...${NC}"
rsync -avz --delete --progress \
	-e "ssh -i $SSH_KEY" \
	--exclude 'venv' \
	--exclude '__pycache__' \
	--exclude '.git' \
	--exclude '.env.local' \
	--exclude '*.log' \
	--exclude 'logs/' \
	--exclude 'deploy*.sh' \
	"$LOCAL_PATH/" "$SERVER_USER@$SERVER_IP:$SERVER_PATH/"

echo -e "${GREEN}[OK] Source code synced successfully${NC}"
