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

echo -e "${YELLOW}>> Step 2: Restarting FastAPI application via Supervisor...${NC}"
ssh -i "$SSH_KEY" "$SERVER_USER@$SERVER_IP" <<'ENDSSH'
    set -e
    
    echo "Restarting fastapi-workers:fastapi..."
    sudo supervisorctl restart fastapi-workers:fastapi
    
    echo "Waiting for application to start..."
    sleep 3
    
    echo "Checking status..."
    sudo supervisorctl status fastapi-workers:fastapi
ENDSSH

if [ $? -eq 0 ]; then
	echo -e "${GREEN}[OK] Application restarted successfully${NC}"
	echo -e "${GREEN}+++ DEPLOYMENT COMPLETED at $(date) +++${NC}"
else
	echo -e "${RED}[FAILED] Application restart failed${NC}"
	exit 1
fi
