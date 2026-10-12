#!/bin/bash
# Bootstrap script for an EC2 instance hosting the FastAPI service + Streamlit app.
# Paste into EC2 launch wizard -> Advanced details -> User data.
# Open ports 8501 (Streamlit) and 8000 (FastAPI) in the security group.

set -eu

# -------- EDIT THESE -----------------------------------------------------
GIT_REPO="https://github.com/virgiequeena/AWS-Churn-Prediction.git"
SUBFOLDER=""
APP_FILE="app_fastapi.py"
# -------------------------------------------------------------------------

APP_DIR="/opt/churn-app"
VENV_DIR="/opt/streamlit-venv"

if [ -z "$SUBFOLDER" ]; then
  APP_PATH="$APP_DIR"
else
  APP_PATH="$APP_DIR/$SUBFOLDER"
fi

dnf update -y
dnf install -y python3.11 python3.11-pip git

git clone "$GIT_REPO" "$APP_DIR"
chown -R ec2-user:ec2-user "$APP_DIR"

if [ ! -f "$APP_PATH/$APP_FILE" ]; then
  echo "FATAL: $APP_PATH/$APP_FILE not found."
  find "$APP_DIR" -maxdepth 4 -type f | head -40
  exit 1
fi

# Use a venv to avoid conflicts with rpm-managed system Python packages.
python3.11 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$APP_PATH/requirements.txt"

cat >/etc/systemd/system/fastapi.service <<EOF
[Unit]
Description=FastAPI Churn Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_PATH
ExecStart=$VENV_DIR/bin/uvicorn fastAPI:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_PATH
Environment=API_URL=http://127.0.0.1:8000
ExecStart=$VENV_DIR/bin/streamlit run $APP_FILE \\
  --server.address 0.0.0.0 \\
  --server.port 8501 \\
  --server.headless true \\
  --server.enableCORS false \\
  --server.enableXsrfProtection false
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now fastapi.service
systemctl enable --now streamlit.service

sleep 5
if systemctl is-active --quiet fastapi && systemctl is-active --quiet streamlit; then
  touch "$APP_DIR/.userdata-success"
  chown ec2-user:ec2-user "$APP_DIR/.userdata-success"
else
  echo "FATAL: fastapi or streamlit service failed to start."
  journalctl -u fastapi -n 30 --no-pager || true
  journalctl -u streamlit -n 30 --no-pager || true
  exit 1
fi
