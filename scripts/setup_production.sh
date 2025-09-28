#!/bin/bash

# AI Benchmark Suite - Production Setup Script
# This script sets up the production environment with all necessary components

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root for security reasons"
   exit 1
fi

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="/opt/ai-benchmark-backups"
LOG_DIR="/var/log/ai-benchmark"

log "Starting AI Benchmark Suite Production Setup"
log "Project directory: $PROJECT_DIR"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install Git first."
        exit 1
    fi

    log_success "All prerequisites are satisfied"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."

    # Create backup directory (requires sudo)
    if [ ! -d "$BACKUP_DIR" ]; then
        sudo mkdir -p "$BACKUP_DIR"
        sudo chown $USER:$USER "$BACKUP_DIR"
        log_success "Created backup directory: $BACKUP_DIR"
    fi

    # Create log directory (requires sudo)
    if [ ! -d "$LOG_DIR" ]; then
        sudo mkdir -p "$LOG_DIR"
        sudo chown $USER:$USER "$LOG_DIR"
        log_success "Created log directory: $LOG_DIR"
    fi

    # Create project directories
    mkdir -p "$PROJECT_DIR/cache"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/backups"
    mkdir -p "$PROJECT_DIR/config"

    log_success "All directories created"
}

# Setup environment configuration
setup_environment() {
    log "Setting up environment configuration..."

    # Create production environment file if it doesn't exist
    if [ ! -f "$PROJECT_DIR/.env.production" ]; then
        log "Creating production environment file..."
        cat > "$PROJECT_DIR/.env.production" << EOF
# Environment
ENVIRONMENT=production
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=$(openssl rand -hex 32)

# Database Configuration
POSTGRES_DB=ai_benchmark_prod
POSTGRES_USER=benchmark_user
POSTGRES_PASSWORD=$(openssl rand -hex 16)
DATABASE_URL=postgresql://benchmark_user:$(openssl rand -hex 16)@postgres:5432/ai_benchmark_prod

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 12)

# Resource Limits
MAX_PARALLEL_EVALUATIONS=8
MEMORY_LIMIT_GB=30
CACHE_SIZE_GB=10

# Security
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
CORS_ORIGINS=https://your-domain.com

# Model APIs (add your keys)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
HUGGINGFACE_TOKEN=your-huggingface-token-here
EOF
        log_success "Created .env.production file"
        log_warning "Please update .env.production with your actual API keys and domain"
    else
        log_warning ".env.production already exists, skipping creation"
    fi
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."

    cd "$PROJECT_DIR"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Created Python virtual environment"
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Installed Python dependencies"
    else
        log_warning "requirements.txt not found, installing core dependencies"
        pip install fastapi uvicorn redis psycopg2-binary sqlalchemy streamlit plotly pandas numpy
    fi

    deactivate
}

# Setup Docker environment
setup_docker() {
    log "Setting up Docker environment..."

    cd "$PROJECT_DIR"

    # Pull required Docker images
    log "Pulling Docker images..."
    docker-compose -f docker-compose.production.yml pull

    # Build custom images if needed
    if [ -f "Dockerfile" ]; then
        log "Building custom Docker image..."
        docker build -t ai-benchmark-suite .
    fi

    log_success "Docker environment ready"
}

# Initialize database
initialize_database() {
    log "Initializing database..."

    cd "$PROJECT_DIR"

    # Start database service
    docker-compose -f docker-compose.production.yml up -d postgres

    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 10

    # Run database migrations/setup
    if [ -f "scripts/init_database.py" ]; then
        source venv/bin/activate
        python scripts/init_database.py
        deactivate
        log_success "Database initialized"
    else
        log_warning "Database initialization script not found"
    fi
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."

    cd "$PROJECT_DIR"

    # Create Prometheus configuration if it doesn't exist
    if [ ! -f "config/prometheus.yml" ]; then
        mkdir -p config
        cat > config/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-benchmark-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF
        log_success "Created Prometheus configuration"
    fi

    # Create Grafana dashboard configuration
    if [ ! -f "config/grafana-dashboards.json" ]; then
        cat > config/grafana-dashboards.json << EOF
{
  "dashboard": {
    "title": "AI Benchmark Suite - Production Dashboard",
    "panels": [
      {
        "title": "API Requests per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[1m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Evaluation Success Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(evaluations_successful_total[5m]) / rate(evaluations_total[5m])",
            "legendFormat": "Success Rate"
          }
        ]
      },
      {
        "title": "System Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU Usage"
          },
          {
            "expr": "memory_usage_percent",
            "legendFormat": "Memory Usage"
          }
        ]
      }
    ]
  }
}
EOF
        log_success "Created Grafana dashboard configuration"
    fi
}

# Setup SSL/TLS (optional)
setup_ssl() {
    log "Setting up SSL/TLS certificates..."

    # Check if SSL certificates exist
    if [ ! -f "config/ssl/server.crt" ] || [ ! -f "config/ssl/server.key" ]; then
        log_warning "SSL certificates not found. Generating self-signed certificates for development..."

        mkdir -p config/ssl

        # Generate self-signed certificate
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout config/ssl/server.key \
            -out config/ssl/server.crt \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

        log_success "Generated self-signed SSL certificates"
        log_warning "For production, replace with proper SSL certificates from a CA"
    else
        log_success "SSL certificates already exist"
    fi
}

# Setup backup system
setup_backup_system() {
    log "Setting up backup system..."

    # Create backup script
    cat > "$PROJECT_DIR/scripts/backup_production.sh" << 'EOF'
#!/bin/bash

# Production Backup Script
set -e

BACKUP_DIR="/opt/ai-benchmark-backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="ai-benchmark-backup-$DATE.tar.gz"

echo "Starting backup at $(date)"

# Create database backup
docker exec postgres pg_dump -U benchmark_user ai_benchmark_prod > "$BACKUP_DIR/db-backup-$DATE.sql"

# Create Redis backup
docker exec redis redis-cli BGSAVE
docker cp redis:/data/dump.rdb "$BACKUP_DIR/redis-backup-$DATE.rdb"

# Create configuration backup
tar -czf "$BACKUP_DIR/config-backup-$DATE.tar.gz" -C "$(dirname "$0")/.." config/ .env.production

# Create full application backup
tar -czf "$BACKUP_DIR/$BACKUP_FILE" -C "$(dirname "$0")/.." \
    --exclude=venv \
    --exclude=.git \
    --exclude=cache \
    --exclude=logs \
    --exclude=__pycache__ \
    .

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.rdb" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
EOF

    chmod +x "$PROJECT_DIR/scripts/backup_production.sh"
    log_success "Created backup script"

    # Setup cron job for automated backups
    (crontab -l 2>/dev/null; echo "0 2 * * * $PROJECT_DIR/scripts/backup_production.sh") | crontab -
    log_success "Setup automated daily backups at 2 AM"
}

# Create systemd service
create_systemd_service() {
    log "Creating systemd service..."

    cat > /tmp/ai-benchmark-suite.service << EOF
[Unit]
Description=AI Benchmark Suite
After=docker.service
Requires=docker.service

[Service]
Type=forking
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/docker-compose -f docker-compose.production.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.production.yml down
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/ai-benchmark-suite.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable ai-benchmark-suite

    log_success "Created systemd service"
    log "Service can be controlled with: sudo systemctl start/stop/restart ai-benchmark-suite"
}

# Validate installation
validate_installation() {
    log "Validating installation..."

    cd "$PROJECT_DIR"

    # Start services
    log "Starting services..."
    docker-compose -f docker-compose.production.yml up -d

    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30

    # Health checks
    log "Performing health checks..."

    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi

    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log_success "Prometheus health check passed"
    else
        log_warning "Prometheus health check failed"
    fi

    # Check Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        log_success "Grafana health check passed"
    else
        log_warning "Grafana health check failed"
    fi

    log_success "Installation validation completed"
}

# Print final information
print_final_info() {
    log_success "AI Benchmark Suite production setup completed!"
    echo
    echo "=== Access Information ==="
    echo "API Server: http://localhost:8000"
    echo "API Documentation: http://localhost:8000/docs"
    echo "Monitoring Dashboard: python scripts/monitoring_dashboard.py"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000"
    echo
    echo "=== Default Credentials ==="
    echo "Grafana: admin / $(grep GRAFANA_ADMIN_PASSWORD .env.production | cut -d'=' -f2)"
    echo
    echo "=== Important Files ==="
    echo "Environment: .env.production"
    echo "Logs: $LOG_DIR"
    echo "Backups: $BACKUP_DIR"
    echo
    echo "=== Next Steps ==="
    echo "1. Update .env.production with your API keys"
    echo "2. Configure your domain name and SSL certificates"
    echo "3. Setup external monitoring and alerting"
    echo "4. Review security settings"
    echo "5. Run your first evaluation test"
    echo
    echo "=== Useful Commands ==="
    echo "Start services: docker-compose -f docker-compose.production.yml up -d"
    echo "Stop services: docker-compose -f docker-compose.production.yml down"
    echo "View logs: docker-compose -f docker-compose.production.yml logs -f"
    echo "Backup: ./scripts/backup_production.sh"
    echo
    log_success "Setup complete! Please review the information above."
}

# Main execution
main() {
    check_prerequisites
    create_directories
    setup_environment
    install_dependencies
    setup_docker
    initialize_database
    setup_monitoring
    setup_ssl
    setup_backup_system
    create_systemd_service
    validate_installation
    print_final_info
}

# Run main function
main "$@"