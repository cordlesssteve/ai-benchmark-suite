# Production Deployment Guide

## Overview

This guide covers deploying the AI Benchmark Suite in production environments with enterprise-grade features including monitoring, caching, analytics, and multi-model support.

## Prerequisites

### System Requirements

- **CPU**: 8+ cores recommended (16+ for high-throughput)
- **Memory**: 32GB+ RAM (64GB+ recommended)
- **Storage**: 500GB+ SSD for caching and results
- **Network**: High-bandwidth connection for model downloads
- **OS**: Linux (Ubuntu 20.04+ or CentOS 8+)

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- Git
- Redis (handled by Docker Compose)
- PostgreSQL (handled by Docker Compose)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-benchmark-suite
```

### 2. Environment Configuration

Create production environment file:

```bash
cp .env.example .env.production
```

Edit `.env.production` with your settings:

```env
# Environment
ENVIRONMENT=production
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-super-secret-key-here

# Database
POSTGRES_DB=ai_benchmark_prod
POSTGRES_USER=benchmark_user
POSTGRES_PASSWORD=secure-password-here
DATABASE_URL=postgresql://benchmark_user:secure-password-here@postgres:5432/ai_benchmark_prod

# Redis
REDIS_URL=redis://redis:6379/0

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=admin-password-here

# Resource Limits
MAX_PARALLEL_EVALUATIONS=8
MEMORY_LIMIT_GB=30
CACHE_SIZE_GB=10
```

### 3. Deploy with Docker Compose

```bash
# Production deployment
docker-compose -f docker-compose.production.yml up -d

# Check service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f api
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# API status
curl http://localhost:8000/status

# Prometheus metrics
curl http://localhost:9090/metrics

# Access Grafana dashboard
# http://localhost:3000 (admin/your-grafana-password)
```

## Service Architecture

### Core Services

1. **API Server** (Port 8000)
   - FastAPI REST API
   - WebSocket real-time updates
   - Authentication and rate limiting
   - Evaluation orchestration

2. **Database** (PostgreSQL)
   - Result persistence
   - Configuration storage
   - User management

3. **Cache** (Redis)
   - Result caching
   - Session management
   - Real-time data

4. **Monitoring Stack**
   - Prometheus (metrics collection)
   - Grafana (visualization)
   - Custom dashboard

### Service Dependencies

```
API Server ──► PostgreSQL (results)
     │     └─► Redis (cache)
     │
     └─► Model Interfaces
           ├─► HuggingFace
           ├─► OpenAI
           └─► Anthropic

Monitoring ──► Prometheus ──► Grafana
```

## Configuration Management

### Environment-Specific Configs

The system supports multiple environments:

- **Development**: `config/dev.yaml`
- **Staging**: `config/staging.yaml`
- **Production**: `config/production.yaml`

### Configuration Structure

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 300

models:
  huggingface:
    cache_dir: "/app/cache/models"
    device: "auto"
    max_memory_gb: 16

  openai:
    timeout: 60
    max_retries: 3

  anthropic:
    timeout: 60
    max_retries: 3

evaluation:
  max_parallel: 8
  timeout_seconds: 1800
  retry_attempts: 2

caching:
  enabled: true
  ttl_hours: 24
  max_size_gb: 10

monitoring:
  metrics_enabled: true
  dashboard_enabled: true
  log_level: "INFO"
```

### Secrets Management

Secrets are encrypted using the Enterprise Config Manager:

```python
from config.enterprise_config_manager import EnterpriseConfigManager

config = EnterpriseConfigManager()
config.set_secret("openai_api_key", "sk-...")
config.set_secret("anthropic_api_key", "sk-ant-...")
```

## API Usage

### Authentication

```bash
# Get authentication token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'
```

### Single Evaluation

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt-3.5-turbo",
    "task": "humaneval",
    "language": "python",
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 512
    }
  }'
```

### Suite Evaluation

```bash
curl -X POST http://localhost:8000/api/v1/evaluate/suite \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["gpt-3.5-turbo", "gpt-4"],
    "tasks": ["humaneval", "mbpp"],
    "languages": ["python", "javascript"],
    "suite_config": {
      "parallel_executions": 4,
      "use_cache": true
    }
  }'
```

### WebSocket Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/evaluations');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Evaluation update:', data);
};
```

## Monitoring and Alerting

### Prometheus Metrics

Key metrics exposed:

- `evaluation_duration_seconds` - Evaluation execution time
- `cache_hit_rate` - Cache performance
- `active_evaluations` - Current running evaluations
- `system_cpu_usage` - CPU utilization
- `system_memory_usage` - Memory utilization

### Grafana Dashboards

Pre-configured dashboards:

1. **System Overview** - CPU, memory, disk usage
2. **Evaluation Performance** - Success rates, timing
3. **Cache Analytics** - Hit rates, size, efficiency
4. **API Metrics** - Request rates, response times

### Alert Rules

Example Prometheus alert rules:

```yaml
groups:
- name: ai-benchmark-alerts
  rules:
  - alert: HighCPUUsage
    expr: system_cpu_usage > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"

  - alert: EvaluationFailureRate
    expr: rate(evaluation_failures_total[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High evaluation failure rate"
```

## Scaling and Performance

### Horizontal Scaling

Scale API servers:

```bash
docker-compose -f docker-compose.production.yml up -d --scale api=3
```

### Vertical Scaling

Adjust resource limits in `docker-compose.production.yml`:

```yaml
api:
  deploy:
    resources:
      limits:
        cpus: '4.0'
        memory: 8G
      reservations:
        cpus: '2.0'
        memory: 4G
```

### Performance Optimization

1. **Enable Caching**
   - Set `CACHE_ENABLED=true`
   - Increase `CACHE_SIZE_GB` based on available memory

2. **Parallel Execution**
   - Adjust `MAX_PARALLEL_EVALUATIONS` based on CPU cores
   - Monitor resource usage to find optimal value

3. **Model Optimization**
   - Use quantized models for HuggingFace
   - Enable model caching
   - Configure optimal batch sizes

## Security

### Authentication

- JWT-based authentication
- Role-based access control
- API key management

### Network Security

- TLS/SSL encryption (configure reverse proxy)
- Network isolation between services
- Firewall configuration

### Secrets Security

- Encrypted secrets storage
- Environment variable isolation
- Key rotation procedures

## Backup and Recovery

### Database Backup

```bash
# Manual backup
docker exec postgres pg_dump -U benchmark_user ai_benchmark_prod > backup.sql

# Automated backup script
scripts/backup_database.sh
```

### Redis Backup

```bash
# Redis persistence is configured in docker-compose.production.yml
# Manual backup
docker exec redis redis-cli BGSAVE
```

### Configuration Backup

```bash
# Backup configuration and secrets
tar -czf config-backup-$(date +%Y%m%d).tar.gz config/ .env.production
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   # Check logs
   docker-compose -f docker-compose.production.yml logs service-name

   # Check resource usage
   docker stats
   ```

2. **High Memory Usage**
   ```bash
   # Monitor memory
   docker exec api python -c "
   from model_interfaces.memory_optimizer import MemoryOptimizer
   optimizer = MemoryOptimizer()
   print(optimizer.get_memory_stats())
   "
   ```

3. **Evaluation Timeouts**
   - Check network connectivity
   - Verify model availability
   - Increase timeout values in configuration

4. **Cache Issues**
   ```bash
   # Clear Redis cache
   docker exec redis redis-cli FLUSHALL

   # Check cache stats
   curl http://localhost:8000/api/v1/cache/stats
   ```

### Log Analysis

Important log locations:

- API logs: `docker-compose logs api`
- Database logs: `docker-compose logs postgres`
- Redis logs: `docker-compose logs redis`
- System logs: `/var/log/ai-benchmark/`

### Performance Debugging

```bash
# Profile API performance
curl http://localhost:8000/api/v1/debug/performance

# Monitor real-time metrics
curl http://localhost:8000/metrics

# Check evaluation queue
curl http://localhost:8000/api/v1/queue/status
```

## Maintenance

### Regular Tasks

1. **Weekly**
   - Database maintenance and optimization
   - Log rotation and cleanup
   - Cache cleanup and optimization

2. **Monthly**
   - Security updates
   - Performance review
   - Backup verification

3. **Quarterly**
   - Capacity planning review
   - Configuration audit
   - Security assessment

### Update Procedure

1. **Backup Current State**
   ```bash
   scripts/backup_production.sh
   ```

2. **Update Code**
   ```bash
   git pull origin main
   ```

3. **Update Services**
   ```bash
   docker-compose -f docker-compose.production.yml pull
   docker-compose -f docker-compose.production.yml up -d
   ```

4. **Verify Update**
   ```bash
   scripts/verify_deployment.sh
   ```

## Support and Contact

For production support:

- **Documentation**: `docs/` directory
- **Issue Tracking**: GitHub Issues
- **Performance Monitoring**: Grafana dashboard
- **Health Checks**: `/health` endpoint